import argparse
import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civetwatch.preprocess import Preprocessor


def clock_s(text: str) -> float:
    parts = [float(x) for x in text.split(":")]
    return parts[0] * 3600 + parts[1] * 60 + (parts[2] if len(parts) > 2 else 0.0)


def parse_windows(specs: list[str] | None, t0: datetime) -> list[tuple[float, float]]:
    day0 = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    off = (t0 - day0).total_seconds()
    out = []
    for spec in specs or []:
        a, b = spec.split("-")
        out.append((clock_s(a) - off, clock_s(b) - off))
    return out


def inside(t: float, windows: list[tuple[float, float]]) -> bool:
    return not windows or any(s <= t < e for s, e in windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--cam", required=True)
    ap.add_argument("--start", required=True, help="wall-clock time of the first frame (ISO)")
    ap.add_argument("--bursts", help="CSV from find_activity.py")
    ap.add_argument("--times", help="comma-separated seconds to extract instead")
    ap.add_argument("--per-burst", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--quiet", type=int, default=20, help="extra frames from inactive periods")
    ap.add_argument("--min-gap", type=float, default=1.0, help="seconds between frames")
    ap.add_argument("--color", action="store_true", help="keep colour (default: grayscale x3)")
    ap.add_argument("--out", default="labelling/images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", action="append", help="HH:MM[:SS]-HH:MM[:SS], repeatable")
    ap.add_argument("--manifest", help="CSV to append one row per frame to")
    ap.add_argument("--split", default="", help="split name written to the manifest")
    ap.add_argument("--kind", default="times", help="kind written to the manifest for --times frames")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    t0 = datetime.fromisoformat(a.start)
    windows = parse_windows(a.window, t0)

    cap = cv2.VideoCapture(a.src, cv2.CAP_FFMPEG)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = n_frames / 15.0 if n_frames > 0 else 1e9

    times: list[tuple[float, str]] = []
    if a.times:
        times = [(float(x), a.kind) for x in a.times.split(",")]
    elif a.bursts:
        spans = []
        with open(a.bursts) as fh:
            for r in csv.DictReader(fh):
                s, e = float(r["start_s"]), float(r["end_s"])
                spans.append((s, e))
                k = max(1, min(a.per_burst, int((e - s) / a.min_gap) + 1))
                times += [(t, "burst") for t in (np.linspace(s, e, k) if k > 1 else [s]) if inside(t, windows)]
        pool = windows or [(0.0, min(duration, spans[-1][1] + 600 if spans else 3600))]
        weights = [e - s for s, e in pool]
        n_burst = len(times)
        tries = 0
        while a.quiet and tries < 10000 and len(times) < max(a.max_frames, n_burst) + a.quiet:
            tries += 1
            s, e = rnd.choices(pool, weights)[0]
            t = rnd.uniform(s, e)
            if all(not (bs - 5 <= t <= be + 5) for bs, be in spans):
                times.append((t, "quiet"))
                a.quiet -= 1
    times.sort()
    kept, last = [], -1e9
    for t, kind in times:
        if t - last >= a.min_gap:
            kept.append((t, kind))
            last = t
    if len(kept) > a.max_frames:
        kept = sorted(rnd.sample(kept, a.max_frames))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    manifest = []
    n = 0
    for t, kind in kept:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        if not ok:
            continue
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not inside(pos, windows):
            continue
        night = Preprocessor.is_night(img)
        if not a.color:
            g = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        ts = t0 + timedelta(seconds=pos)
        name = f"{a.cam}_{ts.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
        if (out / name).exists():
            continue
        cv2.imwrite(str(out / name), img)
        manifest.append([name, a.cam, ts.isoformat(timespec="milliseconds"), a.split, kind, int(night)])
        n += 1
    if a.manifest:
        new = not Path(a.manifest).exists()
        with open(a.manifest, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["file", "cam", "time", "split", "kind", "night"])
            w.writerows(manifest)
    print(f"{n} frames -> {out}")


if __name__ == "__main__":
    main()
