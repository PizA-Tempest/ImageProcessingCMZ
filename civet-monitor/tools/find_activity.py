import argparse
import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remux import ffmpeg_exe

W, H = 192, 108


def frames(src: str, fps: float, from_s: float = 0.0, to_s: float | None = None):
    span = ["-ss", f"{from_s:.3f}"] if from_s else []
    if to_s is not None:
        span += ["-t", f"{to_s - from_s:.3f}"]
    cmd = [ffmpeg_exe(), "-v", "error", *span, "-i", src, "-vf", f"fps={fps},scale={W}:{H},format=gray",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W)
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--start", default=None, help="wall-clock time of the first frame (ISO)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--thr", type=int, default=25)
    ap.add_argument("--min-area", type=int, default=30, help="pixels at 192x108")
    ap.add_argument("--gap", type=float, default=5.0, help="merge bursts closer than this (s)")
    ap.add_argument("--min-len", type=float, default=1.0)
    ap.add_argument("--from-s", type=float, default=0.0, help="scan from this many seconds into the file")
    ap.add_argument("--to-s", type=float, default=None, help="stop at this many seconds into the file")
    a = ap.parse_args()
    t0 = datetime.fromisoformat(a.start) if a.start else None

    mask = np.ones((H, W), bool)
    mask[:8, :70] = False
    mask[93:, 120:170] = False
    hist, rows = [], []
    bg = None
    for i, f in enumerate(frames(a.src, a.fps, a.from_s, a.to_s)):
        t = a.from_s + i / a.fps
        hist.append(f)
        if bg is None or i % 60 == 0:
            bg = np.median(np.stack(hist[-120:][::4]), axis=0).astype(np.int16)
        d = ((np.abs(f.astype(np.int16) - bg) > a.thr) & mask).astype(np.uint8)
        d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(d)
        if n > 1:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            rows.append((t, int(st[k, cv2.CC_STAT_AREA]), cen[k][0] * 1920 / W, cen[k][1] * 1080 / H))
        else:
            rows.append((t, 0, -1, -1))
        if len(hist) > 240:
            hist = hist[-240:]

    bursts, cur = [], None
    for t, area, x, y in rows:
        if area >= a.min_area:
            if cur and t - cur["end"] <= a.gap:
                cur["end"] = t
                cur["areas"].append(area)
                if area > cur["peak"]:
                    cur.update(peak=area, x=x, y=y)
            else:
                if cur:
                    bursts.append(cur)
                cur = {"start": t, "end": t, "areas": [area], "peak": area, "x": x, "y": y}
    if cur:
        bursts.append(cur)
    bursts = [b for b in bursts if b["end"] - b["start"] >= a.min_len]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_s", "end_s", "start_time", "peak_area", "mean_area", "peak_x", "peak_y"])
        for b in bursts:
            st = (t0 + timedelta(seconds=b["start"])).isoformat() if t0 else ""
            w.writerow([f"{b['start']:.1f}", f"{b['end']:.1f}", st, b["peak"],
                        f"{np.mean(b['areas']):.0f}", f"{b['x']:.0f}", f"{b['y']:.0f}"])
    total = sum(b["end"] - b["start"] for b in bursts)
    print(f"{len(bursts)} bursts, {total:.0f} s active of {len(rows) / a.fps:.0f} s "
          f"(from {a.from_s:.0f} s) -> {a.out}")


if __name__ == "__main__":
    main()
