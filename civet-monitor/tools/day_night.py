import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civetwatch.preprocess import Preprocessor


def saturation(bgr: np.ndarray) -> float:
    small = cv2.resize(bgr, (64, 36), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--start", required=True, help="wall-clock time of the first frame (ISO)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=60.0, help="seconds between samples")
    a = ap.parse_args()
    t0 = datetime.fromisoformat(a.start)

    cap = cv2.VideoCapture(a.src, cv2.CAP_FFMPEG)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / 15.0
    rows: list[tuple[float, float, bool]] = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        if ok:
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            rows.append((pos, saturation(img), Preprocessor.is_night(img)))
        t += a.step

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pos_s", "time", "saturation", "night"])
        for pos, sat, night in rows:
            w.writerow([f"{pos:.1f}", (t0 + timedelta(seconds=pos)).strftime("%H:%M:%S"), f"{sat:.1f}", int(night)])
    for (p0, _, n0), (p1, _, n1) in zip(rows, rows[1:]):
        if n0 != n1:
            print(f"{'night -> day' if n0 else 'day -> night'} between "
                  f"{(t0 + timedelta(seconds=p0)):%H:%M:%S} and {(t0 + timedelta(seconds=p1)):%H:%M:%S}")


if __name__ == "__main__":
    main()
