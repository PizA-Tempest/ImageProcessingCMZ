import argparse, sys, time
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civetwatch.config import load_config
from civetwatch.detect import MotionDetector, YoloDetector
from civetwatch.preprocess import Preprocessor
from civetwatch.sources import FileStepper
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/system.yaml")
ap.add_argument("--cam", required=True)
ap.add_argument("--out", default="outputs/cache")
ap.add_argument("--start", type=float, default=0.0, help="seconds into the file (times stay file-relative)")
ap.add_argument("--duration", type=float, default=0, help="seconds to cache from --start (0 = to the end)")
ap.add_argument("--detector", choices=["motion", "yolo"], default="motion")
ap.add_argument("--windows", help="CSV with start,end clock times: cache only these (no --resume)")
ap.add_argument("--pad", type=float, default=5.0, help="seconds added on both sides of each --windows row")
ap.add_argument("--resume", action="store_true",
                help="continue from <out>/<cam>.part.npz (saved every 2250 frames); the motion "
                     "detector restarts its background there, YOLO has no state")
a = ap.parse_args()
cfg = load_config(a.config)
c = cfg["cameras"][a.cam]
pre = Preprocessor(cfg["preprocess"])
md = MotionDetector(cfg["motion"], pre.overlay_mask((1080, 1920)))
yolo = YoloDetector(cfg["detector"], cfg.get("_base")) if a.detector == "yolo" else None
dt = 1.0 / float(cfg["rates"]["active_fps"])
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
part = out / f"{a.cam}.part.npz"


def save(path: Path) -> None:
    extra = {"kpts": np.array(kps, np.float32).reshape(-1, 6, 3)} if yolo is not None else {}
    np.savez_compressed(path, t=np.array(ts, np.float64), dets=np.array(rows, np.float32).reshape(-1, 6), dt=dt, **extra)


def detect(f) -> None:
    g = pre.gray(f.image)
    if yolo is None:
        xyxy, conf = md.update(g, f_t)
        k = np.zeros((len(conf), 6, 3), np.float32)
    else:
        xyxy, conf, k = yolo.detect(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), keypoints=True)
    for b, s, kp in zip(xyxy, conf, k):
        rows.append((len(ts), *b, s))
        kps.append(kp)
    ts.append(f_t)


ts, rows, kps = [], [], []
if a.windows:
    import csv
    fs0 = datetime.fromisoformat(c["file_start"])
    with open(a.windows, newline="", encoding="utf-8") as fh:
        wins = [(r["start"], r["end"]) for r in csv.DictReader(fh)]
    w0 = time.time()
    for s0, e0 in wins:
        day = fs0.date().isoformat()
        lo = (datetime.fromisoformat(f"{day}T{s0}") - fs0).total_seconds() - a.pad
        hi = (datetime.fromisoformat(f"{day}T{e0}") - fs0).total_seconds() + a.pad
        if ts and lo <= ts[-1]:
            lo = ts[-1] + dt
        fs = FileStepper(a.cam, c["source"], fs0, max(0.0, lo))
        f_t = lo
        while f_t <= hi:
            f = fs.frame_at(f_t)
            if f is None or fs.finished:
                break
            detect(f)
            f_t += dt
        fs.release()
    save(out / f"{a.cam}.npz")
    print(a.cam, "done", len(wins), "windows", len(ts), "frames", len(rows), "dets", f"{time.time() - w0:.0f}s")
    sys.exit()
t, i, w0 = a.start, 0, time.time()
if a.resume and part.exists():
    with np.load(part) as z:
        ts, rows = z["t"].tolist(), [tuple(r) for r in z["dets"]]
        kps = list(z["kpts"]) if "kpts" in z else [np.zeros((6, 3), np.float32)] * len(rows)
    i, t = len(ts), (ts[-1] + dt if ts else a.start)
    print(a.cam, f"resuming at {t:.0f}s ({i} frames)", flush=True)
t_start = t
fs = FileStepper(a.cam, c["source"], datetime.fromisoformat(c["file_start"]), t)
while not fs.finished and (not a.duration or t < a.start + a.duration):
    f = fs.frame_at(t)
    if f is None or fs.finished:
        break
    f_t = t
    detect(f)
    i += 1
    t += dt
    if i % 2250 == 0:
        save(part)
        print(a.cam, f"{t:.0f}s {(t - t_start) / (time.time() - w0):.1f}x", flush=True)
save(out / f"{a.cam}.npz")
part.unlink(missing_ok=True)
print(a.cam, "done", i, "frames", len(rows), "dets", f"{time.time() - w0:.0f}s")
