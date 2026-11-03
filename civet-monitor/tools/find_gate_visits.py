import argparse
import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from civetwatch.config import load_config
from civetwatch.pipeline import Pipeline
from civetwatch.run import CachedSource, _file_start


def at_gate(obs, d23_cam03, near: float) -> bool:
    for o in obs:
        if o.track_age < 2 or o.place.kind not in ("floor", "door"):
            continue
        if o.place.zone_id == "D23" and o.cam_id in ("CAM02", "CAM03"):
            return True
        if o.cam_id == "CAM03" and o.place.kind == "floor":
            pt = (0.5 * (o.box[0] + o.box[2]), float(o.box[3]))
            if cv2.pointPolygonTest(d23_cam03, pt, True) >= -near:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/april5_24h.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the files")
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--near", type=float, default=80.0, help="CAM03 floor within this many px of D23 counts")
    ap.add_argument("--gap", type=float, default=2.0, help="join at-gate ticks closer than this (s)")
    ap.add_argument("--min", type=float, default=4.0, help="drop visits shorter than this (s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", action="store_true", help="cut a 4-camera clip per visit next to --out")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["storage"]["room_log_csv"] = str(Path(a.out).with_suffix(".room_log.csv").resolve())
    cfg["storage"]["db_path"] = str(Path(a.out).with_suffix(".sqlite").resolve())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d23 = next(d for d in cfg["zones"]["cameras"]["CAM03"]["doors"] if d["id"] == "D23")
    import numpy as np
    d23_cam03 = np.asarray(d23["poly"], np.float32).reshape(-1, 1, 2)

    p = Pipeline(cfg)
    st = {c: CachedSource(c, str(Path(a.cache) / f"{c}.npz"), _file_start(cc)) for c, cc in cfg["cameras"].items()}
    t0 = min(s.file_start for s in st.values())
    ticks: list[tuple[datetime, str]] = []
    t, dt = a.start, p.active_dt
    while t <= a.start + a.duration:
        ts = t0 + timedelta(seconds=t)
        frames = {c: s.frame_at((ts - s.file_start).total_seconds()) for c, s in st.items()}
        if all(s.finished for s in st.values()):
            break
        state, obs = p.tick(ts, frames, {c for c, s in st.items() if s.finished})
        if at_gate(obs, d23_cam03, a.near):
            ticks.append((ts, state.room))
        t += dt
    p.close()

    visits: list[list] = []
    for ts, room in ticks:
        if visits and (ts - visits[-1][1]).total_seconds() <= a.gap:
            visits[-1][1] = ts
        else:
            visits.append([ts, ts, room])
    visits = [v for v in visits if (v[1] - v[0]).total_seconds() >= a.min]
    changes = [c for c in p.handoff.changes if c["state"] == "VISIBLE"]
    rows = []
    for k, (s, e, room) in enumerate(visits, 1):
        logged = [f"{c['ts'][11:19]} {c['room']}" for c in changes
                  if s - timedelta(seconds=3) <= datetime.fromisoformat(c["ts"]) <= e + timedelta(seconds=3)]
        rows.append({"visit": k, "start": f"{s:%H:%M:%S}", "end": f"{e:%H:%M:%S}",
                     "seconds": round((e - s).total_seconds()), "clip": "",
                     "room_before": room, "logged": "; ".join(logged)})
    if a.clips:
        for r in rows:
            s = datetime.strptime(r["start"], "%H:%M:%S") - timedelta(seconds=5)
            e = datetime.strptime(r["end"], "%H:%M:%S") + timedelta(seconds=5)
            clip = Path(a.out).parent / "clips" / f"gate_{r['visit']:03d}_{r['start'].replace(':', '')}.mp4"
            subprocess.run([sys.executable, str(ROOT / "tools" / "mosaic_video.py"), "--config", a.config,
                            "--start", f"{s:%H:%M:%S}", "--end", f"{e:%H:%M:%S}", "--out", str(clip)],
                           check=True, capture_output=True)
            r["clip"] = clip.name
            print("clip", clip.name, flush=True)
    sheet = ["visit", "start", "end", "seconds", "clip"]
    for path, cols in ((Path(a.out), sheet), (Path(a.out).with_suffix(".model.csv"), sheet[:4] + ["room_before", "logged"])):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} gate visits of {a.min:.0f} s or more -> {a.out}")


if __name__ == "__main__":
    main()
