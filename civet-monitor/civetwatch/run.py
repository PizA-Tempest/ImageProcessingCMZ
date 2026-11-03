from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import load_config
from .pipeline import Pipeline
from .sources import CameraReader, FileStepper, is_live


def _file_start(c: dict) -> datetime:
    return datetime.fromisoformat(c["file_start"]) if c.get("file_start") else datetime.now()


def _writer(path: str, fps: float, size):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def _report(p: Pipeline, wall: float, sim: float):
    print(f"\nprocessed {sim:.0f} s of video in {wall:.0f} s wall ({sim / max(wall, 1e-6):.1f}x real time)")
    for cid, w in sorted(p.workers.items()):
        s = w.stats
        per = 1000 * s.seconds / max(s.processed, 1)
        print(f"  {cid}: {s.processed} frames ({s.processed / max(sim, 1e-6):.2f}/s), "
              f"detector {s.detector_calls}, dets {s.detections}, room obs {s.observations}, {per:.1f} ms/frame")
    print(f"  state changes: {len(p.handoff.changes)}")


def run_offline(cfg: dict, args) -> Pipeline:
    p = Pipeline(cfg, args.detector)
    if args.cache:
        steppers = {cid: CachedSource(cid, str(Path(args.cache) / f"{cid}.npz"), _file_start(c))
                    for cid, c in cfg["cameras"].items()}
    else:
        steppers = {cid: FileStepper(cid, c["source"], _file_start(c), args.start)
                    for cid, c in cfg["cameras"].items()}
    t0 = min(s.file_start for s in steppers.values()) + timedelta(seconds=args.start)
    dt = p.active_dt
    end = args.start + (args.duration if args.duration else 1e9)
    vw = None
    t = args.start
    wall0 = time.perf_counter()
    last_print = wall0
    while t <= end:
        frames = {}
        for cid, s in steppers.items():
            off = (t0 + timedelta(seconds=t - args.start) - s.file_start).total_seconds()
            frames[cid] = s.frame_at(off)
        if all(s.finished for s in steppers.values()):
            break
        ts = t0 + timedelta(seconds=t - args.start)
        lost = {cid for cid, s in steppers.items() if s.finished}
        p.tick(ts, frames, lost)
        if args.debug_video:
            m = p.draw(0.25)
            if vw is None:
                vw = _writer(args.debug_video, 1.0 / dt, (m.shape[1], m.shape[0]))
            cv2.putText(m, ts.strftime("%H:%M:%S.%f")[:-4], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            vw.write(m)
        t += dt
        if not args.cache and time.perf_counter() - last_print > 30:
            last_print = time.perf_counter()
            s = p.handoff.s
            print(f"  {ts:%H:%M:%S}  {s.state} {s.room} {s.reason or ''}  "
                  f"({(t - args.start) / (last_print - wall0):.1f}x)", flush=True)
    for s in steppers.values():
        s.release()
    if vw:
        vw.release()
    _report(p, time.perf_counter() - wall0, t - args.start)
    return p


class CachedSource:

    def __init__(self, cam_id: str, path: str, file_start: datetime):
        import numpy as np
        z = np.load(path)
        self.cam_id, self.file_start = cam_id, file_start
        self.t = z["t"]
        d = z["dets"]
        k = z["kpts"] if "kpts" in z else None
        self.by_idx: dict[int, tuple] = {}
        for i in np.unique(d[:, 0]).astype(int):
            m = d[:, 0] == i
            r = d[m]
            self.by_idx[i] = (r[:, 1:5].copy(), r[:, 5].copy()) + ((k[m].copy(),) if k is not None else ())
        self.empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
        self.finished = False

    def frame_at(self, t_s: float):
        from .sources import Frame
        i = int(np.searchsorted(self.t, t_s + 1e-6)) - 1
        if i < 0:
            return None
        if i >= len(self.t) - 1:
            self.finished = True
        return Frame(self.cam_id, i + 1, self.file_start + timedelta(seconds=float(self.t[i])), None,
                     self.by_idx.get(i, self.empty))

    def release(self):
        pass


def run_live(cfg: dict, args) -> Pipeline:
    p = Pipeline(cfg, args.detector)
    rp, rd = cfg.get("replay", {}), cfg.get("reader", {})
    readers = {}
    for cid, c in cfg["cameras"].items():
        r = CameraReader(cid, c["source"], _file_start(c), replay=rp.get("enabled", True),
                         speed=args.speed or rp.get("speed", 1.0),
                         start_offset_s=args.start or rp.get("start_offset_s", 0),
                         reconnect_initial_s=rd.get("reconnect_initial_s", 1.0),
                         reconnect_max_s=rd.get("reconnect_max_s", 30.0))
        r.start()
        readers[cid] = r
    lost_after = float(rd.get("stream_lost_after_s", 5.0))
    files = not any(is_live(c["source"]) for c in cfg["cameras"].values())
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    vw = None
    wall0 = time.perf_counter()
    sim0 = None
    dt = p.active_dt
    next_t = time.monotonic()
    while not stop["flag"]:
        now = time.monotonic()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += dt
        frames = {cid: r.latest() for cid, r in readers.items()}
        lost = {cid for cid, r in readers.items()
                if r.latest() is None or time.monotonic() - r.last_frame_wall > lost_after}
        if files:
            got = [f for f in frames.values() if f is not None]
            if not got:
                continue
            ts = max(f.ts for f in got)
            if all(r.finished for r in readers.values()):
                break
        else:
            ts = datetime.now()
        sim0 = sim0 or ts
        p.tick(ts, frames, lost)
        if args.debug_video:
            m = p.draw(0.25)
            if vw is None:
                vw = _writer(args.debug_video, 1.0 / dt, (m.shape[1], m.shape[0]))
            vw.write(m)
        if args.duration and (ts - sim0).total_seconds() >= args.duration:
            break
    for r in readers.values():
        r.stop()
    if vw:
        vw.release()
    _report(p, time.perf_counter() - wall0, (ts - sim0).total_seconds() if sim0 else 0)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="civetwatch phase 1: where is A01")
    ap.add_argument("--config", default="config/system.yaml")
    ap.add_argument("--offline", action="store_true", help="step recorded files as fast as possible")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the files")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to process (0 = all)")
    ap.add_argument("--detector", choices=["motion", "yolo"], default=None)
    ap.add_argument("--debug-video", default=None)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--room-log", default=None, help="override storage.room_log_csv")
    ap.add_argument("--cache", default=None, help="offline: replay detections from tools/cache_dets.py")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a setting for this run, e.g. handoff.min_dwell_s=3 (YAML value)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for kv in args.set:
        key, val = kv.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for k in path:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(val)
    if args.room_log:
        cfg["storage"]["room_log_csv"] = str(Path(args.room_log).resolve())
    p = run_offline(cfg, args) if args.offline else run_live(cfg, args)
    p.close()
    print(f"room log: {cfg['storage']['room_log_csv']}")


if __name__ == "__main__":
    main()
