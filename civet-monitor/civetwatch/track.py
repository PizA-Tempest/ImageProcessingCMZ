from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


class Dets:

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray | None = None):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), np.float32) if cls is None else np.asarray(cls, np.float32).reshape(-1)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


class Tracker:

    def __init__(self, cfg: dict, fps: float):
        buffer_frames = max(1, int(round(cfg.get("track_buffer_s", 1.0) * fps)))
        self.args = SimpleNamespace(
            track_high_thresh=cfg.get("track_high_thresh", 0.5),
            track_low_thresh=cfg.get("track_low_thresh", 0.1),
            new_track_thresh=cfg.get("new_track_thresh", 0.5),
            match_thresh=cfg.get("match_thresh", 0.8),
            track_buffer=buffer_frames,
            fuse_score=True,
        )
        self.bt = BYTETracker(self.args)
        self.age: dict[int, int] = {}

    def update(self, xyxy, conf, kpts=None) -> list[tuple[tuple, int, float, int, np.ndarray | None]]:
        out = self.bt.update(Dets(xyxy, conf))
        res = []
        alive = set()
        for row in out:
            x1, y1, x2, y2, tid, score = row[:6]
            tid, idx = int(tid), int(row[7])
            kp = kpts[idx] if kpts is not None and 0 <= idx < len(kpts) else None
            alive.add(tid)
            self.age[tid] = self.age.get(tid, 0) + 1
            res.append(((float(x1), float(y1), float(x2), float(y2)), tid, float(score), self.age[tid], kp))
        for tid in list(self.age):
            if tid not in alive and not any(t.track_id == tid for t in self.bt.lost_stracks):
                del self.age[tid]
        return res

    def reset(self):
        self.bt = BYTETracker(self.args)
        self.age.clear()
