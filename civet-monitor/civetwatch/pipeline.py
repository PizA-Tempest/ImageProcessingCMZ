from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from .db import RoomLog, Store
from .detect import MotionDetector, YoloDetector
from .handoff import HandoffConfig, HandoffManager, Observation, HIDDEN, PENDING, UNKNOWN, VISIBLE
from .preprocess import Preprocessor
from .sources import Frame
from .track import Tracker
from .zones import CameraZones, load_zones


@dataclass
class CamStats:
    processed: int = 0
    detector_calls: int = 0
    detections: int = 0
    observations: int = 0
    seconds: float = 0.0


class CameraWorker:

    def __init__(self, cam_id: str, cfg: dict, zones: CameraZones, overlay_mask: np.ndarray):
        self.cam_id = cam_id
        self.room = cfg["cameras"][cam_id]["room"]
        self.zones = zones
        self.motion = MotionDetector(cfg["motion"], overlay_mask)
        fps = float(cfg["rates"]["active_fps"])
        self.tracker = Tracker(cfg["tracker"], fps)
        self.last_t: float | None = None
        self.next_due: float | None = None
        self.last_obs: list[Observation] = []
        self.last_raw: list[Observation] = []
        self.last_tracks: list = []
        self.last_frame: Frame | None = None
        self.last_seq = -1
        self.stats = CamStats()

    def observations(self, tracks) -> list[Observation]:
        out = []
        for box, tid, score, age, kp in tracks:
            pt = self.zones.ground_point(box)
            place = self.zones.place(pt)
            out.append(Observation(self.cam_id, self.room, tid, score, box, place, age,
                                   self.zones.hind_room(pt, place, kp), kp))
        return out

    def raw_observations(self, xyxy, conf, kpts) -> list[Observation]:
        out = []
        for i, (box, score) in enumerate(zip(xyxy, conf)):
            place = self.zones.place(self.zones.ground_point(box))
            out.append(Observation(self.cam_id, self.room, -1, float(score), tuple(box), place, 0, None,
                                   None if kpts is None else kpts[i]))
        return out


class Pipeline:
    def __init__(self, cfg: dict, detector_kind: str | None = None, write_outputs: bool = True):
        self.cfg = cfg
        self.pre = Preprocessor(cfg["preprocess"])
        self.zones = load_zones(cfg)
        mask = self.pre.overlay_mask((1080, 1920))
        self.workers = {cid: CameraWorker(cid, cfg, self.zones[cid], mask) for cid in cfg["cameras"]}
        self.room_cams: dict[str, list[str]] = {}
        for cid, c in cfg["cameras"].items():
            self.room_cams.setdefault(c["room"], []).append(cid)
        h = cfg["handoff"]
        self.handoff = HandoffManager(cfg["adjacency"], HandoffConfig(
            t_handoff_s=float(h.get("t_handoff_s", 10.0)),
            lost_after_s=float(h.get("lost_after_s", 1.0)),
            recover_after_s=float(h.get("recover_after_s", 5.0)),
            min_track_age=int(h.get("min_track_age", 2)),
            door_min_age=int(h.get("door_min_age", 4)),
            min_dwell_s=float(h.get("min_dwell_s", 2.0)),
            hind_window_s=float(h.get("hind_window_s", 0.0)),
            hind_min_votes=int(h.get("hind_min_votes", 3)),
            door_track_s=float(h.get("door_track_s", 0.0)),
        ))
        self.kind = detector_kind or cfg["detector"].get("kind", "motion")
        self.yolo = YoloDetector(cfg["detector"], cfg.get("_base")) if self.kind == "yolo" else None
        self.active_dt = 1.0 / float(cfg["rates"]["active_fps"])
        self.idle_dt = 1.0 / float(cfg["rates"]["idle_fps"])
        self.obs_hold_s = self.idle_dt + 0.05
        self.write_outputs = write_outputs
        self.room_log = RoomLog(cfg["storage"]["room_log_csv"]) if write_outputs else None
        self.store = Store(cfg["storage"]["db_path"], cfg) if write_outputs else None
        self._logged = 0
        self.behaviour = None
        bw = cfg.get("behaviour", {}).get("weights")
        if bw:
            from pathlib import Path
            from .behaviour import BehaviourRunner
            w = Path(bw) if Path(bw).is_absolute() else Path(cfg.get("_base") or ".") / bw
            if w.exists():
                self.behaviour = BehaviourRunner(str(w), self.zones)
        self.t0: datetime | None = None
        self.ticks = 0
        self.last_raw: list[Observation] = []

    def active_cams(self) -> set[str]:
        s = self.handoff.s
        if s.state == UNKNOWN or s.room is None:
            return set(self.workers)
        cams = set(self.room_cams.get(s.room, []))
        if s.state == PENDING and s.to_room:
            cams |= set(self.room_cams.get(s.to_room, []))
        return cams

    def _due(self, w: CameraWorker, t: float, active: bool) -> bool:
        dt = self.active_dt if active else self.idle_dt
        if w.next_due is None or w.next_due > t + dt or t - w.next_due > dt:
            w.next_due = t
        if t + 1e-6 >= w.next_due:
            w.next_due += dt
            return True
        return False

    def tick(self, ts: datetime, frames: dict[str, Frame | None], lost_cams: set[str] | None = None):
        if self.t0 is None:
            self.t0 = ts
        t = (ts - self.t0).total_seconds()
        lost_cams = lost_cams or set()
        active = self.active_cams()

        todo: list[tuple[CameraWorker, Frame]] = []
        for cid, w in self.workers.items():
            f = frames.get(cid)
            if f is None or cid in lost_cams or f.seq == w.last_seq:
                continue
            if self._due(w, t, cid in active):
                todo.append((w, f))

        yolo_jobs = []
        for w, f in todo:
            t1 = time.perf_counter()
            w.last_seq = f.seq
            w.last_t = t
            w.last_frame = f
            w.stats.processed += 1
            if f.dets is not None:
                w._dets = f.dets
                w.stats.detector_calls += 1
                continue
            g = self.pre.gray(f.image)
            mdet = w.motion.update(g, t)
            if self.kind == "motion":
                w._dets = mdet
                w.stats.detector_calls += 1
            else:
                need = (w.cam_id in active) or w.motion.awake or bool(w.last_tracks)
                w._dets = None
                if need:
                    yolo_jobs.append((w, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)))
                else:
                    w._dets = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
            w.stats.seconds += time.perf_counter() - t1

        if yolo_jobs:
            t1 = time.perf_counter()
            res = self.yolo.detect_batch([img for _, img in yolo_jobs])
            dt = (time.perf_counter() - t1) / len(yolo_jobs)
            for (w, _), r in zip(yolo_jobs, res):
                w._dets = r
                w.stats.detector_calls += 1
                w.stats.seconds += dt

        for w, f in todo:
            xyxy, conf, *kp = w._dets
            w.stats.detections += len(conf)
            tracks = w.tracker.update(xyxy, conf, kp[0] if kp else None)
            w.last_tracks = tracks
            w.last_obs = w.observations(tracks)
            w.last_raw = w.raw_observations(xyxy, conf, kp[0] if kp else None)
            w.stats.observations += sum(o.place.kind in ("floor", "door") for o in w.last_obs)

        obs: list[Observation] = []
        self.last_raw = []
        for w in self.workers.values():
            if w.last_t is not None and t - w.last_t <= self.obs_hold_s and w.cam_id not in lost_cams:
                obs.extend(w.last_obs)
                self.last_raw.extend(w.last_raw)

        lost_rooms = {r for r, cams in self.room_cams.items() if all(c in lost_cams for c in cams)}
        state = self.handoff.update(ts, obs, lost_rooms)
        if self.behaviour:
            bout = self.behaviour.update(ts, state, obs, self.last_raw)
            if bout and self.store:
                self.store.add_event(bout.camera or "", bout.label, bout.start, bout.end, bout.conf)
        self.ticks += 1
        self._flush()
        return state, obs

    def _flush(self):
        new = self.handoff.changes[self._logged:]
        if new and self.room_log:
            self.room_log.write(new)
        self._logged = len(self.handoff.changes)

    def close(self):
        self._flush()
        last = self.behaviour.bouts.close() if self.behaviour else None
        if last and self.store:
            self.store.add_event(last.camera or "", last.label, last.start, last.end, last.conf)
        if self.room_log:
            self.room_log.close()
        if self.store:
            self.store.close()

    def draw(self, scale: float = 0.5) -> np.ndarray:
        tiles = []
        s = self.handoff.s
        for cid in sorted(self.workers):
            w = self.workers[cid]
            if w.last_frame is None or w.last_frame.image is None:
                tiles.append(np.zeros((int(1080 * scale), int(1920 * scale), 3), np.uint8))
                continue
            img = w.last_frame.image.copy()
            z = w.zones
            cv2.polylines(img, [z.floor.astype(np.int32)], True, (0, 200, 0), 2)
            for did, to, poly in z.doors:
                cv2.polylines(img, [poly.astype(np.int32)], True, (0, 200, 255), 2)
            for o in w.last_obs:
                x1, y1, x2, y2 = map(int, o.box)
                col = {"floor": (0, 255, 0), "door": (0, 200, 255)}.get(o.place.kind, (128, 128, 128))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
                cv2.putText(img, f"#{o.track_id} {o.place.kind} {o.conf:.2f}", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            on = s.room == w.room
            cv2.putText(img, f"{cid} {w.room}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 0, 255) if on else (255, 255, 255), 3)
            tiles.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        top = np.hstack(tiles[:2])
        bot = np.hstack(tiles[2:4])
        m = np.vstack([top, bot])
        label = f"{s.state} {s.room or '-'} {s.reason or ''} {s.door or ''}"
        cv2.rectangle(m, (0, m.shape[0] // 2 - 22), (m.shape[1], m.shape[0] // 2 + 22), (0, 0, 0), -1)
        cv2.putText(m, label, (10, m.shape[0] // 2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return m
