from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

SOURCE_FPS = 15.0


@dataclass
class Frame:
    cam_id: str
    seq: int
    ts: datetime
    image: np.ndarray | None
    dets: tuple | None = None


def is_live(source: str) -> bool:
    return "://" in str(source)


class CameraReader(threading.Thread):

    def __init__(self, cam_id: str, source: str, file_start: datetime | None = None,
                 replay: bool = True, speed: float = 1.0, start_offset_s: float = 0.0,
                 reconnect_initial_s: float = 1.0, reconnect_max_s: float = 30.0):
        super().__init__(name=f"reader-{cam_id}", daemon=True)
        self.cam_id = cam_id
        self.source = source
        self.live = is_live(source)
        self.file_start = file_start or datetime.now()
        self.replay = replay
        self.speed = speed
        self.start_offset_s = start_offset_s
        self.reconnect_initial_s = reconnect_initial_s
        self.reconnect_max_s = reconnect_max_s
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self.finished = False
        self.connected = False
        self.last_frame_wall = 0.0

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()

    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if cap.isOpened() and not self.live and self.start_offset_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_offset_s * 1000.0)
        return cap

    def run(self):
        backoff = self.reconnect_initial_s
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                if not self.live:
                    self.finished = True
                    return
                time.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_max_s)
                continue
            self.connected = True
            backoff = self.reconnect_initial_s
            wall0 = time.monotonic()
            pos0 = None
            n = 0
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break
                n += 1
                if self.live:
                    ts = datetime.now()
                else:
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if pos <= 0:
                        pos = self.start_offset_s + n / SOURCE_FPS
                    if pos0 is None:
                        pos0 = pos
                    ts = self.file_start + timedelta(seconds=pos)
                    if self.replay:
                        due = wall0 + (pos - pos0) / self.speed
                        delay = due - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                self._seq += 1
                with self._lock:
                    self._latest = Frame(self.cam_id, self._seq, ts, img)
                self.last_frame_wall = time.monotonic()
            cap.release()
            self.connected = False
            if not self.live:
                self.finished = True
                return
            time.sleep(backoff)
            backoff = min(backoff * 2, self.reconnect_max_s)


class FileStepper:

    def __init__(self, cam_id: str, source: str, file_start: datetime, start_offset_s: float = 0.0):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {source}")
        # Use the container's own fps: our archive videos are 30 fps with
        # duplicated frames (15 fps real motion), so a fixed 15.0 would run
        # the sim clock at 2x and distort all temporal features.
        _fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = _fps if 1.0 <= _fps <= 120.0 else SOURCE_FPS
        if start_offset_s > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_offset_s * 1000.0)
        self.file_start = file_start
        self.pos = start_offset_s - 1.0 / self.fps
        self.seq = 0
        self.finished = False
        self._img = None

    def frame_at(self, t_s: float) -> Frame | None:
        while not self.finished and self.pos + 1.0 / self.fps <= t_s + 1e-6:
            if not self.cap.grab():
                self.finished = True
                break
            self.pos += 1.0 / self.fps
            self.seq += 1
            self._img = None
        if self.seq == 0:
            return None
        if self._img is None:
            ok, img = self.cap.retrieve()
            if not ok:
                return None
            self._img = img
        return Frame(self.cam_id, self.seq, self.file_start + timedelta(seconds=self.pos), self._img)

    def release(self):
        self.cap.release()
