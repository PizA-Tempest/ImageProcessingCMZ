from __future__ import annotations

import cv2
import numpy as np


def _empty():
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)


class MotionDetector:

    def __init__(self, cfg: dict, overlay_mask_full: np.ndarray | None = None):
        self.w = int(cfg.get("width", 192))
        self.h = int(cfg.get("height", 108))
        self.thr = int(cfg.get("diff_threshold", 25))
        self.fd_thr = int(cfg.get("frame_diff_threshold", 12))
        self.min_area = int(cfg.get("min_area_px", 40))
        self.max_area_frac = float(cfg.get("max_area_frac", 0.25))
        self.trigger_frames = int(cfg.get("trigger_frames", 2))
        self.lr = float(cfg.get("bg_learning_rate", 0.002))
        self.hold_s = float(cfg.get("hold_s", 2.0))
        self.max_blobs = int(cfg.get("max_blobs", 3))
        self.mask = None
        if overlay_mask_full is not None:
            self.mask = cv2.resize(overlay_mask_full, (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
        self.bg: np.ndarray | None = None
        self.prev: np.ndarray | None = None
        self.mhi: np.ndarray | None = None
        self.hits = 0
        self.awake = False
        self.fg_area = 0

    def reset(self):
        self.bg = self.prev = self.mhi = None
        self.hits = 0

    def update(self, gray_full: np.ndarray, t: float):
        H, W = gray_full.shape[:2]
        small = cv2.resize(gray_full, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        if self.bg is None:
            self.bg = small.copy()
            self.prev = small
            self.mhi = np.full((self.h, self.w), -1e9, np.float32)
            return _empty()
        fg = np.abs(small - self.bg) > self.thr
        fd = np.abs(small - self.prev) > self.fd_thr
        if self.mask is not None:
            fg &= self.mask
            fd &= self.mask
        self.prev = small
        if fg.mean() > 0.4:
            self.bg = small.copy()
            self.hits = 0
            self.awake = False
            return _empty()
        self.mhi[fd] = t
        recent = (self.mhi >= t - self.hold_s).astype(np.uint8)
        recent = cv2.dilate(recent, np.ones((7, 7), np.uint8))
        live = (fg & (recent > 0)).astype(np.uint8)
        live = cv2.morphologyEx(live, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        live = cv2.morphologyEx(live, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.fg_area = int(live.sum())

        rate = np.where(fg, self.lr * 0.1, self.lr).astype(np.float32)
        self.bg += rate * (small - self.bg)

        self.hits = self.hits + 1 if self.fg_area >= self.min_area else 0
        self.awake = self.hits >= self.trigger_frames
        if self.fg_area < self.min_area:
            return _empty()

        n, _, stats, _ = cv2.connectedComponentsWithStats(live)
        sx, sy = W / self.w, H / self.h
        boxes, confs = [], []
        max_area = self.max_area_frac * self.w * self.h
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < self.min_area or a > max_area:
                continue
            boxes.append((x * sx, y * sy, (x + w) * sx, (y + h) * sy))
            confs.append(min(0.95, 0.3 + a / 300.0))
        if not boxes:
            return _empty()
        order = np.argsort(confs)[::-1][: self.max_blobs]
        return np.asarray(boxes, np.float32)[order], np.asarray(confs, np.float32)[order]


class YoloDetector:

    def __init__(self, cfg: dict, base_dir: str | None = None):
        from ultralytics import YOLO
        from pathlib import Path

        w = Path(cfg.get("weights", "yolo26n.pt"))
        if not w.is_absolute() and base_dir and (Path(base_dir) / w).exists():
            w = Path(base_dir) / w
        self.model = YOLO(str(w))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.25))
        self.classes = cfg.get("classes")
        dev = cfg.get("device", 0)
        try:
            import torch
            if not torch.cuda.is_available():
                dev = "cpu"
        except ImportError:
            dev = "cpu"
        self.device = dev
        self.half = bool(cfg.get("half", True)) and dev != "cpu"

    def detect(self, img3: np.ndarray, keypoints: bool = False):
        r = self.model.predict(img3, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
                               device=self.device, half=self.half, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (*_empty(), np.zeros((0, 6, 3), np.float32)) if keypoints else _empty()
        xyxy, conf = r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32)
        if not keypoints:
            return xyxy, conf
        k = r.keypoints.data.cpu().numpy().astype(np.float32) if r.keypoints is not None else np.zeros((len(conf), 6, 3), np.float32)
        return xyxy, conf, k

    def detect_batch(self, imgs: list[np.ndarray]):
        if not imgs:
            return []
        rs = self.model.predict(imgs, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
                                device=self.device, half=self.half, verbose=False)
        out = []
        for r in rs:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(_empty())
            else:
                kp = r.keypoints.data.cpu().numpy().astype(np.float32) if r.keypoints is not None else None
                out.append((r.boxes.xyxy.cpu().numpy().astype(np.float32),
                            r.boxes.conf.cpu().numpy().astype(np.float32), kp))
        return out
