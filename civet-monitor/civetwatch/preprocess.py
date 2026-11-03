from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, cfg: dict):
        self.gray3 = cfg.get("gray3", True)
        c = cfg.get("clahe", {})
        self.clahe = (cv2.createCLAHE(clipLimit=float(c.get("clip_limit", 2.0)),
                                      tileGridSize=(int(c.get("tiles", 8)),) * 2)
                      if c.get("enabled", True) else None)
        self.overlay_masks = cfg.get("overlay_masks", [])

    @staticmethod
    def is_night(bgr: np.ndarray) -> bool:
        small = cv2.resize(bgr, (64, 36), interpolation=cv2.INTER_AREA)
        sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
        return float(sat.mean()) < 12.0

    def gray(self, bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.clahe is not None:
            g = self.clahe.apply(g)
        return g

    def model_input(self, bgr: np.ndarray) -> np.ndarray:
        if not self.gray3:
            return bgr
        return cv2.cvtColor(self.gray(bgr), cv2.COLOR_GRAY2BGR)

    def overlay_mask(self, shape) -> np.ndarray:
        h, w = shape[:2]
        m = np.full((h, w), 255, np.uint8)
        sx, sy = w / 1920.0, h / 1080.0
        for x, y, bw, bh in self.overlay_masks:
            cv2.rectangle(m, (int(x * sx), int(y * sy)), (int((x + bw) * sx), int((y + bh) * sy)), 0, -1)
        return m
