"""Shared helpers for the detector demos: load the civet pose model, run it the way the
pipeline does (3-channel grey + CLAHE), and draw the box + 6-point skeleton.

Used by demo_app.py (Streamlit) and demo_video.py (command line).
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models" / "civet_yolo26n.pt"

KP_NAMES = ["nose", "eyes", "neck", "mid_back", "tail_base", "tail_tip"]
# Same colours as detect_app.py (hex → BGR for OpenCV).
KP_HEX = ["#f87171", "#fb923c", "#fbbf24", "#34d399", "#22d3ee", "#60a5fa"]
BOX_HEX = "#22c55e"
SKELETON_HEX = "#e2e8f6"
# All six points lie on the body midline, so the skeleton is one chain.
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


KP_BGR = [hex_bgr(h) for h in KP_HEX]
BOX_BGR = hex_bgr(BOX_HEX)
SKELETON_BGR = hex_bgr(SKELETON_HEX)


@dataclass
class Detection:
    box: np.ndarray        # x1, y1, x2, y2 (pixels)
    conf: float
    kpts: np.ndarray       # (6, 3): x, y, confidence


class CivetPose:
    """YOLO26n-pose civet model, run on grey + CLAHE input like the live pipeline."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if device is None:
            try:
                import torch
                device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, bgr: np.ndarray, grey_clahe: bool = True) -> np.ndarray:
        if not grey_clahe:
            return bgr
        g = self.clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def __call__(self, bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640,
                 grey_clahe: bool = True) -> list[Detection]:
        r = self.model.predict(self.preprocess(bgr, grey_clahe), imgsz=imgsz, conf=conf,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kps = (r.keypoints.data.cpu().numpy() if r.keypoints is not None
               else np.zeros((len(boxes), 6, 3), np.float32))
        dets = [Detection(b, float(c), k) for b, c, k in zip(boxes, confs, kps)]
        return sorted(dets, key=lambda d: -d.conf)


def draw(bgr: np.ndarray, dets: list[Detection], kp_thr: float = 0.5,
         show_names: bool = False, show_low: bool = True, banner: str | None = None) -> np.ndarray:
    """Draw boxes, skeleton and keypoints. Keypoints under kp_thr are drawn hollow
    (or hidden when show_low is False); skeleton lines join only confident points."""
    out = bgr.copy()
    h, w = out.shape[:2]
    s = max(w, h) / 1920.0                       # scale lines and text with the frame
    lw, r_dot, fs = max(1, round(3 * s)), max(3, round(9 * s)), 0.9 * s
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_BGR, lw + 1, cv2.LINE_AA)
        label = f"civet {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw - 1))
        ty = y1 - 8 if y1 - th - 12 > 0 else y1 + th + 12
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 10, ty + 6), BOX_BGR, -1)
        cv2.putText(out, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    max(1, lw - 1), cv2.LINE_AA)
        pts = d.kpts
        ok = pts[:, 2] >= kp_thr
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                cv2.line(out, tuple(int(v) for v in pts[a, :2]), tuple(int(v) for v in pts[b, :2]),
                         SKELETON_BGR, lw, cv2.LINE_AA)
        for i, (x, y, c) in enumerate(pts):
            if x <= 0 and y <= 0:
                continue
            p = (int(x), int(y))
            if ok[i]:
                cv2.circle(out, p, r_dot, KP_BGR[i], -1, cv2.LINE_AA)
                cv2.circle(out, p, r_dot, (0, 0, 0), 1, cv2.LINE_AA)
            elif show_low:
                cv2.circle(out, p, r_dot, KP_BGR[i], max(1, lw - 1), cv2.LINE_AA)
            else:
                continue
            if show_names:
                cv2.putText(out, KP_NAMES[i], (p[0] + r_dot + 3, p[1] - r_dot), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6 * s, KP_BGR[i], max(1, lw - 2), cv2.LINE_AA)
    if banner:
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw - 1))
        cv2.rectangle(out, (0, h - th - 20), (tw + 20, h), (0, 0, 0), -1)
        cv2.putText(out, banner, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    max(1, lw - 1), cv2.LINE_AA)
    return out


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


class VideoOut:
    """Writes an MP4 that plays in browsers (H.264 through ffmpeg); falls back to OpenCV mp4v."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path, self.size = str(path), size
        exe = ffmpeg_exe()
        self.proc = None
        self.vw = None
        if exe:
            w, h = size
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", self.path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    @property
    def browser_playable(self) -> bool:
        return self.proc is not None

    def write(self, frame: np.ndarray):
        if frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.vw.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.vw.release()