"""Detector demo video: runs the civet pose model on a stretch of a recording and writes
an MP4 with the box and 6-point skeleton drawn on every frame.

Run from this folder:
    python demo_video.py ../../civetwatch_data/cam01_20260405_0000.mp4 --start 3236 --duration 20
    python demo_video.py VIDEO --start 00:53:56 --duration 20 --out outputs/demo/cam01.mp4

Several clips into one reel (each clip: path@start@duration):
    python demo_video.py --reel cam01.mp4@3236@12 cam04.mp4@2565@12 --out outputs/demo/reel.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from pose_draw import DEFAULT_WEIGHTS, CivetPose, VideoOut, draw


def seconds(s: str) -> float:
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(s)


def clip(model: CivetPose, vw_holder: dict, path: str, start: float, duration: float, args,
         label: str | None = None, progress=None) -> tuple[int, int]:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    src_fps = args.source_fps
    every = max(1, round(src_fps / args.fps))
    n_in, n_out, n_det = 0, 0, 0
    last_dets = []
    name = label or Path(path).stem
    while n_in < duration * src_fps:
        ok, frame = cap.read()
        if not ok:
            break
        if n_in % every == 0:                       # run the model at --fps, draw on every output frame
            last_dets = model(frame, conf=args.conf, imgsz=args.imgsz, grey_clahe=not args.no_clahe)
            n_det += bool(last_dets)
            t = start + n_in / src_fps
            banner = f"{name}  t={int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"
            out = draw(frame, last_dets, kp_thr=args.kp_thr, show_names=args.names, banner=banner)
            if args.width:
                h = round(out.shape[0] * args.width / out.shape[1] / 2) * 2
                out = cv2.resize(out, (args.width, h), interpolation=cv2.INTER_AREA)
            if vw_holder.get("vw") is None:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                vw_holder["vw"] = VideoOut(args.out, args.fps, (out.shape[1], out.shape[0]))
            vw_holder["vw"].write(out)
            n_out += 1
            if progress is not None:
                progress(n_out, max(1, round(duration * args.fps)))
        n_in += 1
    cap.release()
    return n_out, n_det


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="recording to run on")
    ap.add_argument("--start", default="0", help="seconds, or HH:MM:SS from the file start")
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--reel", nargs="+", help="clips as path@start@duration (start may be HH:MM:SS)")
    ap.add_argument("--out", default="outputs/demo/detector_demo.mp4")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kp-thr", type=float, default=0.5, help="keypoints below this are drawn hollow")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fps", type=float, default=7.5, help="output and inference rate (pipeline: 7.5)")
    ap.add_argument("--source-fps", type=float, default=15.0, help="real NVR rate (headers say 25/30)")
    ap.add_argument("--width", type=int, default=1280, help="output width (0 = full size)")
    ap.add_argument("--names", action="store_true", help="write keypoint names next to the dots")
    ap.add_argument("--no-clahe", action="store_true", help="feed colour frames (the model was trained on grey + CLAHE)")
    ap.add_argument("--device", default=None, help="0 for the GPU, cpu; default: GPU if available")
    args = ap.parse_args()
    if not args.video and not args.reel:
        ap.error("give a video or --reel")

    model = CivetPose(args.weights, args.device)
    holder: dict = {}
    t0 = time.perf_counter()
    jobs = []
    if args.reel:
        for item in args.reel:
            path, st, du = item.rsplit("@", 2)
            jobs.append((path, seconds(st), float(du)))
    else:
        jobs.append((args.video, seconds(args.start), args.duration))
    total, hits = 0, 0
    for path, st, du in jobs:
        n, d = clip(model, holder, path, st, du, args)
        total, hits = total + n, hits + d
        print(f"{Path(path).name} @ {st:.0f}s: {n} frames, civet detected in {d}")
    if holder.get("vw"):
        holder["vw"].close()
        note = "" if holder["vw"].browser_playable else " (mp4v: install ffmpeg for a browser-playable H.264 file)"
        print(f"wrote {args.out}: {total} frames, detections in {hits}, "
              f"{time.perf_counter() - t0:.0f} s on {model.device}{note}")


if __name__ == "__main__":
    main()