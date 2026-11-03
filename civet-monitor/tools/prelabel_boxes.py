import argparse
import csv
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from civetwatch.config import load_config
from civetwatch.detect import MotionDetector
from civetwatch.preprocess import Preprocessor

BG_OFFSETS_S = (-60, -40, -20, 20, 40, 60)
WARM_S = 3.0
STEP = 3
MERGE_PX = 40
ANIMAL_CLASSES = list(range(14, 24))


def small_of(md: MotionDetector, gray: np.ndarray) -> np.ndarray:
    s = cv2.resize(gray, (md.w, md.h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return cv2.GaussianBlur(s, (3, 3), 0)


def read_at(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    return img if ok else None


def merge(boxes: list[list[float]]) -> list[list[float]]:
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (a[0] - MERGE_PX < b[2] and b[0] - MERGE_PX < a[2]
                        and a[1] - MERGE_PX < b[3] and b[1] - MERGE_PX < a[3]):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def overlaps(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def with_yolo(motion: list[list[float]], yolo: list[list[float]]) -> list[list[float]]:
    return yolo + [m for m in motion if not any(overlaps(m, y) for y in yolo)]


def boxes_for(cap: cv2.VideoCapture, pos: float, pre: Preprocessor,
              md: MotionDetector) -> tuple[list[list[float]], np.ndarray | None]:
    bgs = [read_at(cap, pos + o) for o in BG_OFFSETS_S if pos + o >= 0]
    bgs = [small_of(md, pre.gray(b)) for b in bgs if b is not None]
    md.reset()
    if bgs:
        md.bg = np.median(np.stack(bgs), axis=0).astype(np.float32)
        md.mhi = np.full((md.h, md.w), -1e9, np.float32)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, pos - WARM_S) * 1000.0)
    xyxy = np.zeros((0, 4))
    img = None
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        last = t >= pos - 0.034
        if i % STEP == 0 or last:
            _, img = cap.retrieve()
            g = pre.gray(img)
            if md.prev is None:
                md.prev = small_of(md, g)
            else:
                xyxy, _ = md.update(g, t)
        i += 1
        if last:
            break
    return merge(xyxy.tolist()), img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="train or val (rows of the manifest)")
    ap.add_argument("--manifest", default=str(ROOT / "labelling" / "manifest.csv"))
    ap.add_argument("--config", default=str(ROOT / "config" / "april5_24h.yaml"))
    ap.add_argument("--out", required=True, help=".zip holding annotations.xml (CVAT 1.1)")
    ap.add_argument("--yolo", default=str(ROOT / "models" / "yolo26s.pt"), help="stock COCO model; '' to skip")
    ap.add_argument("--yolo-conf", type=float, default=0.1)
    a = ap.parse_args()

    yolo = None
    if a.yolo:
        from ultralytics import YOLO
        yolo = YOLO(a.yolo)
    cfg = load_config(a.config)
    pre = Preprocessor(cfg["preprocess"])
    md = MotionDetector(cfg["motion"], pre.overlay_mask((1080, 1920)))
    caps = {cid: cv2.VideoCapture(c["source"], cv2.CAP_FFMPEG) for cid, c in cfg["cameras"].items()}
    starts = {cid: datetime.fromisoformat(c["file_start"]) for cid, c in cfg["cameras"].items()}

    with open(a.manifest, newline="") as fh:
        rows = sorted((r for r in csv.DictReader(fh) if r["split"] == a.split), key=lambda r: r["file"])
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    n_boxes = n_with = 0
    for idx, r in enumerate(rows):
        pos = (datetime.fromisoformat(r["time"]) - starts[r["cam"]]).total_seconds()
        boxes, colour = boxes_for(caps[r["cam"]], pos, pre, md)
        if yolo is not None and colour is not None and r["night"] == "0":
            res = yolo.predict(colour, imgsz=1280, conf=a.yolo_conf, classes=ANIMAL_CLASSES, verbose=False)[0]
            boxes = with_yolo(boxes, res.boxes.xyxy.tolist())
        lines.append(f'  <image id="{idx}" name={quoteattr(r["file"])} width="1920" height="1080">')
        for x1, y1, x2, y2 in boxes:
            lines.append(f'    <box label="civet_box" source="auto" occluded="0" '
                         f'xtl="{x1:.1f}" ytl="{y1:.1f}" xbr="{x2:.1f}" ybr="{y2:.1f}" z_order="0"></box>')
        lines.append("  </image>")
        n_boxes += len(boxes)
        n_with += bool(boxes)
        if idx % 100 == 0:
            print(f"{idx}/{len(rows)}", flush=True)
    lines.append("</annotations>")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations.xml", "\n".join(lines) + "\n")
    print(f"{a.split}: {len(rows)} frames, {n_with} with boxes, {n_boxes} boxes -> {out}")


if __name__ == "__main__":
    main()
