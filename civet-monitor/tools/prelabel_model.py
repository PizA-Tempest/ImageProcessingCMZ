import argparse
import copy
import csv
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]


def in_ranges(i: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= i <= hi for lo, hi in ranges)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help=".zip holding annotations.xml (CVAT 1.1)")
    ap.add_argument("--merge-export", help="CVAT for images 1.1 export of the same task (annotations.xml or .zip)")
    ap.add_argument("--keep-frames", help="task frame ids labelled by hand, kept from the export, e.g. 0-299")
    ap.add_argument("--clear-frames", help="kept frame ids whose shapes are wrong and are emptied, e.g. 399,491")
    ap.add_argument("--conf", type=float, default=0.5, help="box confidence (corner false positives on CAM03 reach 0.51 with v2)")
    ap.add_argument("--max-det", type=int, default=1, help="one civet: a second box is a duplicate or a false positive")
    ap.add_argument("--kpt-conf", type=float, default=0.5, help="keypoints below this are marked outside")
    ap.add_argument("--config", default=str(ROOT / "config" / "train_pose.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    names = cfg["kpt_names"]
    img_dir = ROOT / cfg["data_root"] / "images" / a.split

    kept: dict[str, ET.Element] = {}
    ids: dict[str, int] = {}
    task_names: dict[str, str] = {}
    if a.merge_export:
        src = Path(a.merge_export)
        xml = zipfile.ZipFile(src).read("annotations.xml") if src.suffix == ".zip" else src.read_bytes()
        ranges = [tuple(int(v) for v in r.split("-")) for r in a.keep_frames.split(",")] if a.keep_frames else []
        clear = {int(v) for v in a.clear_frames.split(",")} if a.clear_frames else set()
        for img in ET.fromstring(xml).iter("image"):
            name = Path(img.get("name")).name
            ids[name] = int(img.get("id"))
            task_names[name] = img.get("name")
            if in_ranges(int(img.get("id")), ranges):
                if int(img.get("id")) in clear:
                    for shape in list(img):
                        img.remove(shape)
                    img.text = None
                kept[name] = img

    from ultralytics import YOLO
    model = YOLO(a.weights)
    with open(ROOT / cfg["manifest"], newline="") as fh:
        files = sorted(r["file"] for r in csv.DictReader(fh) if r["split"] == a.split)
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    n_kept = n_pred = n_civets = 0
    for idx, name in enumerate(files):
        if name in kept:
            root.append(copy.deepcopy(kept[name]))
            n_kept += 1
            continue
        frame = cv2.imread(str(img_dir / name))
        h, w = frame.shape[:2]
        el = ET.SubElement(root, "image", id=str(ids.get(name, idx)), name=task_names.get(name, name), width=str(w), height=str(h))
        r = model.predict(frame, conf=a.conf, max_det=a.max_det, imgsz=cfg["train"]["imgsz"], verbose=False)[0]
        kxy = r.keypoints.xy.tolist() if r.keypoints is not None else []
        kcf = r.keypoints.conf.tolist() if r.keypoints is not None and r.keypoints.conf is not None else []
        for k, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.tolist()):
            ET.SubElement(el, "box", label="civet_box", source="auto", occluded="0", xtl=f"{x1:.1f}",
                          ytl=f"{y1:.1f}", xbr=f"{x2:.1f}", ybr=f"{y2:.1f}", z_order="0")
            sk = ET.SubElement(el, "skeleton", label="civet", source="auto", z_order="0")
            for j, kname in enumerate(names):
                x, y = kxy[k][j] if k < len(kxy) else (0.0, 0.0)
                c = kcf[k][j] if k < len(kcf) else 0.0
                ET.SubElement(sk, "points", label=kname, source="auto", occluded="0",
                              outside="0" if c >= a.kpt_conf else "1", points=f"{x:.1f},{y:.1f}")
            n_civets += 1
        n_pred += 1
        if idx % 100 == 0:
            print(f"{idx}/{len(files)}", flush=True)
    missing = set(kept) - set(files)
    if missing:
        sys.exit(f"{len(missing)} kept frames are not in the manifest split, e.g. {sorted(missing)[:3]}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations.xml", '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n")
    print(f"{a.split}: {n_kept} frames kept from the export, {n_pred} predicted ({n_civets} civets) -> {out}")


if __name__ == "__main__":
    main()
