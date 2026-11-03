import argparse
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BOX_LABEL, SKEL_LABEL = "civet_box", "civet"


def read_xml(path: Path) -> ET.Element:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return ET.fromstring(z.read("annotations.xml"))
    return ET.parse(path).getroot()


def skeleton_points(sk: ET.Element, names: list[str]) -> list[tuple[float, float, int]]:
    pts = {n: (0.0, 0.0, 0) for n in names}
    for p in sk.findall("points"):
        if p.get("label") not in pts:
            continue
        x, y = (float(v) for v in p.get("points").split(";")[0].split(","))
        if p.get("outside") == "1":
            pts[p.get("label")] = (0.0, 0.0, 0)
        else:
            pts[p.get("label")] = (x, y, 1 if p.get("occluded") == "1" else 2)
    return [pts[n] for n in names]


def inside(box: tuple[float, float, float, float], kps) -> int:
    x1, y1, x2, y2 = box
    m = max(20.0, 0.2 * max(x2 - x1, y2 - y1))
    return sum(1 for x, y, v in kps if v and x1 - m <= x <= x2 + m and y1 - m <= y <= y2 + m)


def pair(boxes, skeletons):
    free = list(range(len(boxes)))
    pairs = []
    for kps in skeletons:
        best = max(free, key=lambda i: inside(boxes[i], kps), default=None)
        if best is not None and (inside(boxes[best], kps) > 0 or not any(v for _, _, v in kps)):
            free.remove(best)
            pairs.append((boxes[best], kps))
        else:
            pairs.append((None, kps))
    pairs += [(boxes[i], None) for i in free]
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="CVAT for images 1.1 export (.zip or annotations.xml)")
    ap.add_argument("--split", required=True, help="train or val: writes labelling/labels/<split>")
    ap.add_argument("--config", default=str(ROOT / "config" / "train_pose.yaml"))
    ap.add_argument("--out-root", default=str(ROOT / "labelling" / "labels"))
    ap.add_argument("--keep-auto-boxes", action="store_true",
                    help="keep pre-drawn boxes that have no skeleton and were never edited")
    ap.add_argument("--frames", help="task frame ranges that are fully labelled, e.g. 0-299 or 0-299,400-499 "
                                     "(a task export holds every frame; unlabelled ones must not become negatives)")
    a = ap.parse_args()
    names = yaml.safe_load(open(a.config, encoding="utf-8"))["kpt_names"]
    ranges = [tuple(int(v) for v in r.split("-")) for r in a.frames.split(",")] if a.frames else None

    out = Path(a.out_root) / a.split
    out.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "civets": 0, "no_box": 0, "no_skeleton": 0, "skipped_auto": 0}
    for img in read_xml(Path(a.export)).iter("image"):
        if ranges and not any(lo <= int(img.get("id")) <= hi for lo, hi in ranges):
            continue
        W, H = float(img.get("width")), float(img.get("height"))
        box_els = [b for b in img.findall("box") if b.get("label") == BOX_LABEL]
        boxes = [tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")) for b in box_els]
        auto = {boxes[i] for i, b in enumerate(box_els) if b.get("source") in ("auto", "file")}
        skels = [skeleton_points(s, names) for s in img.findall("skeleton") if s.get("label") == SKEL_LABEL]
        lines = []
        for box, kps in pair(boxes, skels):
            if box is None:
                vis = [(x, y) for x, y, v in kps if v]
                if not vis:
                    print(f"warning: {img.get('name')}: skeleton with every point outside and no box, skipped")
                    continue
                xs, ys = [p[0] for p in vis], [p[1] for p in vis]
                pad = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys), 20)
                box = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
                stats["no_box"] += 1
                print(f"warning: {img.get('name')}: skeleton without a civet_box, padded keypoint box used")
            if kps is None:
                if box in auto and not a.keep_auto_boxes:
                    stats["skipped_auto"] += 1
                    continue
                kps = [(0.0, 0.0, 0)] * len(names)
                stats["no_skeleton"] += 1
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
            vals = [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]
            for x, y, v in kps:
                vals += [x / W, y / H, v] if v else [0.0, 0.0, 0]
            lines.append("0 " + " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals))
            stats["civets"] += 1
        stem = Path(img.get("name")).stem
        (out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
    print(f"{a.split}: {stats['images']} images, {stats['civets']} civets "
          f"({stats['no_box']} without a box, {stats['no_skeleton']} box only); "
          f"{stats['skipped_auto']} untouched pre-drawn boxes skipped -> {out}")
    if stats["images"] == 0:
        sys.exit("no <image> elements found: export as 'CVAT for images 1.1'")


if __name__ == "__main__":
    main()
