import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def labelled(root: Path, split: str) -> list[Path]:
    return [p for p in sorted((root / "images" / split).glob("*.png"))
            if (root / "labels" / split / f"{p.stem}.txt").exists()]


def write_list(paths: list[Path], out: Path) -> str:
    out.write_text("\n".join(str(p.resolve()) for p in paths) + "\n", encoding="utf-8")
    return str(out.resolve())


def write_data_yaml(cfg: dict, out: Path, bootstrap_val: float, seed: int, test: bool = False) -> Path:
    root = (ROOT / cfg["data_root"]).resolve()
    train = [img for split in cfg.get("train_splits", ["train"]) for img in labelled(root, split)]
    val = labelled(root, "test" if test else "val")
    if bootstrap_val:
        rnd = random.Random(seed)
        val = sorted(rnd.sample(train, max(1, round(bootstrap_val * len(train)))))
        train = [p for p in train if p not in set(val)]
        print(f"bootstrap val: {len(val)} of the labelled train frames held back (scores are NOT a real result)")
    if not train or not val:
        sys.exit(f"labelled frames: train {len(train)}, val {len(val)}. Label the val task, "
                 f"or use --bootstrap-val 0.15 for a first pre-labelling model.")
    print(f"training on {len(train)} labelled frames, validating on {len(val)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(root),
        "train": write_list(train, out.with_name("train.txt")),
        "val": write_list(val, out.with_name("val.txt")),
        "names": cfg["names"],
        "kpt_names": {0: cfg["kpt_names"]},
        "kpt_shape": cfg["kpt_shape"],
        "flip_idx": cfg["flip_idx"],
    }
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def subset_yamls(cfg: dict, data_yaml: Path) -> dict[str, Path]:
    night: dict[str, bool] = {}
    with open(ROOT / cfg["manifest"], newline="") as fh:
        for r in csv.DictReader(fh):
            night[r["file"]] = r["night"] == "1"
    base = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    val_imgs = [Path(p) for p in Path(base["val"]).read_text(encoding="utf-8").split()]
    out = {}
    for label, want in (("night", True), ("day", False)):
        imgs = [p for p in val_imgs if night.get(p.name) == want]
        if not imgs:
            continue
        lst = data_yaml.with_name(f"val_{label}.txt")
        lst.write_text("\n".join(str(p.resolve()) for p in imgs) + "\n", encoding="utf-8")
        y = data_yaml.with_name(f"data_val_{label}.yaml")
        with open(y, "w", encoding="utf-8") as f:
            yaml.safe_dump({**base, "val": str(lst.resolve())}, f, sort_keys=False)
        out[f"{label} ({len(imgs)} images)"] = y
    return out


def read_label(path: Path, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        v = [float(x) for x in line.split()]
        if v:
            out.append((np.array(v[1:5]), np.array(v[5:5 + 3 * k]).reshape(k, 3)))
    return out


def xyxy(cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def body_length(kpts: np.ndarray, names: list[str]) -> float | None:
    idx = [names.index(n) for n in ("nose", "neck", "mid_back", "tail_base")]
    if kpts[idx[0], 2] == 0 or kpts[idx[-1], 2] == 0:
        return None
    pts = kpts[[i for i in idx if kpts[i, 2] > 0], :2]
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def pck(model, imgs: list[Path], cfg: dict, thr: float = 0.1, conf: float = 0.25) -> dict:
    names, k = cfg["kpt_names"], cfg["kpt_shape"][0]
    hits: dict[str, list[float]] = {"all": [], "visible": [], "body": [], **{n: [] for n in names}}
    civets = matched = with_body = 0
    for img in imgs:
        r = model.predict(str(img), imgsz=cfg["train"]["imgsz"], conf=conf, device=cfg["train"]["device"],
                          verbose=False)[0]
        H, W = r.orig_shape
        pb = r.boxes.xyxy.cpu().numpy()
        pk = r.keypoints.xy.cpu().numpy() if len(pb) else np.zeros((0, k, 2))
        free = list(np.argsort(-r.boxes.conf.cpu().numpy()))
        for box, kp in read_label(img.parents[2] / "labels" / img.parent.name / f"{img.stem}.txt", k):
            civets += 1
            gb = xyxy(box) * np.array([W, H, W, H])
            best = max(free, key=lambda j: iou(gb, pb[j]), default=None)
            if best is None or iou(gb, pb[best]) < 0.5:
                continue
            free.remove(best)
            matched += 1
            g = kp * np.array([W, H, 1.0])
            side, body = float(max(gb[2] - gb[0], gb[3] - gb[1])), body_length(g, names)
            with_body += body is not None
            for i, n in enumerate(names):
                if g[i, 2] > 0:
                    err = float(np.linalg.norm(pk[best, i] - g[i, :2]))
                    ok = float(err <= thr * side)
                    hits[n].append(ok)
                    hits["all"].append(ok)
                    if g[i, 2] == 2:
                        hits["visible"].append(ok)
                    if body is not None:
                        hits["body"].append(float(err <= thr * body))
    hits["head"] = hits["nose"] + hits["eyes"]
    hits["tail"] = hits["tail_base"] + hits["tail_tip"]
    score = {key: (float(np.mean(v)) if v else float("nan")) for key, v in hits.items()}
    return {**score, "civets": civets, "matched": matched, "with_body": with_body}


def evaluate(weights: str, cfg: dict, data_yaml: Path, split: str = "val") -> None:
    from ultralytics import YOLO
    model = YOLO(weights)
    t = cfg["train"]
    subsets = [(f"all {split}", data_yaml), *subset_yamls(cfg, data_yaml).items()]
    print(f"\n{'subset':<22} {'box mAP50':>10} {'box mAP50-95':>13} {'pose mAP50':>11} {'pose mAP50-95':>14}")
    for label, y in subsets:
        m = model.val(data=str(y), imgsz=t["imgsz"], batch=t["batch"], device=t["device"],
                      split="val", plots=False, verbose=False)
        print(f"{label:<22} {m.box.map50:>10.3f} {m.box.map:>13.3f} {m.pose.map50:>11.3f} {m.pose.map:>14.3f}")
    names = cfg["kpt_names"]
    print("\nPCK@0.1 of the longer box side (matched civets, labelled points; visible = v=2 only;"
          " body = 0.1 x body length, civets with nose and tail_base labelled)")
    print(f"{'subset':<22} {'civets':>9} {'all':>6} {'visible':>8} {'head':>6} {'tail':>6}  "
          + " ".join(f"{n:>9}" for n in names) + f"  {'body (n)':>12}")
    for label, y in subsets:
        lst = yaml.safe_load(Path(y).read_text(encoding="utf-8"))["val"]
        s = pck(model, [Path(p) for p in Path(lst).read_text(encoding="utf-8").split()], cfg)
        print(f"{label:<22} {s['matched']:>4}/{s['civets']:<4} {s['all']:>6.3f} {s['visible']:>8.3f} "
              f"{s['head']:>6.3f} {s['tail']:>6.3f}  " + " ".join(f"{s[n]:>9.3f}" for n in names)
              + f"  {s['body']:>6.3f} ({s['with_body']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "train_pose.yaml"))
    ap.add_argument("--eval-only", metavar="WEIGHTS", help="skip training, evaluate these weights")
    ap.add_argument("--epochs", type=int, help="override train.epochs (e.g. 1 for a smoke test)")
    ap.add_argument("--bootstrap-val", type=float, default=0.0, metavar="FRACTION",
                    help="no labelled val yet: hold back this share of labelled train frames (pre-label models only)")
    ap.add_argument("--name", help="override train.name (run folder under train.project)")
    ap.add_argument("--test", action="store_true",
                    help="with --eval-only: score the held-out test block (once, at the end; never to pick settings)")
    a = ap.parse_args()
    if a.test and not a.eval_only:
        sys.exit("--test only works with --eval-only: the test block is never trained or tuned on")
    cfg = load_cfg(Path(a.config))
    t = dict(cfg["train"])
    if a.epochs:
        t["epochs"] = a.epochs
    if a.name:
        t["name"] = a.name
    t["project"] = str(ROOT / t["project"])
    name = "civet_pose_test.yaml" if a.test else "civet_pose.yaml"
    data_yaml = write_data_yaml(cfg, Path(t["project"]) / "data" / name, a.bootstrap_val, t["seed"], a.test)

    if a.eval_only:
        evaluate(a.eval_only, cfg, data_yaml, "test" if a.test else "val")
        return
    from ultralytics import YOLO
    weights = ROOT / cfg["model"]
    model = YOLO(str(weights) if weights.parent != ROOT else cfg["model"])
    result = model.train(data=str(data_yaml), **t)
    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")
    evaluate(str(best), cfg, data_yaml)


if __name__ == "__main__":
    sys.exit(main())
