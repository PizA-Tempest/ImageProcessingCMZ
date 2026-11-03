import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from civetwatch.config import load_config
from remux import ffmpeg_exe


def read_log(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["t"] = datetime.fromisoformat(r["ts"])
    return rows


def summarise(rows, end: datetime):
    dur = defaultdict(float)
    for a, b in zip(rows, rows[1:] + [{"t": end}]):
        dur[(a["room"], a["state"])] += (b["t"] - a["t"]).total_seconds()
    total = sum(dur.values())
    rooms = sorted({k[0] for k in dur})
    print(f"\n{'room':8} {'visible':>9} {'hidden':>9} {'pending':>9} {'total':>9} {'share':>6}")
    per_room = {}
    for r in rooms:
        v, h, p = (dur[(r, s)] for s in ("VISIBLE", "HIDDEN", "PENDING"))
        per_room[r] = v + h + p
        print(f"{r:8} {v:9.0f} {h:9.0f} {p:9.0f} {v + h + p:9.0f} {100 * (v + h + p) / total:5.1f}%")
    return per_room


def room_changes(rows):
    out, prev = [], None
    for r in rows:
        if r["state"] == "VISIBLE" and r["room"] != prev:
            if prev is not None:
                out.append({"ts": r["ts"], "from": prev, "to": r["room"], "camera": r["camera"],
                            "reason": r["reason"], "t": r["t"]})
            prev = r["room"]
    return out


def grab(src, sec, w=640, h=360):
    cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{max(sec, 0):.2f}", "-i", src, "-frames:v", "1",
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    b = subprocess.run(cmd, capture_output=True).stdout
    if len(b) < w * h * 3:
        return np.zeros((h, w, 3), np.uint8)
    return np.frombuffer(b[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--config", default="config/system.yaml")
    ap.add_argument("--end", default=None, help="end of the analysed period (ISO); default last row + 1 s")
    ap.add_argument("--sheets", default=None, help="folder for contact sheets")
    ap.add_argument("--after", type=float, default=1.0, help="seconds after the change to show")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rows = read_log(a.log)
    if not rows:
        sys.exit("empty log")
    end = datetime.fromisoformat(a.end) if a.end else rows[-1]["t"] + timedelta(seconds=1)
    summarise(rows, end)
    ch = room_changes(rows)
    print(f"\n{len(ch)} room changes")
    for i, c in enumerate(ch):
        print(f"  {i:3d} {c['ts'][11:23]}  {c['from']} -> {c['to']}  {c['reason']}")
    pairs = defaultdict(int)
    for c in ch:
        pairs[tuple(sorted((c["from"], c["to"])))] += 1
    print("\nchanges per room pair:", ", ".join(f"{a}-{b}: {n}" for (a, b), n in sorted(pairs.items())))
    if not a.sheets:
        return
    out = Path(a.sheets)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "changes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "ts", "from", "to", "reason", "ok"])
        for i, c in enumerate(ch):
            w.writerow([i, c["ts"], c["from"], c["to"], c["reason"], ""])
    cams = sorted(cfg["cameras"])
    tiles = []
    for i, c in enumerate(ch):
        imgs = []
        for cid in cams:
            cc = cfg["cameras"][cid]
            sec = (c["t"] - datetime.fromisoformat(cc["file_start"])).total_seconds() + a.after
            im = grab(cc["source"], sec)
            col = (0, 0, 255) if cc["room"] == c["to"] else (0, 255, 255) if cc["room"] == c["from"] else (255, 255, 255)
            cv2.putText(im, f"{cid} {cc['room']}", (8, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            imgs.append(im)
        m = np.vstack([np.hstack(imgs[:2]), np.hstack(imgs[2:])])
        cv2.rectangle(m, (0, 0), (m.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(m, f"#{i} {c['ts'][11:22]} {c['from']} -> {c['to']} {c['reason']}", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(m)
        if len(tiles) == 4 or i == len(ch) - 1:
            while len(tiles) < 4:
                tiles.append(np.zeros_like(m))
            sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            cv2.imwrite(str(out / f"changes_{i // 4:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            tiles = []
    print(f"sheets -> {out}")


if __name__ == "__main__":
    main()
