"""Evaluate the friend system's behaviour events against our 711 human labels.

Joins civetwatch's behavior_event table (SQLite) with data/annotations.csv on
simulated time: friend ts = file_start + timestamp_seconds (FileStepper now
steps by container fps, so sim time == our scan time).

Usage:
  python eval_friend_behaviour.py --datasets 1 2 3
  python eval_friend_behaviour.py --datasets 1        # one session only
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path

FRIEND_TO_OURS = {
    "walking": "Walking",
    "running": "Running",
    "standing": "Standing",
    "resting": "Resting",
    "eating": "Eating",
    "drinking": "Drinking",
    "grooming": "Grooming",
    "marking": "Other",
    "urination": "Other",
    "defecation": "Other",
    "unknown": "No Civet",
}
NO_EVENT = "no_event"


def load_events(db_path: Path):
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "SELECT camera_id, behavior, timestamp_start, timestamp_end, confidence "
        "FROM behavior_event ORDER BY timestamp_start"
    ).fetchall()
    con.close()
    per_cam: dict[str, list[tuple[datetime, datetime, str, float]]] = {}
    for cam, beh, s, e, conf in rows:
        per_cam.setdefault(cam, []).append(
            (datetime.fromisoformat(s), datetime.fromisoformat(e), beh, conf or 0.0)
        )
    return per_cam


def find_event(events, ts: datetime):
    """Return (label, camera, conf, side) of an event covering ts, else None.
    side: 'inside' | 'before_first' | 'after_last' | 'gap' (for diagnostics)."""
    if not events:
        return None
    starts = [e[0] for e in events]
    i = bisect_right(starts, ts) - 1
    if i >= 0:
        s, e, beh, conf = events[i]
        if ts <= e:
            return beh, conf, "inside"
    if i < 0:
        return None, None, "before_first"
    return None, None, "gap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--db-dir", default="civet-monitor/outputs")
    ap.add_argument("--annotations", default="data/annotations.csv")
    ap.add_argument("--file-start", default="2026-04-05T00:00:00")
    ap.add_argument("--out", default="results/friend_eval.json")
    args = ap.parse_args()

    t0 = datetime.fromisoformat(args.file_start)
    sessions = {}
    for d in args.datasets:
        db = Path(args.db_dir) / f"our_dataset{d}.sqlite"
        if not db.exists():
            print(f"skip missing db: {db}")
            continue
        sessions[d] = load_events(db)
    if not sessions:
        raise SystemExit("no session databases found")

    rows = list(csv.DictReader(open(args.annotations, encoding="utf-8-sig")))
    results, conf_mat = [], {}
    for r in rows:
        video = r["source_video"]
        d = int(video.split("_Dataset")[1].split(".")[0])
        if d not in sessions:
            continue
        ts = t0 + timedelta(seconds=float(r["timestamp_seconds"]))
        cam = "CAM%02d" % int(r["camera"].replace("CAM", ""))
        events = sessions[d].get(cam, [])
        beh, conf, side = find_event(events, ts)
        ours = r["label"]
        if beh is None:
            pred, matched = NO_EVENT, False
            conf = 0.0
        else:
            pred, matched = FRIEND_TO_OURS.get(beh, beh), True
        results.append({
            "video": video, "camera": cam, "t": float(r["timestamp_seconds"]),
            "ours": ours, "friend_raw": beh or "", "friend": pred,
            "conf": conf or 0.0, "matched": matched, "side": side,
        })
        conf_mat.setdefault(ours, {}).setdefault(pred, 0)
        conf_mat[ours][pred] += 1

    n = len(results)
    matched = [x for x in results if x["matched"]]
    civet = [x for x in results if x["ours"] != "No Civet"]
    civet_matched = [x for x in civet if x["matched"]]

    def acc(xs):
        return sum(x["ours"] == x["friend"] for x in xs) / len(xs) if xs else 0.0

    presence_rows = [{"t": 1 if x["ours"] != "No Civet" else 0,
                      "p": 1 if x["friend"] not in (NO_EVENT, "No Civet") else 0}
                     for x in results]
    presence = sum(a["t"] == a["p"] for a in presence_rows) / len(presence_rows)

    # camera agreement is reported implicitly via time-only matching

    summary = {
        "sessions": sorted(sessions),
        "labels_evaluated": n,
        "coverage": len(matched) / n if n else 0.0,
        "accuracy_all": acc(results),
        "accuracy_civet_present": acc(civet),
        "presence_accuracy": presence,
        "n_civet_present": len(civet),
        "n_civet_matched": len(civet_matched),
        "per_class": {},
    }
    classes = sorted({x["ours"] for x in results})
    for c in classes:
        tp = sum(1 for x in results if x["ours"] == c and x["friend"] == c)
        fp = sum(1 for x in results if x["ours"] != c and x["friend"] == c)
        fn = sum(1 for x in results if x["ours"] == c and x["friend"] != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        summary["per_class"][c] = {"precision": round(prec, 3),
                                   "recall": round(rec, 3), "f1": round(f1, 3),
                                   "support": sum(1 for x in results if x["ours"] == c)}

    print(json.dumps(summary, indent=2))
    print("\nconfusion (rows=ours, cols=friend):")
    cols = classes + [NO_EVENT]
    print(f"{'':12s}" + "".join(f"{c[:10]:>11s}" for c in cols))
    for c in classes:
        line = f"{c[:12]:12s}"
        for p in cols:
            line += f"{conf_mat.get(c, {}).get(p, 0):>11d}"
        print(line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": results}, open(out, "w"), indent=1)
    with open(out.with_name("friend_confusion.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ours\\friend"] + cols)
        for c in classes:
            w.writerow([c] + [conf_mat.get(c, {}).get(p, 0) for p in cols])
    print(f"\nwrote {out} and {out.with_name('friend_confusion.csv')}")


if __name__ == "__main__":
    main()
