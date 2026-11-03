import argparse
import csv
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from civetwatch.behaviour import LABELS
from review_room_log import read_log, room_changes

DAY = "2026-04-05"


def per_second(db: str, log_path: str, t0: datetime, t1: datetime) -> list[tuple[datetime, str, str]]:
    con = sqlite3.connect(db)
    ev = [(datetime.fromisoformat(s), datetime.fromisoformat(e), b)
          for s, e, b in con.execute("SELECT timestamp_start, timestamp_end, behavior FROM behavior_event ORDER BY timestamp_start")]
    con.close()
    log = [(r["t"], r["room"]) for r in read_log(log_path)]
    out, i, j, room = [], 0, 0, ""
    t = t0
    while t < t1:
        while i + 1 < len(ev) and ev[i][1] <= t:
            i += 1
        while j < len(log) and log[j][0] <= t:
            room = log[j][1] or room
            j += 1
        b = ev[i][2] if ev and ev[i][0] <= t < ev[i][1] else "unknown"
        out.append((t, b, room))
        t += timedelta(seconds=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--room-log", required=True)
    ap.add_argument("--start", required=True, help="HH:MM[:SS] on April 5")
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fix = lambda s: s if s.count(":") == 2 else s + ":00"
    t0, t1 = datetime.fromisoformat(f"{DAY}T{fix(a.start)}"), datetime.fromisoformat(f"{DAY}T{fix(a.end)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    secs = per_second(a.db, a.room_log, t0, t1)

    hours = defaultdict(list)
    for t, b, r in secs:
        hours[t.replace(minute=0, second=0)].append((b, r))
    with open(out / "hourly_activity.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hour", "active_min", "inactive_min", "not_seen_min", "main_room", "main_behaviour"])
        for h, v in sorted(hours.items()):
            act = sum(b not in ("resting", "unknown") for b, _ in v) / 60
            ina = sum(b == "resting" for b, _ in v) / 60
            seen = [b for b, _ in v if b != "unknown"]
            w.writerow([f"{h:%H:00}", round(act, 1), round(ina, 1), round(len(v) / 60 - act - ina, 1),
                        Counter(r for _, r in v if r).most_common(1)[0][0] if any(r for _, r in v) else "",
                        Counter(seen).most_common(1)[0][0] if seen else "unknown"])

    budget = defaultdict(float)
    for t, b, r in secs:
        part = "day" if 6 <= t.hour < 18 else "night"
        budget[(part, b, r or "?")] += 1 / 60
    rooms = sorted({r for _, _, r in secs if r})
    with open(out / "time_budget.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "behaviour", *rooms, "total_min"])
        for part in ("night", "day"):
            for b in LABELS:
                row = [round(budget[(part, b, r)], 1) for r in rooms]
                if sum(row):
                    w.writerow([part, b, *row, round(sum(row), 1)])
    log = read_log(a.room_log)
    ch = [c for c in room_changes(log) if t0 <= c["t"] < t1]
    pairs = Counter(tuple(sorted((c["from"], c["to"]))) for c in ch)
    with open(out / "room_use.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["room", "minutes", "share_%"])
        for r in rooms:
            m = sum(1 for _, _, x in secs if x == r) / 60
            w.writerow([r, round(m, 1), round(100 * m * 60 / len(secs), 1)])
        w.writerow([])
        w.writerow(["crossing", "count"])
        for (x, y), n in sorted(pairs.items()):
            w.writerow([f"{x}-{y}", n])
    charts(secs, rooms, out, f"{t0:%H:%M}-{t1:%H:%M}")
    print(f"report -> {out} (hourly_activity.csv, time_budget.csv, room_use.csv, timeline.png, time_budget.png)")


def charts(secs, rooms, out: Path, span: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cmap = plt.get_cmap("tab20")
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    runs = defaultdict(list)
    i0 = 0
    for i in range(1, len(secs) + 1):
        if i == len(secs) or secs[i][1:] != secs[i0][1:]:
            if secs[i0][2]:
                runs[secs[i0][1:]].append((mdates.date2num(secs[i0][0]), (i - i0) / 86400))
            i0 = i
    fig, ax = plt.subplots(figsize=(16, 1 + 0.6 * len(rooms)))
    for (lab, r), spans in runs.items():
        ax.broken_barh(spans, (rooms.index(r) + 0.1, 0.8), color=cmap(LABELS.index(lab)), linewidth=0)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_yticks([i + 0.5 for i in range(len(rooms))], rooms)
    ax.set_title(f"A01 timeline {span} (one lane per room, colour = behaviour)")
    used = [l for l in LABELS if any(k[0] == l for k in runs)]
    ax.legend([Patch(color=cmap(LABELS.index(l))) for l in used], used, ncol=11, fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(out / "timeline.png", dpi=120)
    mins = Counter(b for _, b, _ in secs)
    fig, ax = plt.subplots(figsize=(8, 4))
    labs = [l for l in LABELS if mins[l]]
    ax.barh(labs, [mins[l] / 60 for l in labs], color=[cmap(LABELS.index(l)) for l in labs])
    ax.set_xlabel("minutes")
    ax.set_title(f"Time budget {span}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "time_budget.png", dpi=120)


if __name__ == "__main__":
    main()
