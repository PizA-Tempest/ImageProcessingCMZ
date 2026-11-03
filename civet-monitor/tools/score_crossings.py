import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_room_log import read_log, room_changes


def room_name(s: str) -> str:
    s = s.strip().upper()
    digits = s.removeprefix("ROOM").removeprefix("R")
    return f"ROOM{int(digits):02d}" if digits.isdigit() else s


def read_crossings(path: Path, day: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("time", "").strip()]
    for r in rows:
        r["t"] = datetime.fromisoformat(f"{day}T{r['time'].strip()}")
        r["note"] = (r.get("note") or "").strip().lower()
        until = (r.get("until") or "").strip()
        try:
            r["until_t"] = datetime.fromisoformat(f"{day}T{until}") if until else r["t"]
        except ValueError:
            print(f"row {r['time'].strip()}: until '{until}' is not a time, ignored")
            r["until_t"] = r["t"]
        r["from"], r["to"] = room_name(r["from"]), room_name(r["to"] or "")
    return sorted(rows, key=lambda r: r["t"])


def take(changes: list[dict], used: set[int], frm: str, to: str, t: datetime, tol: float) -> int | None:
    best = [i for i, c in enumerate(changes) if i not in used and c["from"] == frm and c["to"] == to
            and abs((c["t"] - t).total_seconds()) <= tol]
    if not best:
        return None
    i = min(best, key=lambda j: abs((changes[j]["t"] - t).total_seconds()))
    used.add(i)
    return i


def room_at(timeline: list[tuple[datetime, str]], t: datetime) -> str | None:
    room = None
    for ts, r in timeline:
        if ts > t:
            break
        room = r
    return room


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("crossings")
    ap.add_argument("log")
    ap.add_argument("--day", default="2026-04-05", help="date of the clock times in the crossing list")
    ap.add_argument("--tol", type=float, default=10.0, help="seconds between a crossing and its log change")
    ap.add_argument("--out", help="CSV with every crossing and log change and how it was scored")
    ap.add_argument("--windows", help="CSV with start,end clock times (e.g. find_gate_visits.py sheet): score only "
                                      "inside them, widened by --pad; no START/END rows needed, no time-in-room score; "
                                      "unmatched log changes within --pad of an edge are not counted")
    ap.add_argument("--pad", type=float, default=5.0, help="seconds added on both sides of each --windows row")
    a = ap.parse_args()
    hand = read_crossings(Path(a.crossings), a.day)
    start = next((r for r in hand if r["from"] == "START"), None)
    end = next((r for r in hand if r["from"] == "END"), None)
    wins = None
    if a.windows:
        with open(a.windows, newline="", encoding="utf-8") as fh:
            wins = [(datetime.fromisoformat(f"{a.day}T{w['start']}") - timedelta(seconds=a.pad),
                     datetime.fromisoformat(f"{a.day}T{w['end']}") + timedelta(seconds=a.pad)) for w in csv.DictReader(fh)]
        start = start or {"t": min(w[0] for w in wins), "to": "ROOM01"}
        end = end or {"t": max(w[1] for w in wins)}
    elif not start or not end:
        sys.exit("the crossing list needs a START row (room at the start) and an END row")
    rooms = {"ROOM01", "ROOM02", "ROOM03", "ROOM04"}
    bad = [r["time"] for r in hand if r["from"] != "END" and (r["to"] not in rooms or r["from"] not in rooms | {"START"})]
    if bad:
        sys.exit(f"rooms must be ROOM01-ROOM04; check the rows at {', '.join(bad)}")
    t0, t1 = start["t"], end["t"]

    def inside(t: datetime, widen: timedelta = timedelta(0)) -> bool:
        if wins is None:
            return t0 - widen <= t <= t1 + widen
        return any(s - widen <= t <= e + widen for s, e in wins)

    log = read_log(a.log)
    pad = timedelta(seconds=a.tol)
    changes = [c for c in room_changes(log) if inside(c["t"], pad)]
    used: set[int] = set()
    out: list[dict] = []
    for r in hand:
        if r["note"] == "poke" and any(c is not r and c["note"] == "" and (c["from"], c["to"]) == (r["from"], r["to"])
                                       and abs((c["t"] - r["until_t"]).total_seconds()) <= 1 for c in hand):
            r["note"] = "linger"
    for r in [r for r in hand if r["note"] == "linger"]:
        if not any(c["note"] == "" and (c["from"], c["to"]) == (r["from"], r["to"])
                   and abs((c["t"] - r["until_t"]).total_seconds()) <= 1 for c in hand):
            hand.append({**r, "time": f"{r['until_t']:%H:%M:%S}", "t": r["until_t"], "note": ""})
    hand.sort(key=lambda r: r["t"])
    crossings =[r for r in hand if r["from"] not in ("START", "END") and inside(r["t"])]
    missed = matched = pokes_logged = 0
    for r in sorted(crossings, key=lambda r: r["note"] == "poke"):
        if r["note"] == "unsure":
            continue
        if r["note"] == "linger":
            out.append({"time": r["time"], "hand": f"{r['from']}->{r['to']}", "note": "linger",
                        "log": "", "result": f"stood in the opening until {r['until_t']:%H:%M:%S}, then crossed"})
            continue
        if r["note"] == "poke":
            i = take(changes, used, r["from"], r["to"], r["t"], a.tol)
            j = take(changes, used, r["to"], r["from"], r["until_t"], a.tol) if i is not None else None
            if i is not None and j is None:
                used.discard(i)
            got = i is not None and j is not None
            pokes_logged += got
            out.append({"time": r["time"], "hand": f"{r['from']}->{r['to']}", "note": "poke",
                        "log": changes[i]["ts"][11:23] if got else "", "result": "poke logged" if got else "poke not logged"})
            continue
        i = take(changes, used, r["from"], r["to"], r["t"], a.tol)
        matched += i is not None
        missed += i is None
        out.append({"time": r["time"], "hand": f"{r['from']}->{r['to']}", "note": r["note"],
                    "log": changes[i]["ts"][11:23] if i is not None else "", "result": "ok" if i is not None else "MISSED"})
    unsure_t = [r["t"] for r in crossings if r["note"] == "unsure"]
    wrong = near_unsure = at_edge = 0
    for i, c in enumerate(changes):
        if i in used or not inside(c["t"]):
            continue
        if any(abs((c["t"] - u).total_seconds()) <= a.tol for u in unsure_t):
            near_unsure += 1
            res = "near unsure"
        elif wins is not None and any(min(abs((c["t"] - s).total_seconds()), abs((c["t"] - e).total_seconds())) <= a.pad
                                      for s, e in wins):
            at_edge += 1
            res = "at clip edge"
        else:
            wrong += 1
            res = "WRONG"
        out.append({"time": "", "hand": "", "note": "", "log": c["ts"][11:23],
                    "result": f"{res} {c['from']}->{c['to']} ({c['reason']})"})

    hand_tl = [(t0, start["to"])] + [(r["t"], r["to"]) for r in crossings if r["note"] not in ("poke", "linger") and r["to"]]
    log_tl = [(r["t"], r["room"]) for r in log]
    step, t, right, total, no_room = timedelta(seconds=1), t0, 0, 0, 0
    while t < t1:
        total += 1
        room = room_at(log_tl, t)
        no_room += room is None
        right += room == room_at(hand_tl, t)
        t += step

    n = len([r for r in crossings if r["note"] not in ("poke", "linger", "unsure")])
    print(f"window {t0:%H:%M:%S}-{t1:%H:%M:%S}" + (f" ({len(wins)} windows only)" if wins else "")
          + f", tolerance {a.tol:.0f} s")
    print(f"crossings (hand)      {n}")
    print(f"  matched by the log  {matched}")
    print(f"  MISSED              {missed}")
    print(f"WRONG log changes     {wrong}")
    print(f"  counting pokes      {wrong + 2 * pokes_logged}   (pokes logged as out-and-back: {pokes_logged}"
          f" of {sum(r['note'] == 'poke' for r in crossings)})")
    print(f"lingers in an opening {sum(r['note'] == 'linger' for r in crossings)}   (poke row + crossing at its until time)")
    print(f"near an unsure row    {near_unsure}   (not counted; unsure rows: {len(unsure_t)})")
    if wins is not None:
        print(f"at a clip edge        {at_edge}   (not counted: within {a.pad:.0f} s of a window edge)")
    print(f"errors (missed + wrong) {missed + wrong}   (plan target: under 2)")
    if wins is None:
        print(f"time in the right room  {100 * right / max(total, 1):.1f}% of {total} s"
              + (f"   (log has no room yet for {no_room} s: civet not seen since the run started)" if no_room > 2 else ""))
    for o in out:
        if o["result"] != "ok":
            print(f"  {o['time']:>12} {o['hand']:>16}  log {o['log']:>12}  {o['result']}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["time", "hand", "note", "log", "result"])
            w.writeheader()
            w.writerows(out)
        print("details:", a.out)


if __name__ == "__main__":
    main()
