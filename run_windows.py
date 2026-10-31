"""Run the friend civetwatch pipeline over label-centred windows of our videos.

Windows are rebuilt from data/annotations.csv (pad 120 s, merge gap 300 s) so
every one of the 711 labels falls inside a processed window. Sessions run
sequentially (shared GPU), one subprocess per window; outputs append to
civet-monitor/outputs/our_dataset{N}.sqlite.

Usage:  python run_windows.py            (run from project root)
Logs:   winlogs/win_00_ds1.log ...      and winlogs/orchestrator.log
"""
from __future__ import annotations

import collections
import csv
import subprocess
import sys
import time
from pathlib import Path

PAD, GAP = 120.0, 300.0
ROOT = Path(__file__).resolve().parent
CM = ROOT / "civet-monitor"
LOGS = ROOT / "winlogs"


def build_windows():
    per = collections.defaultdict(list)
    with open(ROOT / "data/annotations.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            d = int(r["source_video"].split("_Dataset")[1].split(".")[0])
            per[d].append(float(r["timestamp_seconds"]))
    wins = []
    for d in sorted(per):
        iv = sorted((t - PAD, t + PAD) for t in per[d])
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s - merged[-1][1] <= GAP:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            wins.append((d, max(0.0, s), e - max(0.0, s)))
    return wins


def main():
    LOGS.mkdir(exist_ok=True)
    wins = build_windows()
    # Resume: skip windows whose log already ends with a completed report.
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if first == 0:
        for i in range(len(wins)):
            lg = LOGS / f"win_{i:02d}_ds{wins[i][0]}.log"
            if lg.exists() and "processed" in lg.read_text(errors="replace"):
                first = i + 1
    wins = wins[first:]
    total = sum(w[2] for w in wins)
    print(f"{len(wins)} windows remaining ({first} done), {total / 3600:.1f} h sim", flush=True)
    t_all = time.time()
    for j, (d, start, dur) in enumerate(wins):
        i = first + j
        log = LOGS / f"win_{i:02d}_ds{d}.log"
        cmd = [sys.executable, "-m", "civetwatch.run", "--offline",
               "--config", f"config/our_dataset{d}.yaml",
               "--start", f"{start:.1f}", "--duration", f"{dur:.1f}"]
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as lo:
            r = subprocess.run(cmd, cwd=CM, stdout=lo, stderr=subprocess.STDOUT)
        print(f"[{i:02d}/23] ds{d} start={start:.0f}s dur={dur:.0f}s "
              f"rc={r.returncode} wall={time.time() - t0:.0f}s", flush=True)
        if r.returncode != 0:
            print(f"  FAILED, see {log}", flush=True)
    print(f"ALL DONE in {(time.time() - t_all) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
