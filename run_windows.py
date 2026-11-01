"""Run the friend civetwatch pipeline per-video (single-camera sessions).

Our 4 camera videos are NOT time-synchronised (different durations), so the
multi-camera handoff thrashes. Each video runs as an independent single-camera
session using configs in civet-monitor/config/single/. Windows are rebuilt
from data/annotations.csv (pad 120 s, merge gap 300 s) so every label falls
inside a processed window. Videos are ordered by label count so the most
labelled footage is covered first if the run is interrupted.

Usage:  python run_windows.py [first_index]
Logs:   winlogs/win_NN_<video>.log  and winlogs/orchestrator.log
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


def build_jobs():
    per = collections.defaultdict(list)
    with open(ROOT / "data/annotations.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            per[r["source_video"]].append(float(r["timestamp_seconds"]))
    jobs = []
    for video in sorted(per, key=lambda v: -len(per[v])):
        stem = video.replace(".mp4", "")
        iv = sorted((t - PAD, t + PAD) for t in per[video])
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s - merged[-1][1] <= GAP:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            jobs.append((video, stem, max(0.0, s), e - max(0.0, s), len(per[video])))
    return jobs


def main():
    LOGS.mkdir(exist_ok=True)
    jobs = build_jobs()
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if first == 0:
        for i in range(len(jobs)):
            lg = LOGS / f"win_{i:02d}_{jobs[i][1]}.log"
            if lg.exists() and "processed" in lg.read_text(errors="replace"):
                first = i + 1
    jobs = jobs[first:]
    total = sum(j[3] for j in jobs)
    print(f"{len(jobs)} windows remaining ({first} done), {total / 3600:.1f} h sim", flush=True)
    t_all = time.time()
    for j, (video, stem, start, dur, nlab) in enumerate(jobs):
        i = first + j
        log = LOGS / f"win_{i:02d}_{stem}.log"
        cmd = [sys.executable, "-m", "civetwatch.run", "--offline",
               "--config", f"config/single/{stem}.yaml",
               "--start", f"{start:.1f}", "--duration", f"{dur:.1f}"]
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as lo:
            r = subprocess.run(cmd, cwd=CM, stdout=lo, stderr=subprocess.STDOUT)
        print(f"[{i:02d}/{first + len(jobs)}] {video} start={start:.0f}s dur={dur:.0f}s "
              f"rc={r.returncode} wall={time.time() - t0:.0f}s", flush=True)
        if r.returncode != 0:
            print(f"  FAILED, see {log}", flush=True)
    print(f"ALL DONE in {(time.time() - t_all) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
