import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

CAMS = ["CAM01", "CAM02", "CAM03", "CAM04"]
FONT = "C\\:/Windows/Fonts/consola.ttf"


def seconds(hms: str) -> float:
    h, m, s = (float(v) for v in hms.split(":"))
    return h * 3600 + m * 60 + s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/april5_24h.yaml")
    ap.add_argument("--start", required=True, help="clock time HH:MM:SS")
    ap.add_argument("--end", required=True, help="clock time HH:MM:SS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ff = shutil.which("ffmpeg") or sys.exit("ffmpeg not found (winget install Gyan.FFmpeg)")
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(a.config).resolve().parents[1]
    t0, t1 = seconds(a.start), seconds(a.end)
    args, chains = [ff, "-hide_banner", "-y"], []
    for i, c in enumerate(CAMS):
        cam = cfg["cameras"][c]
        start = datetime.fromisoformat(cam["file_start"])
        off = t0 - (start - start.replace(hour=0, minute=0, second=0)).total_seconds()
        args += ["-ss", f"{off:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str((base / cam["source"]).resolve())]
        chains.append(f"[{i}:v]fps=15,scale=960:540,"
                      f"drawtext=fontfile='{FONT}':fontsize=30:fontcolor=yellow:box=1:boxcolor=black@0.6:"
                      f"x=10:y=h-44:text='{c}  %{{pts\\:hms\\:{t0:.0f}}}'[v{i}]")
    graph = ";".join(chains) + ";[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    script = out.with_suffix(".filter.txt")
    script.write_text(graph, encoding="utf-8")
    args += ["-/filter_complex", str(script), "-map", "[out]", "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "30", "-movflags", "+faststart", str(out)]
    subprocess.run(args, check=True)
    script.unlink()
    print("video:", out)


if __name__ == "__main__":
    main()
