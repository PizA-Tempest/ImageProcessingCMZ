import argparse
import shutil
import subprocess
import sys


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: winget install Gyan.FFmpeg  (or pip install imageio-ffmpeg)")


def codec(ff: str, src: str) -> str:
    probe = ff.replace("ffmpeg", "ffprobe")
    try:
        out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=codec_name", "-of", "csv=p=0", src],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        if out:
            return out
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "hevc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--h264", action="store_true", help="re-encode to H.264 (slow, plays everywhere)")
    a = ap.parse_args()
    ff = ffmpeg_exe()
    if a.h264:
        cmd = [ff, "-y", "-i", a.src, "-map", "0:v", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.dst]
    else:
        cmd = [ff, "-y", "-fflags", "+genpts", "-i", a.src, "-map", "0:v", "-c", "copy"]
        if codec(ff, a.src) == "hevc":
            cmd += ["-tag:v", "hvc1"]
        cmd += ["-movflags", "+faststart", a.dst]
    print(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
