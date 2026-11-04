"""Web port of demo_video.py: run the civet pose detector on one clip — or a reel of several
clips — straight from the browser and download the annotated MP4.

Same engine as the CLI (demo_video.clip), so results match: box + 6-point skeleton drawn at the
model rate, banner with the source name and running timecode.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import streamlit as st

from demo_video import clip, seconds
from pose_draw import CivetPose, DEFAULT_WEIGHTS, VideoOut

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SRC_DIRS = [ROOT / "outputs" / "remux", REPO / "data" / "videos"]

st.set_page_config(page_title="Civet demo video", page_icon=":material/movie:", layout="wide")


def recordings() -> list[str]:
    return sorted({str(p) for d in SRC_DIRS if d.exists() for p in d.glob("*.mp4")})


def upload_tmp(up) -> str:
    tmp = Path(tempfile.gettempdir()) / f"civet_dv_{up.file_id}.mp4"
    if not tmp.exists():
        tmp.write_bytes(up.getvalue())
    return str(tmp)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### :material/tune: Render settings")
    st.text_input("Weights", str(DEFAULT_WEIGHTS), key="dv_weights")
    st.slider("Box confidence", 0.05, 0.95, 0.25, 0.05, key="dv_conf")
    st.slider("Keypoint confidence (solid dot)", 0.0, 1.0, 0.5, 0.05, key="dv_kp")
    st.select_slider("Image size", [480, 640, 800, 960], 640, key="dv_imgsz")
    st.select_slider("Model / output rate (fps)", [2.5, 5.0, 7.5, 15.0], 7.5, key="dv_fps")
    st.selectbox("Source rate (NVR headers)", [15.0, 25.0, 30.0], 0, key="dv_src_fps")
    st.select_slider("Output width", [640, 960, 1280], 1280, key="dv_width")
    st.toggle("Keypoint names next to dots", False, key="dv_names")
    st.toggle("Grey + CLAHE input (as trained)", True, key="dv_grey")
    st.caption("Everything runs causal, exactly like the CLI: inference at the chosen rate, "
               "drawing on every output frame.")


def build_args() -> argparse.Namespace:
    s = st.session_state
    return argparse.Namespace(conf=s.dv_conf, kp_thr=s.dv_kp, imgsz=s.dv_imgsz, fps=s.dv_fps,
                              source_fps=s.dv_src_fps, width=s.dv_width, names=s.dv_names,
                              no_clahe=not s.dv_grey, weights=s.dv_weights)


# ---------------------------------------------------------------- page
st.title("Civet detector — demo video (web)")
st.caption("The demo_video.py engine in the browser: pick one clip or build a reel of several — "
           "each becomes a stretch of the output MP4, banner = source name + timecode.")

st.session_state.setdefault("dv_clips", [])
clips: list[dict] = st.session_state.dv_clips

# ---------------------------------------------------------------- add clips
st.markdown("#### Clips")
opts = ["Upload a file…"] + recordings()
c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
src = c1.selectbox("Recording", opts, key="dv_src", label_visibility="collapsed",
                   placeholder="Recording…")
start_s = c2.text_input("Start (s or HH:MM:SS)", "0", key="dv_start")
dur = c3.number_input("Duration (s)", 1.0, 300.0, 20.0, 1.0, key="dv_dur")
add = c4.button("Add", type="primary", icon=":material/add:", key="dv_add")
if add:
    path = upload_tmp(st.session_state.dv_up) if src == "Upload a file…" else src
    if src == "Upload a file…" and st.session_state.get("dv_up") is None:
        st.warning("Choose a file first.")
    else:
        try:
            t = seconds(start_s)
            clips.append({"path": path, "start": t, "dur": float(dur)})
        except ValueError:
            st.warning(f"Bad start value: {start_s!r}")
up = st.file_uploader("…or upload a recording (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"],
                      key="dv_up") if src == "Upload a file…" else None

if clips:
    st.markdown("")
    for i, cl in enumerate(clips):
        r1, r2, r3, r4 = st.columns([5, 2, 2, 1])
        name = Path(cl["path"]).name
        r1.markdown(f"**{i + 1}.** {name}")
        r2.caption(f"start {cl['start']:.1f} s")
        r3.caption(f"{cl['dur']:.0f} s")
        if r4.button("✕", key=f"dv_rm{i}", help="Remove this clip"):
            clips.pop(i)
            st.rerun()
    if st.button("Clear all", icon=":material/delete:", key="dv_clear"):
        clips.clear()
        st.rerun()
else:
    st.info("No clips yet — pick a recording, set a start time and press **Add**. "
            "Several clips become one reel.")

# ---------------------------------------------------------------- render
run = st.button("Render MP4", type="primary", icon=":material/movie:", key="dv_run",
                disabled=not clips)
if run and clips:
    args = build_args()
    holder: dict = {}
    t0 = time.perf_counter()
    bar = st.progress(0.0, text="Loading model…")
    try:
        model = CivetPose(args.weights)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load the model: {e}")
        st.stop()
    out_path = Path(tempfile.gettempdir()) / f"civet_demo_video_{int(time.time())}.mp4"
    args.out = str(out_path)
    results = []
    try:
        for i, cl in enumerate(clips):
            label = Path(cl["path"]).name

            def prog(done, total, _i=i, _label=label):
                bar.progress(min(0.99, done / total),
                             text=f"clip {_i + 1}/{len(clips)} · {_label}: {done}/{total} frames")

            n, d = clip(model, holder, cl["path"], cl["start"], cl["dur"], args,
                        label=label, progress=prog)
            results.append((label, n, d))
    except SystemExit as e:
        st.error(str(e))
        st.stop()
    finally:
        if holder.get("vw"):
            holder["vw"].close()
    bar.empty()

    if not holder.get("vw"):
        st.error("No frames could be read — the files may start after the chosen time.")
    else:
        elapsed = time.perf_counter() - t0
        total = sum(r[1] for r in results)
        hits = sum(r[2] for r in results)
        data = out_path.read_bytes()
        st.success(f"{len(results)} clip(s) · {total} frames · civet detected in {hits} · "
                   f"{elapsed:.0f} s on {model.device}")
        for label, n, d in results:
            st.caption(f"{label}: {n} frames, detections in {d}")
        if not holder["vw"].browser_playable:
            st.warning("ffmpeg unavailable — file is mp4v and may not play in the browser.")
        st.video(data)
        st.download_button("Download MP4", data, file_name=out_path.name,
                           mime="video/mp4", type="primary",
                           icon=":material/download:", key="dv_dl")
