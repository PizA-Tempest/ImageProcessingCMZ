"""Live detector demo: runs the civet pose model (models/civet_yolo26n.pt) and draws its
box + 6-point skeleton. Unlike detect_app.py, which shows the hand labels, everything
drawn here is the model's own prediction.

Run from this folder:  streamlit run demo_app.py --server.port 8504
"""
from __future__ import annotations

import random
import re
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from pose_draw import DEFAULT_WEIGHTS, KP_HEX, KP_NAMES, CivetPose, VideoOut, draw

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ANNOT = REPO / "data" / "annotations.csv"
SAMPLED = REPO / "data" / "frames" / "frames.csv"
LOCAL_VIDEOS = [ROOT.parent.parent / "civetwatch_data", ROOT / ".." / "data" / "videos"]

LABEL_BADGE = {
    "Walking": "orange", "Standing": "green", "Resting": "blue", "Eating": "orange",
    "Drinking": "blue", "Grooming": "violet", "Running": "red", "Other": "gray", "No Civet": "gray",
}

st.set_page_config(page_title="Civet Detector — Live demo", page_icon=":material/pets:", layout="wide")
st.markdown(
    """
<style>
h1[data-testid="stTitle"] {
  background: linear-gradient(90deg, #22c55e 0%, #38bdf8 55%, #a78bfa 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent !important;
}
div[data-testid="stImage"] img {
  border-radius: 14px; border: 1px solid rgba(148,163,184,.18);
  box-shadow: 0 10px 28px rgba(0,0,0,.45);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- model and data
@st.cache_resource
def load_model(weights: str) -> CivetPose:
    return CivetPose(weights)


@st.cache_data
def load_behaviour_index() -> list:
    """labelling/behaviour_*.csv bout lists -> [(start_sec, end_sec, [(sec, behaviour), ...])]."""
    pieces = []
    for f in sorted((ROOT / "labelling").glob("behaviour_*.csv")):
        m = re.search(r"_(\d{2})(\d{2})_(\d{2})(\d{2})\.csv$", f.name)
        if not m:
            continue
        start = int(m[1]) * 3600 + int(m[2]) * 60
        end = int(m[3]) * 3600 + int(m[4]) * 60
        bouts = []
        for r in pd.read_csv(f).itertuples(index=False):
            b = str(r.behaviour)
            if b in ("START", "END"):
                continue
            hh, mm, ss = map(int, str(r.time).split(":"))
            bouts.append((hh * 3600 + mm * 60 + ss, b))
        pieces.append((start, end, bouts))
    return pieces


def behaviour_at(pieces: list, sec: float) -> str:
    for start, end, bouts in pieces:
        if start <= sec <= end:
            cur = ""
            for t, b in bouts:
                if t <= sec:
                    cur = b
                else:
                    break
            return cur
    return ""


@st.cache_data
def load_sources() -> dict[str, pd.DataFrame]:
    out = {}
    if ANNOT.exists():
        a = pd.read_csv(ANNOT, encoding="utf-8-sig")
        a = a.assign(path=[str(REPO / p) for p in a.image_path])
        a = a[[Path(p).exists() for p in a.path]]
        if not a.empty:
            out["Labelled frames"] = a.rename(columns={"label": "tag"})[
                ["path", "camera", "source_video", "timestamp_seconds", "tag"]]
    if SAMPLED.exists():
        f = pd.read_csv(SAMPLED, encoding="utf-8-sig")
        f = f.assign(path=[str(REPO / p) for p in f.image_path], tag="")
        f = f[[Path(p).exists() for p in f.path]]
        if not f.empty:
            out["Sampled frames"] = f[["path", "camera", "source_video", "timestamp_seconds", "tag"]]
    man = ROOT / "labelling" / "manifest.csv"
    img_dir = ROOT / "labelling" / "images"
    if man.exists() and img_dir.exists():
        pieces = load_behaviour_index()
        rows = []
        for r in pd.read_csv(man).itertuples(index=False):
            p = img_dir / r.split / r.file
            if not p.exists():
                continue
            ts = pd.to_datetime(r.time)
            sec = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6
            beh = behaviour_at(pieces, sec)
            rows.append(dict(path=str(p), camera=r.cam, source_video=r.split,
                             timestamp_seconds=sec, tag=beh.capitalize() if beh else ""))
        if rows:
            out["Pose frames (labelling/images)"] = pd.DataFrame(rows)
    return out


def settings() -> dict:
    return dict(conf=st.session_state.conf, imgsz=st.session_state.imgsz,
                grey=st.session_state.grey, kp_thr=st.session_state.kp_thr,
                names=st.session_state.names, low=st.session_state.low)


@st.cache_data(max_entries=256, show_spinner=False)
def run_on_file(path: str, weights: str, conf: float, imgsz: int, grey: bool):
    img = cv2.imread(path)
    t0 = time.perf_counter()
    dets = load_model(weights)(img, conf=conf, imgsz=imgsz, grey_clahe=grey)
    ms = (time.perf_counter() - t0) * 1000
    return [(d.box, d.conf, d.kpts) for d in dets], ms


def annotate(img: np.ndarray, raw, s: dict) -> np.ndarray:
    from pose_draw import Detection
    dets = [Detection(b, c, k) for b, c, k in raw]
    return cv2.cvtColor(draw(img, dets, kp_thr=s["kp_thr"], show_names=s["names"], show_low=s["low"]),
                        cv2.COLOR_BGR2RGB)


def detection_panel(raw, ms: float, s: dict, tag: str = ""):
    with st.container(border=True):
        st.caption("Detector")
        if raw:
            st.badge(f"civet · {raw[0][1]:.2f}", color="green", icon=":material/check_circle:")
            if len(raw) > 1:
                st.caption(f"{len(raw)} boxes above the threshold (the first is drawn strongest)")
        else:
            st.badge("no civet", color="gray", icon=":material/visibility_off:")
        st.caption(f"{ms:.0f} ms on {load_model(st.session_state.weights).device}")
    if tag:
        with st.container(border=True):
            st.caption("Hand label / behaviour at this time")
            st.badge(tag, color=LABEL_BADGE.get(tag, "gray"), icon=":material/label:")
    if raw:
        k = raw[0][2]
        df = pd.DataFrame({"keypoint": KP_NAMES, "conf": k[:, 2].astype(float).round(2),
                           "x": k[:, 0].round().astype(int), "y": k[:, 1].round().astype(int)})
        df["shown"] = np.where(k[:, 2] >= s["kp_thr"], "solid", "hollow" if s["low"] else "hidden")
        st.dataframe(df, hide_index=True, width="stretch")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### :material/pets: Detector demo")
    st.text_input("Weights", str(DEFAULT_WEIGHTS), key="weights")
    st.slider("Box confidence", 0.05, 0.95, 0.25, 0.05, key="conf")
    st.slider("Keypoint confidence (solid dot)", 0.0, 1.0, 0.5, 0.05, key="kp_thr")
    st.select_slider("Image size", [480, 640, 800, 960], 640, key="imgsz")
    st.toggle("Grey + CLAHE input (as trained)", True, key="grey")
    st.toggle("Show low-confidence keypoints (hollow)", True, key="low")
    st.toggle("Keypoint names", False, key="names")
    legend = "<br>".join(f"<span style='color:{KP_HEX[i]}'>●</span> {n}" for i, n in enumerate(KP_NAMES))
    st.markdown(f"<small>{legend}<br><span style='color:#22c55e'>■</span> box = civet</small>",
                unsafe_allow_html=True)
    st.caption("Skeleton joins neighbouring keypoints along the midline when both are above the "
               "keypoint threshold. The model sees grey + CLAHE; the drawing is on the original frame.")

st.title("Civet detector — live demo")
st.caption("YOLO26n-pose fine-tuned on 1,023 hand-labelled frames from 5 Apr 2026 · one class (civet) · "
           "box + nose, eyes, neck, mid-back, tail base, tail tip · every box and dot here is the model's prediction")

try:
    with st.spinner("Loading pose model…"):
        model = load_model(st.session_state.weights)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model: {e}")
    st.stop()

s = settings()
tab_frames, tab_upload, tab_video = st.tabs([":material/photo_library: Frames in this repo",
                                             ":material/upload: Your image",
                                             ":material/movie: Video clip"])

# ---------------------------------------------------------------- frames in the repo
with tab_frames:
    sources = load_sources()
    if not sources:
        st.info("No frames found under ../data or labelling/images — use the other tabs.")
    else:
        c1, c2, c3 = st.columns([2, 2, 3])
        src = c1.segmented_control("Source", list(sources), default=list(sources)[0], key="src") or list(sources)[0]
        df = sources[src]
        cams = sorted(df.camera.unique())
        cam = c2.segmented_control("Camera", ["All"] + cams, default="All", key="cam") or "All"
        tags = sorted(t for t in df.tag.unique() if t)
        tag = c3.selectbox("Hand label", ["All"] + tags, key="tag") if tags else "All"
        sub = df
        if cam != "All":
            sub = sub[sub.camera == cam]
        if tag != "All":
            sub = sub[sub.tag == tag]
        sub = sub.sort_values(["source_video", "timestamp_seconds"]).reset_index(drop=True)
        if sub.empty:
            st.info("No frames match these filters.")
        else:
            key = f"pos::{src}::{cam}::{tag}"
            pos = min(int(st.session_state.get(key, 0)), len(sub) - 1)
            row = sub.iloc[pos]
            with st.spinner("Running detection…"):
                raw, ms = run_on_file(row.path, st.session_state.weights, s["conf"], s["imgsz"], s["grey"])
            img = cv2.imread(row.path)
            main, side = st.columns([3, 1], gap="large")
            with main:
                st.image(annotate(img, raw, s), width="stretch")
                st.caption(f"Frame {pos + 1} / {len(sub)} · {Path(row.path).name} · {row.source_video} · "
                           f"{row.timestamp_seconds:.0f} s")
                with st.expander("What the model sees (grey + CLAHE)", icon=":material/visibility:"):
                    st.image(cv2.cvtColor(model.preprocess(img, s["grey"]), cv2.COLOR_BGR2RGB), width="stretch")
            with side:
                detection_panel(raw, ms, s, row.tag)
                nav = st.columns(3, gap="small")
                if nav[0].button("Prev", icon=":material/arrow_back:", key="prev", width="stretch", disabled=pos == 0):
                    st.session_state[key] = pos - 1
                    st.rerun()
                if nav[1].button("Random", icon=":material/shuffle:", key="rand", width="stretch"):
                    st.session_state[key] = random.randrange(len(sub))
                    st.rerun()
                if nav[2].button("Next", icon=":material/arrow_forward:", key="next", width="stretch",
                                 disabled=pos >= len(sub) - 1):
                    st.session_state[key] = pos + 1
                    st.rerun()
                jump = st.number_input("Go to frame", 1, len(sub), pos + 1)  # no key: follows Prev/Next
                if jump - 1 != pos:
                    st.session_state[key] = int(jump) - 1
                    st.rerun()

# ---------------------------------------------------------------- uploaded image
with tab_upload:
    up = st.file_uploader("Image (jpg / png)", type=["jpg", "jpeg", "png"], key="up_img")
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        dets = model(img, conf=s["conf"], imgsz=s["imgsz"], grey_clahe=s["grey"])
        ms = (time.perf_counter() - t0) * 1000
        raw = [(d.box, d.conf, d.kpts) for d in dets]
        main, side = st.columns([3, 1], gap="large")
        main.image(annotate(img, raw, s), width="stretch")
        with side:
            detection_panel(raw, ms, s)

# ---------------------------------------------------------------- video clip
with tab_video:
    st.caption("Runs the model on a stretch of a recording and plays the result with box + skeleton. "
               "NVR files run at 15 fps whatever their header says.")
    local = sorted({str(p) for d in LOCAL_VIDEOS if d.exists() for p in d.glob("*.mp4")})
    c1, c2 = st.columns([3, 2])
    choice = c1.selectbox("Recording", ["Upload a file…"] + local, key="vid_choice")
    vpath = None
    if choice == "Upload a file…":
        upv = c1.file_uploader("Video (mp4)", type=["mp4", "mov", "avi", "mkv"], key="up_vid")
        if upv is not None:
            tmp = Path(tempfile.gettempdir()) / f"civet_demo_in_{upv.file_id}.mp4"
            if not tmp.exists():
                tmp.write_bytes(upv.getvalue())
            vpath = str(tmp)
    else:
        vpath = choice
    start = c2.number_input("Start (s from file start)", 0.0, value=0.0, step=10.0, key="vid_start")
    dur = c2.number_input("Duration (s)", 1.0, 120.0, 15.0, 1.0, key="vid_dur")
    fps = c2.select_slider("Model rate (fps)", [2.5, 5.0, 7.5, 15.0], 7.5, key="vid_fps")
    if st.button("Run detector", icon=":material/play_arrow:", type="primary", disabled=vpath is None):
        cap = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        every, n_need = max(1, round(15.0 / fps)), int(dur * 15)
        out_path = Path(tempfile.gettempdir()) / f"civet_demo_{int(time.time())}.mp4"
        vw, n, hits, t0 = None, 0, 0, time.perf_counter()
        bar = st.progress(0.0, text="Running…")
        while n < n_need:
            ok, frame = cap.read()
            if not ok:
                break
            if n % every == 0:
                dets = model(frame, conf=s["conf"], imgsz=s["imgsz"], grey_clahe=s["grey"])
                hits += bool(dets)
                t = start + n / 15.0
                out = draw(frame, dets, kp_thr=s["kp_thr"], show_names=s["names"], show_low=s["low"],
                           banner=f"{Path(vpath).stem}  t={int(t // 60):02d}:{t % 60:04.1f}")
                out = cv2.resize(out, (1280, round(out.shape[0] * 1280 / out.shape[1] / 2) * 2))
                if vw is None:
                    vw = VideoOut(out_path, fps, (out.shape[1], out.shape[0]))
                vw.write(out)
            n += 1
            bar.progress(min(1.0, n / n_need), text=f"{n / 15:.1f} / {dur:.0f} s")
        cap.release()
        if vw is None:
            bar.empty()
            st.error("No frames could be read from this file.")
        else:
            vw.close()
            bar.empty()
            frames_out = (n + every - 1) // every
            st.success(f"{frames_out} frames · civet detected in {hits} · {time.perf_counter() - t0:.0f} s "
                       f"on {model.device}")
            data = out_path.read_bytes()
            if vw.browser_playable:
                st.video(data)
            else:
                st.warning("ffmpeg not found, so the file is mp4v and may not play in the browser — "
                           "download it, or install ffmpeg (winget install Gyan.FFmpeg).")
            st.download_button("Download MP4", data, file_name=out_path.name, mime="video/mp4",
                               icon=":material/download:")