"""Frame viewer for civet-monitor: browse labelled frames with the YOLO-pose
box + 6 keypoints drawn on top, in the same style as annotate_app.py.

Run from this folder:  streamlit run detect_app.py --server.port 8503
"""

import io
import math
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "labelling" / "images"
LBL = ROOT / "labelling" / "labels"
MANIFEST = ROOT / "labelling" / "manifest.csv"

SPLITS = ["train", "val", "test", "train_gate"]
CAMS = ["CAM01", "CAM02", "CAM03", "CAM04"]
KP_NAMES = ["nose", "eyes", "neck", "mid_back", "tail_base", "tail_tip"]
KP_COLORS = ["#f87171", "#fb923c", "#fbbf24", "#34d399", "#22d3ee", "#60a5fa"]
BOX_COLOR = "#22c55e"
SKELETON = "#e2e8f6"

STATUS_STYLE = {
    "civet": ("green", "civet visible"),
    "empty": ("gray", "no civet"),
    "unlabelled": ("orange", "not hand-labelled"),
}

st.set_page_config(page_title="Civet Detector — Frames", page_icon=":material/pets:", layout="wide")


@st.cache_data
def load_frames() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    rows = []
    for r in df.itertuples(index=False):
        img = IMG / r.split / r.file
        if not img.exists():
            continue
        lbl = LBL / r.split / Path(r.file).with_suffix(".txt").name
        status, n_kp, has_box = "unlabelled", 0, False
        if lbl.exists():
            txt = lbl.read_text().strip()
            status = "civet" if txt else "empty"
            if txt:
                parts = txt.split()
                has_box = True
                vals = list(map(float, parts[5:]))
                n_kp = sum(1 for i in range(5, len(vals), 3) if vals[i] > 0)
        rows.append(
            dict(
                file=r.file, cam=r.cam, time=r.time, split=r.split,
                night=int(r.night), img=str(img), lbl=str(lbl) if lbl.exists() else "",
                status=status, n_kp=n_kp, has_box=has_box,
            )
        )
    return pd.DataFrame(rows)


@st.cache_data
def render(img_path: str, lbl_path: str) -> bytes:
    im = Image.open(img_path).convert("RGB")
    if lbl_path:
        txt = Path(lbl_path).read_text().strip()
        if txt:
            d = ImageDraw.Draw(im)
            w, h = im.size
            for line in txt.splitlines():
                p = line.split()
                cx, cy, bw, bh = map(float, p[1:5])
                x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                d.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=4)
                vals = list(map(float, p[5:]))
                pts = []
                for i in range(6):
                    x, y, v = vals[i * 3: i * 3 + 3]
                    px, py = x * w, y * h
                    if v == 0:
                        pts.append(None)
                        continue
                    pts.append((px, py))
                    r = 9 if v == 2 else 7
                    if v == 2:
                        d.ellipse([px - r, py - r, px + r, py + r], fill=KP_COLORS[i])
                    else:
                        d.ellipse([px - r, py - r, px + r, py + r], outline=KP_COLORS[i], width=3)
                seg = [q for q in pts if q]
                if len(seg) >= 2:
                    d.line(seg, fill=SKELETON, width=3)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


frames = load_frames()
if frames.empty:
    st.error("No frames found under labelling/images/ — run the pose labelling export first.")
    st.stop()

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

st.title("Civet detector — frame viewer")
st.caption(
    "1,709 frames from 5 Apr 2026 · hand labels for train + val "
    "(box + nose, eyes, neck, mid-back, tail base, tail tip) · test / gate frames not labelled yet"
)

c1, c2, c3, c4 = st.columns(4)
split_f = c1.segmented_control("Split", ["All"] + SPLITS, default="All", key="f_split", width="stretch") or "All"
cam_f = c2.segmented_control("Camera", ["All"] + CAMS, default="All", key="f_cam", width="stretch") or "All"
night_f = c3.segmented_control("Time", ["All", "night", "day"], default="All", key="f_night", width="stretch") or "All"
status_f = c4.segmented_control(
    "Label", ["All", "civet", "empty", "unlabelled"], default="All", key="f_status", width="stretch"
) or "All"

sub = frames
if split_f != "All":
    sub = sub[sub.split == split_f]
if cam_f != "All":
    sub = sub[sub.cam == cam_f]
if night_f != "All":
    sub = sub[sub.night == (1 if night_f == "night" else 0)]
if status_f != "All":
    sub = sub[sub.status == status_f]
sub = sub.sort_values("time").reset_index(drop=True)

k1, k2, k3 = st.columns(3)
k1.metric("Frames", len(sub))
k2.metric("With civet", int((sub.status == "civet").sum()))
k3.metric("Empty", int((sub.status == "empty").sum()))

legend = "  ·  ".join(
    f"<span style='color:{KP_COLORS[i]}\u25cf'>{n}</span>" for i, n in enumerate(KP_NAMES)
)
st.markdown(f"Keypoints: {legend} · box <span style='color:{BOX_COLOR}'>●</span> = civet", unsafe_allow_html=True)

tab_step, tab_gallery = st.tabs(["Step through", "Gallery"])


def frame_bytes(row) -> bytes:
    return render(row["img"], row["lbl"])


with tab_step:
    if sub.empty:
        st.info("No frames match these filters.")
    else:
        pos_key = "step_pos"
        if pos_key not in st.session_state or st.session_state[pos_key] >= len(sub):
            st.session_state[pos_key] = 0
        pos = min(int(st.session_state[pos_key]), len(sub) - 1)
        row = sub.iloc[pos]

        main, side = st.columns([3, 1], gap="large")
        with main:
            st.image(frame_bytes(row), width="stretch")
            st.caption(
                f"Frame {pos + 1} / {len(sub)} · {row.file} · {row.cam} · {row.time} · "
                f"{row.split} · {'night' if row.night else 'day'}"
            )
        with side:
            with st.container(border=True):
                st.caption("Label status")
                color, label = STATUS_STYLE[row.status]
                st.badge(label, color=color, icon=":material/check_circle:" if row.status == "civet" else ":material/help:")
                if row.has_box:
                    st.caption(f"Box + {row.n_kp}/6 keypoints visible")
                elif row.status == "empty":
                    st.caption("Labelled empty — no civet in this frame")
                else:
                    st.caption("No hand label yet (test / gate frames)")
            st.caption("Neighbours (±1, ±2 in this list)")
            nb = st.columns(2)
            for i, off in enumerate([-2, -1, 1, 2]):
                j = pos + off
                if 0 <= j < len(sub):
                    nb[i % 2].image(frame_bytes(sub.iloc[j]), caption=f"{off:+d}", width="stretch")
            nav = st.columns(2)
            if nav[0].button("Prev", key="step_prev", icon=":material/arrow_back:",
                             width="stretch", disabled=pos == 0):
                st.session_state[pos_key] = pos - 1
                st.rerun()
            if nav[1].button("Next", key="step_next", icon=":material/arrow_forward:",
                             width="stretch", disabled=pos >= len(sub) - 1):
                st.session_state[pos_key] = pos + 1
                st.rerun()


@st.dialog("Frame viewer", width="large")
def enlarge(row_idx: int):
    row = sub.iloc[row_idx]
    st.image(frame_bytes(row), width="stretch")
    st.caption(
        f"{row.file} · {row.cam} · {row.time} · {row.split} · "
        f"{'night' if row.night else 'day'} · {STATUS_STYLE[row.status][1]}"
        + (f" · {row.n_kp}/6 keypoints" if row.has_box else "")
    )
    if st.button("Close", key="enl_close", width="content"):
        st.rerun()


with tab_gallery:
    if sub.empty:
        st.info("No frames match these filters.")
    else:
        PAGE = 12
        pages = max(1, math.ceil(len(sub) / PAGE))
        page = min(int(st.session_state.get("gal_page", 0)), pages - 1)
        st.session_state.gal_page = page

        chunk = sub.iloc[page * PAGE:(page + 1) * PAGE]
        for r in range(0, len(chunk), 4):
            cols = st.columns(4)
            for c in range(4):
                if r + c >= len(chunk):
                    continue
                i = page * PAGE + r + c
                fr = sub.iloc[i]
                with cols[c]:
                    st.image(frame_bytes(fr), width="stretch")
                    cap1, cap2 = st.columns([3, 2])
                    cap1.caption(f"{fr.cam} · {fr.time[11:19]} · {fr.split}")
                    color, label = STATUS_STYLE[fr.status]
                    cap2.badge(label, color=color)
                    if st.button("Enlarge", key=f"enl_{i}", width="stretch",
                                 icon=":material/zoom_in:"):
                        enlarge(i)

        navc = st.columns([2, 3, 2])
        if navc[0].button("Previous", icon=":material/arrow_back:", width="stretch",
                          disabled=page == 0, key="gal_prev"):
            st.session_state.gal_page = page - 1
            st.rerun()
        navc[1].caption(f"Page {page + 1} of {pages}")
        if navc[2].button("Next", icon=":material/arrow_forward:", width="stretch",
                          disabled=page >= pages - 1, key="gal_next"):
            st.session_state.gal_page = page + 1
            st.rerun()

with st.sidebar:
    st.markdown("### :material/pets: Frame viewer")
    st.caption("Read-only viewer for civet-monitor's pose labels (labelling/images + labels).")
    st.caption("**Per split**")
    ov = frames.groupby("split").agg(
        frames=("file", "size"),
        civet=("status", lambda s: int((s == "civet").sum())),
        empty=("status", lambda s: int((s == "empty").sum())),
    ).reset_index()
    st.dataframe(ov, hide_index=True, width="stretch")
    with st.expander("Label format", icon=":material/info:"):
        st.markdown(
            "YOLO-pose: `0 cx cy w h` then `x y v` for nose, eyes, neck, mid-back, "
            "tail base, tail tip (2 = visible, 1 = occluded, 0 = outside frame). "
            "An empty file means a labelled frame with no civet."
        )
