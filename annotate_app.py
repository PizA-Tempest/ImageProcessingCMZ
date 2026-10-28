from pathlib import Path
import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Civet Behavior Labeler", page_icon=":material/pets:", layout="wide")

QUEUE = Path("data/review_queue.csv")
OUTFILE = Path("data/annotations.csv")
CAMERAS = ["CAM1", "CAM2", "CAM3", "CAM4"]
DATASETS = ["Dataset1", "Dataset2", "Dataset3"]

BEH = {
    "Resting":   {"color": "#60a5fa", "badge": "blue",   "key": "1", "hint": "Lying down, body on the ground"},
    "Standing":  {"color": "#34d399", "badge": "green",  "key": "2", "hint": "Upright on legs, stationary"},
    "Walking":   {"color": "#fbbf24", "badge": "orange", "key": "3", "hint": "Calm locomotion, mid-stride"},
    "Eating":    {"color": "#fb923c", "badge": "orange", "key": "4", "hint": "Muzzle at food dish or food on ground"},
    "Drinking":  {"color": "#22d3ee", "badge": "blue",   "key": "5", "hint": "Muzzle at the round water bowl"},
    "Grooming":  {"color": "#f472b6", "badge": "violet", "key": "6", "hint": "Licking or scratching itself"},
    "Running":   {"color": "#f87171", "badge": "red",    "key": "7", "hint": "Fast locomotion, heavy blur"},
    "Other":     {"color": "#a78bfa", "badge": "violet", "key": "8", "hint": "Digging, jumping, climbing, playing"},
    "No Civet":  {"color": "#64748b", "badge": "gray",   "key": "9", "hint": "No animal visible in the frame"},
}
NAMES = list(BEH)


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_queue() -> pd.DataFrame:
    if not QUEUE.exists():
        return pd.DataFrame()
    q = pd.read_csv(QUEUE).fillna("")
    missing = [p for p in q["image_path"] if not Path(p).exists()]
    if missing:
        q = q[~q["image_path"].isin(missing)]
    if OUTFILE.exists():
        ann = pd.read_csv(OUTFILE).fillna("")
        labels = dict(zip(ann["image_path"], ann["label"]))
        q["label"] = q["image_path"].map(labels).fillna("")
    else:
        q["label"] = ""
    return q.sort_values(["source_video", "timestamp_seconds"]).reset_index(drop=True)


def persist(q: pd.DataFrame) -> None:
    labeled = q[q["label"].astype(str).str.strip() != ""]
    labeled.to_csv(OUTFILE, index=False, encoding="utf-8-sig")


def set_label(q: pd.DataFrame, path: str, label: str) -> None:
    q.loc[q["image_path"] == path, "label"] = label


def build_css() -> str:
    parts = ["""<style>
h1[data-testid="stTitle"] {
  background: linear-gradient(90deg, #34d399 0%, #38bdf8 55%, #a78bfa 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent !important;
}
div[data-testid="stImage"] img {
  border-radius: 14px;
  border: 1px solid rgba(148,163,184,.18);
  box-shadow: 0 10px 28px rgba(0,0,0,.45);
}
div[data-testid="stProgress"] div[role="progressbar"] { border-radius: 99px; }
div[data-testid="stBadge"] { border-radius: 999px; }
"""]
    for name, cfg in BEH.items():
        s = slug(name)
        c = cfg["color"]
        parts.append(f"""
.st-key-bh_{s} button {{
  background: {c} !important;
  color: #0b1220 !important;
  border: 1px solid rgba(255,255,255,.22) !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  border-radius: 12px !important;
  transition: transform .12s ease, filter .12s ease, box-shadow .12s ease !important;
}}
.st-key-bh_{s} button:hover {{
  filter: brightness(1.12) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 18px {c}55 !important;
}}
.st-key-bh_{s} button:active {{ transform: translateY(0) scale(.97) !important; }}
.st-key-cell_{s}_sel {{
  outline: 2px solid {c};
  border-radius: 14px;
  padding: 5px;
  box-shadow: 0 0 0 4px {c}26;
}}
.st-key-chip_{s} button {{
  background: transparent !important;
  color: {c} !important;
  border: 1.5px solid {c}aa !important;
  border-radius: 999px !important;
  font-weight: 600 !important;
  transition: transform .12s ease, filter .12s ease, box-shadow .12s ease !important;
}}
.st-key-chip_{s} button:hover {{ filter: brightness(1.2) !important; transform: translateY(-1px) !important; }}
.st-key-chip_{s}_sel button {{
  background: {c} !important;
  color: #0b1220 !important;
  box-shadow: 0 4px 14px {c}66 !important;
}}
""")
    parts.append("""
.st-key-nav_prev button, .st-key-nav_next button {
  border-radius: 10px !important; font-weight: 600 !important;
}
.st-key-enlarge button { border-radius: 10px !important; }
[data-testid="stDialog"] img { box-shadow: none; }
</style>""")
    return "".join(parts)


queue = load_queue()
if queue.empty:
    st.error("data/review_queue.csv not found or empty. Run scan_all_videos.py first.")
    st.stop()

st.html(build_css())

n_labeled = int((queue["label"].astype(str).str.strip() != "").sum())
videos_done = int((queue.groupby("source_video")["label"].apply(lambda s: (s.astype(str).str.strip() != "").all())).sum())

st.title("Civet behavior labeling")
st.caption("Pick a camera and dataset, label frames with one click (or keys 1-9), "
           "then browse your labels in the gallery.")

h1, h2, h3 = st.columns(3)
h1.metric("Labeled", f"{n_labeled} / {len(queue)}")
h2.metric("Remaining", len(queue) - n_labeled)
h3.metric("Videos finished", f"{videos_done} / {queue['source_video'].nunique()}")
st.progress(n_labeled / max(len(queue), 1),
            text=f"Overall progress — {n_labeled} of {len(queue)} frames")

tab_label, tab_gallery = st.tabs(["Label frames", "Browse gallery"])


def next_unlabeled(vid: pd.DataFrame, after: int) -> int:
    order = list(range(after + 1, len(vid))) + list(range(0, after + 1))
    for i in order:
        if str(vid.iloc[i]["label"]).strip() == "":
            return i
    return min(after, len(vid) - 1)


with tab_label:
    c1, c2, c3 = st.columns([3, 3, 5])
    cam = c1.segmented_control("Camera", CAMERAS, default="CAM1", key="sel_cam", width="stretch") or "CAM1"
    ds = c2.segmented_control("Dataset", DATASETS, default="Dataset1", key="sel_ds", width="stretch") or "Dataset1"
    video = f"{cam}_{ds}.mp4"
    vid = queue[queue["source_video"] == video].sort_values("timestamp_seconds").reset_index(drop=True)

    with c3.container(border=True):
        vc1, vc2 = st.columns([3, 2])
        vc1.markdown(f"**:material/video_camera: {video}**")
        vlab = int((vid["label"].astype(str).str.strip() != "").sum()) if len(vid) else 0
        vc2.progress(vlab / max(len(vid), 1), text=f"{vlab}/{len(vid)}")

    if vid.empty:
        st.info(f"No queued frames for {video} — the civet was not detected in this clip. "
                "Try another dataset or camera.")
    else:
        pos_key = f"pos_{video}"
        if pos_key not in st.session_state or not isinstance(st.session_state[pos_key], int) \
                or st.session_state[pos_key] >= len(vid):
            st.session_state[pos_key] = next_unlabeled(vid, -1)
        pos = st.session_state[pos_key]
        row = vid.iloc[pos]
        current = str(row["label"]).strip()

        main, side = st.columns([3, 1], gap="large")
        with main:
            st.image(row["image_path"], width="stretch")
            info = f"Frame {pos + 1} / {len(vid)}"
            st.caption(f"{info} · {row['source_video']} · t = {row['timestamp_seconds']:.0f}s · "
                       f"YOLO hint: {row['top_class']} {row['top_conf']:.2f} (unreliable)")

        with side:
            with st.container(border=True):
                st.caption("Current label")
                if current:
                    st.badge(current, color=BEH.get(current, {}).get("badge", "gray"),
                             icon=":material/check_circle:")
                else:
                    st.badge("Unlabeled", color="gray", icon=":material/pending:")
                st.caption(BEH.get(current, {}).get("hint", "Click a behavior below. Keys 1-9 also work."))
            st.caption("Neighbors (-2 to +2 steps)")
            nb = st.columns(2)
            for i, off in enumerate([-2, -1, 1, 2]):
                j = pos + off
                if 0 <= j < len(vid):
                    nb[i % 2].image(vid.iloc[j]["image_path"], caption=f"{off:+d} step", width="stretch")
            nav = st.columns(2)
            if nav[0].button("Prev", key="nav_prev", icon=":material/arrow_back:",
                             width="stretch", disabled=pos == 0):
                st.session_state[pos_key] = pos - 1
                st.rerun()
            if nav[1].button("Next", key="nav_next", icon=":material/arrow_forward:",
                             width="stretch", disabled=pos >= len(vid) - 1):
                st.session_state[pos_key] = pos + 1
                st.rerun()
            if current and st.button("Clear label", width="stretch", icon=":material/undo:",
                                     key="clear_label"):
                set_label(queue, row["image_path"], "")
                persist(queue)
                st.toast("Label cleared", icon=":material/undo:")
                st.rerun()

        st.caption("What is the civet doing? Click a button or press its number key.")
        clicked = None
        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                name = NAMES[r * 3 + c]
                cfg = BEH[name]
                selected = current == name
                with cols[c]:
                    with st.container(key=f"cell_{slug(name)}_sel" if selected else f"cell_{slug(name)}"):
                        if st.button(
                            f"{'✓ ' if selected else ''}{name}",
                            key=f"bh_{slug(name)}",
                            shortcut=cfg["key"],
                            help=f"Shortcut key {cfg['key']} — {cfg['hint']}",
                            width="stretch",
                        ):
                            clicked = name

        if clicked:
            set_label(queue, row["image_path"], clicked)
            persist(queue)
            st.session_state[pos_key] = next_unlabeled(vid, pos)
            st.toast(f"Saved: {clicked}", icon=":material/check_circle:")
            st.rerun()

        st.caption(f"Tip: consecutive frames are ~30s apart — the behavior is often the same, "
                   f"but check each frame. Use **Clear label** in the side panel to fix mistakes.")


@st.dialog("Frame viewer", width="large")
def frame_dialog(path: str, meta: str, current_label: str, row_key: str):
    st.image(path, width="stretch")
    st.caption(meta)
    pick = st.selectbox("Label", NAMES,
                        index=NAMES.index(current_label) if current_label in NAMES else 0,
                        key=f"dlg_sel_{row_key}")
    c1, c2 = st.columns(2)
    if c1.button("Save label", type="primary", width="stretch", key="dlg_save"):
        set_label(queue, path, pick)
        persist(queue)
        st.toast(f"Saved: {pick}", icon=":material/check_circle:")
        st.rerun()
    if c2.button("Close", width="stretch", key="dlg_close"):
        st.rerun()


with tab_gallery:
    gal_beh = st.session_state.get("gal_beh", "All")

    st.caption("Filter by behavior")
    with st.container(horizontal=True, gap="small"):
        all_sel = gal_beh == "All"
        if st.button(f"All ({n_labeled})", key="chip_all", width="content",
                     type="primary" if all_sel else "secondary"):
            st.session_state.gal_beh = "All"
            st.session_state.pop("gal_sig", None)
            st.rerun()
        counts = queue.loc[queue["label"] != "", "label"].value_counts()
        for name in NAMES:
            s = slug(name)
            sel = gal_beh == name
            with st.container(key=f"chip_{s}_sel" if sel else f"chip_{s}"):
                if st.button(f"{name} ({int(counts.get(name, 0))})",
                             key=f"chip_btn_{s}", width="content"):
                    st.session_state.gal_beh = name
                    st.session_state.pop("gal_sig", None)
                    st.rerun()

    f1, f2, f3 = st.columns([2, 3, 3])
    cam_g = f1.pills("Camera", ["All"] + CAMERAS, key="gal_cam", width="stretch") or "All"
    vid_opts = ["All videos"]
    cams = CAMERAS if cam_g == "All" else [cam_g]
    for c in cams:
        for d in DATASETS:
            vname = f"{c}_{d}.mp4"
            if (queue["source_video"] == vname).any():
                vid_opts.append(vname)
    vid_g = f2.selectbox("Video", vid_opts, key=f"gal_vid_{cam_g}", width="stretch")

    sub = queue.copy()
    if gal_beh != "All":
        sub = sub[sub["label"] == gal_beh]
    if cam_g != "All":
        sub = sub[sub["camera"] == cam_g]
    if vid_g != "All videos":
        sub = sub[sub["source_video"] == vid_g]
    sub = sub.reset_index(drop=True)

    with f3.container(border=True):
        st.markdown(f"**{len(sub)} frames**" + (f" · {gal_beh}" if gal_beh != "All" else " · all labels"))

    sig = (gal_beh, cam_g, vid_g)
    if st.session_state.get("gal_sig") != sig:
        st.session_state.gal_sig = sig
        st.session_state.gal_page = 0

    PAGE = 24
    pages = max(1, math.ceil(len(sub) / PAGE))
    page = min(max(int(st.session_state.get("gal_page", 0)), 0), pages - 1)
    st.session_state.gal_page = page

    if sub.empty:
        st.info("No frames match this filter. Label some frames first, or change the filters.")
    else:
        chunk = sub.iloc[page * PAGE:(page + 1) * PAGE]
        for r in range(0, len(chunk), 4):
            cols = st.columns(4)
            for c in range(4):
                if r + c >= len(chunk):
                    continue
                i = page * PAGE + r + c
                fr = sub.iloc[i]
                with cols[c]:
                    st.image(fr["image_path"], width="stretch")
                    cap1, cap2 = st.columns([3, 2])
                    cap1.caption(f"{fr['source_video'].replace('.mp4', '')} · {fr['timestamp_seconds']:.0f}s")
                    lb = str(fr["label"]).strip()
                    if lb:
                        cap2.badge(lb, color=BEH.get(lb, {}).get("badge", "gray"))
                    if st.button("Enlarge", key=f"enlarge_{i}", width="stretch",
                                 icon=":material/zoom_in:", use_container_width=None):
                        frame_dialog(
                            fr["image_path"],
                            f"{fr['source_video']} · t = {fr['timestamp_seconds']:.0f}s · {fr['camera']}",
                            lb, str(i),
                        )

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


@st.dialog("Reset all labels?")
def confirm_reset():
    st.warning("This deletes every label saved so far. This cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Delete everything", type="primary", width="stretch", key="reset_yes"):
        OUTFILE.unlink(missing_ok=True)
        st.toast("All labels deleted", icon=":material/delete:")
        st.rerun()
    if c2.button("Cancel", width="stretch", key="reset_no"):
        st.rerun()


with st.sidebar:
    st.markdown("### :material/pets: Civet labeler")
    st.caption("Frames come from YOLO pre-filtering of all 12 videos "
               "(425 candidate frames across 4 cameras).")

    st.caption("**Dataset overview**")
    ov = queue.groupby("source_video").agg(
        frames=("label", "size"),
        labeled=("label", lambda s: int((s.astype(str).str.strip() != "").sum())),
    ).reset_index().rename(columns={"source_video": "video"})
    ov["video"] = ov["video"].str.replace(".mp4", "", regex=False)
    ov["done"] = ov["labeled"] == ov["frames"]
    st.dataframe(ov, hide_index=True, width="stretch",
                 column_config={"done": st.column_config.CheckboxColumn("done")})

    with st.expander("Labeling tips", icon=":material/lightbulb:"):
        st.markdown("""
- **Lying down** → Resting
- **Still on feet** → Walking vs Standing: mid-stride or blur = Walking
- **Round metal bowl** → Drinking (water)
- **Flat food dish / food on ground** → Eating
- **Keys 1-9** label instantly
- Unclear but animal present → pick the closest, or **Other**
        """)

    st.space("large")
    if st.button("Reset all labels", width="stretch", icon=":material/delete:", key="reset_btn"):
        confirm_reset()
