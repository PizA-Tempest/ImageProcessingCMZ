"""Viewer for the stored keeper report (outputs/report_v3).

Run from this folder:  streamlit run report_app.py --server.port 8502
"""

import sqlite3
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REP = ROOT / "outputs" / "report_v3"

st.set_page_config(page_title="Civet Monitor — Keeper Report", page_icon="🐾", layout="wide")


@st.cache_data
def load_all():
    tb = pd.read_csv(REP / "time_budget.csv")
    hourly = pd.read_csv(REP / "hourly_activity.csv")

    lines = (REP / "room_use.csv").read_text().splitlines()
    cut = lines.index("")
    rooms = pd.read_csv(StringIO("\n".join(lines[:cut])))
    crosses = pd.read_csv(StringIO("\n".join(lines[cut + 1 :])))

    con = sqlite3.connect(REP / "events.sqlite")
    events = pd.read_sql_query(
        "select behavior, timestamp_start, timestamp_end, duration, confidence, camera_id "
        "from behavior_event order by timestamp_start",
        con,
    )
    con.close()
    events["start"] = pd.to_datetime(events["timestamp_start"], format="ISO8601").dt.strftime("%H:%M:%S")
    events["end"] = pd.to_datetime(events["timestamp_end"], format="ISO8601").dt.strftime("%H:%M:%S")
    events["min"] = (events["duration"] / 60).round(2)
    return tb, hourly, rooms, crosses, events


tb, hourly, rooms, crosses, events = load_all()

title_col, _ = st.columns([3, 1])
with title_col:
    st.title("Civet Monitor — Keeper Report")
    st.caption(
        "5 April 2026 · 17:00–23:59 · stored results from civet-monitor "
        "(outputs/report_v3, 597 behaviour bouts)"
    )

top_room = rooms.loc[rooms["minutes"].idxmax()]
beh_total = tb.groupby("behaviour")["total_min"].sum().sort_values(ascending=False)
seen_min = tb[tb.behaviour != "unknown"]["total_min"].sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Behaviour bouts", f"{len(events):,}")
m2.metric("Time with civet seen", f"{seen_min:.0f} min")
m3.metric("Top room", f"{top_room['room']} · {top_room['share_%']:.0f}%")
m4.metric("Most behaviour", f"{beh_total.index[0]} · {beh_total.iloc[0]:.0f} min")
m5.metric("Room changes", f"{crosses['count'].sum():,}")

tab_budget, tab_hourly, tab_rooms, tab_events, tab_charts = st.tabs(
    ["Time budget", "Hourly activity", "Rooms", "Behaviour bouts", "Original charts"]
)

with tab_budget:
    part = st.radio("Period", ["night", "day", "all"], horizontal=True, index=2)
    df = tb if part == "all" else tb[tb.part == part]
    by_beh = df.groupby("behaviour")["total_min"].sum().sort_values(ascending=False)
    room_cols = [c for c in tb.columns if c.startswith("ROOM")]
    by_room = df[room_cols].sum().rename(index={"ROOM01": "Room 1", "ROOM02": "Room 2", "ROOM03": "Room 3", "ROOM04": "Room 4"})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Minutes per behaviour")
        st.bar_chart(by_beh, width="stretch", horizontal=True)
    with c2:
        st.subheader("Minutes per room")
        st.bar_chart(by_room, width="stretch", horizontal=True)

    st.subheader("Table (minutes, behaviour × room)")
    show = df.drop(columns=["part"]).groupby("behaviour").sum()
    show = show.loc[show["total_min"].sort_values(ascending=False).index]
    st.dataframe(show.style.format("{:.1f}"), width="stretch")

with tab_hourly:
    st.subheader("Active / inactive / not seen per hour")
    chart = hourly.set_index("hour")[["active_min", "inactive_min", "not_seen_min"]]
    st.bar_chart(chart, width="stretch", stack="normalize", y_label="share of hour")
    st.dataframe(
        hourly.rename(
            columns={
                "active_min": "active min",
                "inactive_min": "inactive min",
                "not_seen_min": "not seen min",
                "main_room": "main room",
                "main_behaviour": "main behaviour",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with tab_rooms:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Time per room")
        st.bar_chart(rooms.set_index("room")["minutes"], width="stretch")
    with c2:
        st.subheader("Crossings between rooms")
        st.bar_chart(crosses.set_index("crossing")["count"], width="stretch")
    st.caption("Shares: " + " · ".join(f"{r['room']} {r['share_%']:.1f}%" for _, r in rooms.iterrows()))

with tab_events:
    behaviours = sorted(events.behavior.unique())
    pick = st.multiselect("Filter behaviour", behaviours, default=behaviours)
    view = events[events.behavior.isin(pick)].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Bouts shown", len(view))
    c2.metric("Total minutes", f"{view['min'].sum():.1f}")
    c3.metric("Mean confidence", f"{view['confidence'].mean():.2f}")
    st.dataframe(
        view[["start", "end", "behavior", "min", "confidence", "camera_id"]]
        .rename(columns={"behavior": "behaviour", "min": "minutes", "camera_id": "camera"}),
        width="stretch",
        hide_index=True,
    )

with tab_charts:
    st.image(REP / "timeline.png", caption="Behaviour timeline 17:00–24:00")
    st.image(REP / "time_budget.png", caption="Time budget chart")
