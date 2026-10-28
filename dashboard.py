import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Civet Behavior Dashboard', layout='wide')

st.html("""<style>
h1[data-testid="stTitle"] {
  background: linear-gradient(90deg, #34d399 0%, #38bdf8 55%, #a78bfa 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent !important;
}
div[data-testid="stImage"] img { border-radius: 12px; border: 1px solid rgba(148,163,184,.18); }
div[data-testid="stProgress"] div[role="progressbar"] { border-radius: 99px; }
</style>""")

st.title('Large-Spotted Civet — Behavior Monitoring')
st.caption('Frame-level behavior estimates from EfficientNet-B0 over YOLO-detected civets. '
           'A sampled frame is classified only when the civet is detected (else Not Visible/Idle).')

tab_overview, tab_behaviors, tab_timeline, tab_model = st.tabs(
    ['Overview', 'Behaviors', 'Timeline', 'Model performance'])


@st.cache_data
def load_predictions():
    p = Path('results/predictions.csv')
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ['camera', 'room', 'source_video', 'timestamp_seconds', 'behavior', 'confidence',
              'camera_status', 'civet_detected', 'detection_confidence']:
        if c not in df.columns:
            df[c] = 0 if c in ('timestamp_seconds', 'confidence', 'detection_confidence') else 'Unknown'
    df['timestamp_seconds'] = pd.to_numeric(df.timestamp_seconds, errors='coerce').fillna(0)
    df['confidence'] = pd.to_numeric(df.confidence, errors='coerce')
    df['detection_confidence'] = pd.to_numeric(df.detection_confidence, errors='coerce')
    df['civet_detected'] = df.civet_detected.astype(str).str.lower().isin(['true', '1', 'yes'])
    return df


@st.cache_data
def load_metrics():
    p = Path('results/metrics.json')
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


df = load_predictions()
metrics = load_metrics()

if df is None or df.empty:
    st.info('No predictions yet. Train a model, then run: `python predict_video.py --all`')
    st.stop()

seen = df[df.civet_detected]

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Cameras', int(df.camera.nunique()))
    c2.metric('Sampled frames', f'{len(df):,}')
    c3.metric('Civet present', f'{df.civet_detected.mean():.1%}')
    c4.metric('Detection confidence (mean)', f"{seen.detection_confidence.mean():.1%}" if len(seen) else 'N/A')

    st.subheader('Civet presence by camera')
    pres = df.groupby('camera').agg(
        sampled_frames=('civet_detected', 'size'),
        civet_frames=('civet_detected', 'sum'),
        presence_pct=('civet_detected', lambda s: 100 * s.mean()),
    ).round(1)
    p1, p2 = st.columns([2, 3])
    p1.bar_chart(pres['presence_pct'])
    p2.dataframe(pres.style.format({'presence_pct': '{:.1f}%'}), width='stretch')

    st.subheader('Activity by video')
    vid = df.groupby(['camera', 'source_video']).agg(
        hours=('timestamp_seconds', lambda s: round(s.max() / 3600, 1)),
        civet_frames=('civet_detected', 'sum'),
        presence_pct=('civet_detected', lambda s: round(100 * s.mean(), 1)),
    ).reset_index()
    st.dataframe(vid, hide_index=True, width='stretch')
    st.caption('CAM4_Dataset2 has 0 detections — the enclosure was empty during that recording.')

with tab_behaviors:
    if seen.empty:
        st.info('No civet detections in predictions yet.')
    else:
        st.subheader('Behavior distribution (civet-visible frames)')
        counts = seen.behavior.value_counts()
        b1, b2 = st.columns([3, 2])
        b1.bar_chart(counts)
        b2.dataframe(counts.rename('frames').to_frame().assign(
            pct=lambda t: (100 * t.frames / t.frames.sum()).round(1)),
            width='stretch')

        st.subheader('Behavior mix per camera')
        mix = pd.crosstab(seen.camera, seen.behavior, normalize='index').round(3) * 100
        st.bar_chart(mix)

        st.subheader('Feeding & drinking events')
        fc1, fc2 = st.columns(2)
        for name, col in [('Eating', fc1), ('Drinking', fc2)]:
            sub = seen[seen.behavior == name]
            with col.container(border=True):
                st.markdown(f'**{name}** — {len(sub)} frames')
                if len(sub):
                    st.dataframe(sub.groupby('source_video').size().rename('frames').to_frame(), width='stretch')
                else:
                    st.caption('none predicted')

with tab_timeline:
    st.subheader('Per-video timeline')
    t1, t2 = st.columns([1, 2])
    cam = t1.selectbox('Camera', sorted(df.camera.dropna().unique()))
    vids = sorted(df[df.camera == cam].source_video.unique())
    vidname = t2.selectbox('Video', vids)
    view = df[(df.camera == cam) & (df.source_video == vidname)].sort_values('timestamp_seconds').copy()
    view['video_time_min'] = view.timestamp_seconds / 60
    st.caption(f'{len(view):,} sampled frames · {view.civet_detected.sum():,} with civet '
               f'({100 * view.civet_detected.mean():.1f}%)')
    st.line_chart(view.set_index('video_time_min')['confidence'], color='#34d399')
    st.caption('Prediction confidence over video time (0 where no civet was detected).')
    beh = view[view.civet_detected].behavior.value_counts()
    if len(beh):
        st.write('**Behaviors in this video:** ' + ', '.join(f'{b}: {n}' for b, n in beh.items()))
    st.dataframe(view[view.civet_detected][['timestamp_seconds', 'behavior', 'confidence',
                                            'detection_confidence']].rename(columns={
        'detection_confidence': 'yolo_conf'}),
        hide_index=True, width='stretch')
    st.caption('Timestamps are seconds from video start; sampling every 5s.')

with tab_model:
    if metrics is None:
        st.info('No metrics yet. Run: `python train.py`')
    else:
        st.subheader('Held-out evaluation (group cross-validation by source video)')
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('CV accuracy', f"{metrics['test_accuracy']:.1%} ± {metrics['test_accuracy_std']:.1%}")
        m2.metric('CV macro F1', f"{metrics['test_macro_f1']:.3f} ± {metrics['test_macro_f1_std']:.3f}")
        m3.metric('Pooled accuracy', f"{metrics['pooled_accuracy']:.1%}")
        m4.metric('Labeled frames', f"{metrics.get('labeled_frames', 0):,}")
        st.caption('Each of the 11 videos is used as test exactly once; the classifier never '
                   'sees a video at test time that it trained on (no frame leakage).')

        s1, s2 = st.columns([3, 2])
        with s1:
            st.markdown('**Pooled out-of-fold confusion matrix**')
            if Path('results/confusion_matrix.png').exists():
                st.image('results/confusion_matrix.png', width='stretch')
        with s2:
            st.markdown('**Per-class report (pooled out-of-fold)**')
            rep = pd.DataFrame(metrics['classification_report']).T
            st.dataframe(rep.round(3), width='stretch')
            st.caption(f"Classes: {', '.join(metrics['classes'])}. "
                       'Running (1 frame) and Resting (1 frame) were excluded by the 3+ minimum; '
                       'No Civet frames are handled by the YOLO gate, not the classifier.')

        st.markdown('**Per-fold results**')
        st.dataframe(pd.DataFrame(metrics['fold_results'])[
            ['fold', 'n_train', 'n_val', 'n_test', 'accuracy', 'macro_f1', 'val_macro_f1']],
            hide_index=True, width='stretch')

with st.sidebar:
    st.markdown('### Pipeline')
    st.markdown("""
1. `scan_all_videos.py` — sample frames
2. `annotate_app.py` — label frames
3. `train.py` — group-CV training
4. `predict_video.py --all` — predictions
5. `dashboard.py` — this view
    """)
    if metrics is not None:
        st.caption(f"Model classes: {', '.join(metrics['classes'])}")

st.caption('Limitations: still images cannot perfectly separate Walking vs Standing; rare classes '
           '(Grooming/Other) have <10 labeled examples each. Predictions are estimates for monitoring, '
           'not verified ethogram records.')
