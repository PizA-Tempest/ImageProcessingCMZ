# CMZ Project Full File Index

**Civet Behavior Monitoring · Chiang Mai Zoo · Phase 2**

All links and files, in the order the work was done. Items with a link open directly.
Repository: `github.com/PizA-Tempest/ImageProcessingCMZ`

---

## Shared resources

| Item | Where | What it is |
|---|---|---|
| GitHub repo | [github.com/PizA-Tempest/ImageProcessingCMZ](https://github.com/PizA-Tempest/ImageProcessingCMZ) | All code: `civet-monitor/` (pipeline, tools, apps), `labelling/`, `models/`, `outputs/`, `docs/`. |
| Live website | [piza-tempest.github.io/ImageProcessingCMZ](https://piza-tempest.github.io/ImageProcessingCMZ/) | Static viewer (GitHub Pages): key metrics, pipeline diagram, 8 sample detections, keeper report, held-out results. Opens instantly. |
| Live app · demo video | [cmz-demo-video.streamlit.app](https://cmz-demo-video.streamlit.app/) | Interactive demo video renderer: pick a recording or upload, build a clip or reel, watch the MP4, download it. |
| Live app · report | [cmz-phase2-report.streamlit.app](https://cmz-phase2-report.streamlit.app/) | Interactive keeper report: hourly activity, time budget, room use, bout timeline (17:00–24:00 run). |
| README | [README.md](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/README.md) | Full documentation: quick start, results table, pipeline commands, label formats, limitations. |
| Presentation | `CMZ_Phase2_Presentation.pptx` | Phase 2 slides (deck + assets kept out of the repo, submitted separately). |
| Footage originals | [Releases · footage-originals](https://github.com/PizA-Tempest/ImageProcessingCMZ/releases) | The 11 recordings/mosaics that exceed git's 100 MB limit (152 MB–1.6 GB). Same bytes are in the repo as lossless `*.part*.mp4` splits. |

---

## Step 1 · Footage and labels

| Item | Where | What it is |
|---|---|---|
| Demo recording | [data/videos/cam01_demo_0054_0114.mp4](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/data/videos/cam01_demo_0054_0114.mp4) | 2-minute H.264 clip (00:54–01:14) with the civet visible — used by the demo apps. |
| Remuxed recordings | [civet-monitor/outputs/remux/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/outputs/remux) | The 75-minute sample, camera 01–04, as playable parts (`part000…002`; originals on the Release). |
| Frames | [civet-monitor/labelling/images/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/labelling/images) | 1,709 hand-labelled frames (train 1,023 · val 192 · test 313 · gate 181), split by folder. |
| Frame index | [labelling/manifest.csv](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/labelling/manifest.csv) | Every frame: file, camera, clock time, split, night/day. |
| Pose labels | [labelling/labels/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/labelling/labels) | YOLO-pose txt per frame: box + 6 keypoints (nose, eyes, neck, mid_back, tail_base, tail_tip). |
| CVAT setup | [labelling/cvat_labels.json](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/labelling/cvat_labels.json), [labelling/prelabels/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/labelling/prelabels) | Label set and pre-label zips used for annotation. |
| Label converter | [tools/cvat_to_yolo_pose.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/tools/cvat_to_yolo_pose.py) | CVAT 1.1 export → YOLO-pose labels with train/val split. |
| Behaviour lists | [labelling/behaviour_*.csv](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/labelling) | One row per behaviour change: 5 train pieces (215 min) + 1 blind test piece (30 min), 11 behaviours. |
| Crossing lists | `labelling/crossings_*.csv` | Hand-counted room crossings (test 17:00–19:10, train, gate visits). |
| Clock windows | [windows/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/windows) | Labelled-piece time windows, for fast keypoint caching. |

---

## Step 2 · Train the pose model (detector)

| Item | Where | What it is |
|---|---|---|
| Training | [tools/train_pose.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/tools/train_pose.py), [config/train_pose.yaml](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/config/train_pose.yaml) | Fine-tunes YOLO26n-pose from COCO on our frames; grey + CLAHE input, imgsz 640. `--eval-only` to score. |
| Final weights | [models/civet_yolo26n.pt](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/models/civet_yolo26n.pt) | The deployed pose model: box + 6 keypoints, mAP50 0.981 (night 0.975 / day 0.985). |
| Starting weights | `models/yolo26n-pose.pt`, `models/yolo26s.pt` | Stock COCO weights for training and box pre-labelling. |
| Training curves | [outputs/train/civet_pose_v1/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/outputs/train/civet_pose_v1) | Curves, confusion matrix, validation predictions. |
| Detection cache | [tools/cache_dets.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/tools/cache_dets.py) → [outputs/cache_yolo*](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/outputs) | Per-camera stored detections/keypoints (75-min, train block, test block) — everything below runs from these, no footage needed. |

---

## Step 3 · Room tracking

| Item | Where | What it is |
|---|---|---|
| Pipeline package | [civetwatch/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/civetwatch) | Stream reader → detect → ByteTrack → zones → handoff (which room, or hidden) → behaviour → SQLite. All causal. |
| Configs | [config/system.yaml](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/config/system.yaml), [config/april5_24h.yaml](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/config/april5_24h.yaml), [config/zones.json](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/config/zones.json) | Settings for the 75-min sample / 24 h run; floor, door and gate polygons per camera. |
| Replay command | README · “Pipeline commands” | `python -m civetwatch.run --config config/april5_24h.yaml --offline --cache outputs/cache_yolo …` (test block replay). |
| Crossing scorer | [tools/score_crossings.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/tools/score_crossings.py) | Scores the room log against the hand-counted list: 10 errors in 22 crossings, all at the open D23 gate. |
| Mosaic videos | [outputs/crossings/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/outputs/crossings) | 4-camera label videos (as lossless parts) + `score_1700_1910.csv`. |
| Gate visits | [outputs/gate_visits/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/outputs/gate_visits) | 66 gate-visit clips + tables (score_base, room log, SQLite). |

---

## Step 4 · Behaviour model

| Item | Where | What it is |
|---|---|---|
| Training | [tools/train_behaviour.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/tools/train_behaviour.py) | Extract → train → eval. Causal dilated 1D CNN: 44 pose/kinematic features, 5.3 s window, label every 0.53 s; five nets averaged. |
| Final weights | [models/behaviour_v3.pt](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/models/behaviour_v3.pt) | 5-net ensemble + detection fallback + 2 s hold — set in both configs. |
| Features | [outputs/behaviour/feat_fallback/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/outputs/behaviour) | Per-piece feature files (train pieces + test piece) used by every run below. |
| Test scores | `outputs/behaviour/test_v3/` | Test piece 23:00–23:30: accuracy 0.80, macro F1 0.48 (0.65 without marking/defecation), walking 0.90. |
| Validation scores | `outputs/behaviour/exp/` | Settings tried; validation accuracy 0.84, macro F1 0.67 (5 nets, hold-out parts). |
| Behaviour 3-class (Phase 2 classifier) | [models/behavior_classifier.pt](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/models/behavior_classifier.pt) | Separate root-side EfficientNet-B0 classifier (CMZ Phase 2 deliverable, CV 69.0% ± 5.1). |

---

## Step 5 · Keeper report

| Item | Where | What it is |
|---|---|---|
| Report builder | [tools/behaviour_report.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/tools/behaviour_report.py) | From the SQLite replay: hourly activity, time budget, room use, timeline. |
| Report output | [outputs/report_v3/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/civet-monitor/outputs/report_v3) | Evening 17:00–24:00: 597 bouts / 420 min, Room 2 = 65%; CSVs + timeline and time-budget charts. |
| Report viewer | [report_app.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/report_app.py) | Streamlit app: report tables and charts — **live: [cmz-phase2-report.streamlit.app](https://cmz-phase2-report.streamlit.app/)** (or run locally / deploy yourself). |

---

## Step 6 · Web apps (Streamlit)

Run any of them: `streamlit run civet-monitor/<file>.py --server.port 850x` — or deploy on [share.streamlit.io](https://share.streamlit.io) with repository `PizA-Tempest/ImageProcessingCMZ`, branch `main`, file path as listed.

| Item | Where | What it is |
|---|---|---|
| Detector demo | [demo_app.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/demo_app.py) | Frames in the repo / your image / video clip tabs — pose model runs live, box + keypoints drawn, behaviour tag matched from the hand labels. |
| Demo video renderer | [demo_video_app.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/demo_video_app.py) | Web port of the CLI: one clip or a **reel** of several clips → one MP4, progress bar, download — **live: [cmz-demo-video.streamlit.app](https://cmz-demo-video.streamlit.app/)**. |
| CLI renderer | [demo_video.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/demo_video.py) | Terminal version: `python demo_video.py VIDEO --start 00:53:56 --duration 20` (reel: `--reel a.mp4@3236@12 …`). |
| Frame browser | [detect_app.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/detect_app.py) | Browse all 1,709 labelled frames with detector overlay and hand labels. |
| Shared engine | [pose_draw.py](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/civet-monitor/pose_draw.py) | Loads the model the pipeline way (grey + CLAHE), draws box + skeleton — used by every demo. |
| Legacy viewers | `dashboard.py`, `annotate_app.py` | Early local viewers (frame store removed; kept for reference). |

---

## Step 7 · Static website (GitHub Pages)

| Item | Where | What it is |
|---|---|---|
| Page | [docs/index.html](https://github.com/PizA-Tempest/ImageProcessingCMZ/blob/main/docs/index.html) | The instant-loading site: metric cards, pipeline, sample detections, keeper report, results, limitations. |
| Sample gallery | [docs/samples/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/docs/samples) | 8 annotated JPEGs (4 cameras, day + night) generated with `tmp_make_site_samples.py`. |
| Report charts | [docs/report/](https://github.com/PizA-Tempest/ImageProcessingCMZ/tree/main/docs/report) | Timeline + time-budget PNGs for the site. |
| Live URL | [piza-tempest.github.io/ImageProcessingCMZ](https://piza-tempest.github.io/ImageProcessingCMZ/) | Settings → Pages → branch `main`, folder `/docs`. |

---

## Step 8 · Evaluate and present

| Item | Where | What it is |
|---|---|---|
| Results table | [README · Results](https://github.com/PizA-Tempest/ImageProcessingCMZ#results-held-out-test-block-17002400) | Held-out test block 17:00–24:00: detection, pose, room time, behaviour F1s, bout scores, speed. |
| Evaluation protocol | README · same section | Split by time block, never by frame; nothing trained or tuned on the test block. |
| Limitations | [README · Limitations](https://github.com/PizA-Tempest/ImageProcessingCMZ#limitations-state-these-in-the-presentation) | One day of footage, rare behaviours, night rests, D23 gate, bout edges — state these in the presentation. |
| Slides | `CMZ_Phase2_Presentation.pptx` | Phase 2 presentation deck (submitted separately; not in the repo). |

> **Using the live apps:** both run on Streamlit Community Cloud and sleep after idle — the first open takes ~30–60 s to wake (a bit longer right after an update or rebuild); if the page looks stuck, refresh once. They start from the stored results in this repo, so no footage download is needed to browse them; only the demo renderer actually runs the model (CPU, allow ~1 min per rendered clip).
