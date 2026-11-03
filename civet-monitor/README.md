# Civet Behavior Monitoring (civet-monitor)

Multi-camera pipeline for the large-spotted civet (ชะมดแผงสันหางดำ) A01: 4 rooms in a row, one corner
camera per room, 24 hours of 5 April 2026 (HEVC 1080p, 15 fps, night near-infrared and day colour).
The system finds the civet and 6 body keypoints with a fine-tuned YOLO26n-pose, follows it between
rooms with a rule-based handoff, names 11 behaviours with five averaged causal 1D CNNs over 5.3 s of pose
features, and writes each behaviour bout to SQLite as it happens. Keeper tables and charts are built from that log.

## Results (held-out test block 17:00–24:00)

| Metric | Value |
|---|---|
| Labeled data | 1,215 pose frames (1,023 train + 192 validation); 245 min of behaviour bouts (215 train in 5 lists + 30 test) |
| Detection, box mAP50 (validation) | **0.981** (night 0.975, day 0.985) |
| Pose, PCK@0.1 head / tail (validation) | 0.876 / 0.786 |
| Time in the right room (test 17:00–19:10) | **94.8%** |
| Room-change errors (test 17:00–19:10) | 10 in 22 hand-counted crossings, all at the open D23 gate |
| Behaviour accuracy (test 23:00–23:30) | **80%** of 0.53 s steps |
| Behaviour macro F1 (8 behaviours present) | 0.48 (0.65 without marking and defecation, 3 test bouts between them) |
| Walking / standing / grooming F1 | 0.90 / 0.74 / 0.77 |
| Drinking / unknown / resting F1 | 0.59 / 0.77 / 0.09; marking and defecation 0 |
| Minutes per behaviour vs hand labels | walking −4%, standing +11%, grooming +12% |
| Behaviour bouts | bout F1 0.56 (same label, overlap at least half the union); start and end both within 2 s: precision 0.30 / recall 0.27 |
| Keeper report 17:00–24:00 | 597 bouts covering all 420 min; Room 2 65% of the evening |
| Detector speed | 2.6–2.8× real time per camera, two cameras at once (RTX 2060 Super, FP16) |

Behaviour on validation (last 20% of each of the five train lists, 4,837 steps; the same model trained without
those parts): accuracy 0.84, macro F1 0.67 over the 7 behaviours present (0.74 without marking); resting 0.88,
walking 0.89, standing 0.71, drinking 0.63, grooming 0.36; the 8-minute daylight part 0.63.

Night resting is the main gap between the two: the test piece's resting is seven short rests (0.9 min), read
mostly as grooming, while the long daylight rests in training are recognised. Marking (one test bout) is read
as walking.

Evaluation protocol: the one day is split by time block, never by frame: 00:00–17:00 for training and
validation, 17:00–24:00 held back. Frames within 10 min of a block edge are not used. Nothing was trained or
tuned on the test block. The behaviour test list was labelled blind. It had been scored once before with an
earlier version of the model, and that result led to more training labels for resting, so the score above is
fair but not fully blind. Detection and pose are scored on validation frames because the 313 test frames are
not labelled yet.

## Architecture

```
4 camera streams ──sources.py──► preprocess.py ──detect.py (YOLO26n-pose)──► track.py (ByteTrack)
  ──zones.py──► handoff.py (which room, or hidden) ──behaviour.py (features → 1D CNN → bouts)
  ──db.py──► SQLite behavior_event + room_log.csv ──behaviour_report.py──► keeper tables + charts
```

- **Detection**: YOLO26n-pose fine-tuned from COCO on our labelled frames (box + nose, eyes, neck,
  mid-back, tail base, tail tip), imgsz 640, grayscale input with CLAHE, colour augmentation off.
  Stock COCO detectors miss the civet in infrared. Only the camera that has the civet runs it at
  7.5 fps; idle cameras run a motion check.
- **Room handoff**: rule-based state machine (visible / pending / hidden) over per-camera zone
  polygons (`config/zones.json`); a room change needs the next room to see the civet on its floor,
  with a 2 s dwell at the wide gate.
- **Behaviour classifier**: 44 features per frame (box shape, keypoints, visibility, head height,
  spine and tail angles, speed, nose at dish or tub, time since seen), causal dilated 1D CNN over a
  5.3 s window, a label every 0.53 s, majority of the last 3, bouts of 2 s or more.
  Sqrt-inverse class weights, AdamW lr 3e-4. Five such nets with softmax averaging, trained on every label
  for 6 epochs (the median best epoch on validation).
- **Detection fallback**: a still civet in daylight can score about 0.42, under the tracker's 0.45 start
  threshold, so its tracks keep dropping and the handoff calls it hidden. While the handoff has it hidden in a
  room, the behaviour model then gets an untracked detection from that room's own cameras (floor or door,
  once boxes were there in 3 of the last 5 steps). Gaps out of sight under 2 s keep the current label.
  Both settings are stored in the weights file; the handoff never uses these detections.
- **No camera sees it → `unknown`**: the behaviour model never labels a civet it cannot see.
- **Causal**: no stage looks at future frames, so the same code runs on live RTSP streams.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision          # the CUDA build for your GPU, from pytorch.org
pip install -r requirements.txt
python -m pytest tests -q              # 24 tests
```

Built and tested on Windows 11, Python 3.11 (3.10–3.12 work), RTX 2060 Super 8 GB. ffmpeg comes
with `imageio-ffmpeg`. Run every command from this folder.

The footage is not included (23 GB). To process video, put the eight recordings in a folder named
`civetwatch_data/` next to this folder (the configs read `../civetwatch_data/`):
`cam01_20260405_24h.mp4` … `cam04_20260405_24h.mp4` (24 h of 5 April 2026) and
`cam01_20260405_0000.mp4` … `cam04_20260405_0000.mp4` (the 75-min sample, 00:00–01:14).
Everything under "Rerun the results" works without it, from the stored caches in `outputs/`.

## Footage facts

- The recorder files are Hikvision MPEG-PS saved as `.mp4`: OpenCV/ffmpeg read them, normal players
  don't (`tools/remux.py` makes playable copies, as in `outputs/remux/`).
- HEVC 1920×1080 at **15 fps real** (the file header says 25/30). The pipeline processes 7.5 fps
  on the active camera and 5 fps on idle ones.
- Times come from `file_start` + position and match the burned-in clock within about 1 s.
  No video on any camera at about 11:06:12–11:07:05 (recorder restart); the gap is kept in the timestamps.
- Cameras are mapped by the burned-in label, not the file name (two exports had CAM2/CAM3 swapped;
  the files here are already renamed to match their labels).
- Rooms are in a row, R1–R2–R3–R4. D12 and D34 are small hatches; D23 is a wide open gate, so CAM03
  sees most of Room 2 through it.

## Label formats

**Behaviour lists** (`labelling/behaviour_*.csv`): one row each time the behaviour changes; a row
starts a bout that lasts until the next row. `START` / `END` rows mark the piece; `note` is free text.
Shortest bout worth a row: 2 s. When two behaviours apply, the first one in this table wins.
The label `hidden` is read as `unknown`.

| Behaviour | What it looks like |
|---|---|
| urination | Pause with a squat (hind legs bent, rump low), tail up, 2–10 s; urine itself is invisible in near-infrared |
| defecation | Squat like urination, rump lower, back arched, tail raised; often the same spot |
| marking | Rubs or drags its rear (perineal gland) along an object, floor, log or wall |
| eating | Head at or in the food dish, or chewing food |
| drinking | Head lowered into a water container, lapping |
| grooming | Licks, bites or scratches its own body, incl. washing the face |
| running | Clearly faster than a walk: bounding, trotting or jumping |
| walking | Moves from place to place at a walk, incl. climbing and sniffing while moving |
| standing | On its legs or sitting up, not moving from its place; standing in a door or gate |
| resting | Lying down or curled up, or still in the sleep box / tyre bed, head down |
| unknown | Not visible on any camera for 2 s or more, or visible but unclear |

**Crossing lists** (`labelling/crossings_*.csv`, columns `time,from,to,note,until`): one row per
room crossing, `time` = when the hind legs are past the opening, made by watching the video without
the room log. `START` / `END` rows give the room at the start and the end of the piece. Notes:
`poke` = head or half body into the opening and back out (not a crossing; `to` = the room it pokes
into, `until` = when it is back out); `linger` + `until` = stood in the opening, then went through at
`until`; `unsure` = could not tell (log changes near it are not counted). A log change matches a
crossing with the same from/to within 10 s; missed = crossings with no match, wrong = log changes with none.
The gate-visit list has no START/END rows and is scored only inside `outputs/gate_visits/train_0113_0500.csv`.

**Pose labels** (`labelling/labels/{train,val}/*.txt`, YOLO-pose): one line per civet,
`0 cx cy w h` then `x y v` for each keypoint in the order nose, eyes, neck, mid_back, tail_base,
tail_tip (normalised; `v` 0 = not visible / outside the frame, 1 = occluded, 2 = visible). An empty file = a labelled frame with no civet.
`labelling/manifest.csv` lists every frame with camera, clock time, split and night/day.
Labels were drawn in CVAT (label set `labelling/cvat_labels.json`), exported as "CVAT for images 1.1"
and converted with `tools/cvat_to_yolo_pose.py <export.zip> --split train` (not CVAT's own YOLO-pose
export, which takes the box from the midline keypoints).

## Rules kept throughout

- The test block 17:00–24:00 was never used to train or choose settings.
- At most two `cache_dets.py --detector yolo` runs at once: four in parallel crashed Windows.
- Everything in the live loop is causal (no future frames), so offline replays equal live behaviour.

## Pipeline commands

```powershell
# 1. Pose model: train on the labelled frames, score on validation
python tools/train_pose.py
python tools/train_pose.py --eval-only models/civet_yolo26n.pt

# 2. Cache detections + keypoints per camera (at most two at a time); 24 h files
python tools/cache_dets.py --config config/april5_24h.yaml --cam CAM01 --detector yolo --out outputs/cache_yolo --resume

# 3. Room tracking on the test block, then score against the hand-counted crossings
python -m civetwatch.run --config config/april5_24h.yaml --offline --cache outputs/cache_yolo --start 61200 --duration 25200 --room-log outputs/test/room_log.csv --set storage.db_path=outputs/test/events.sqlite
python tools/score_crossings.py labelling/crossings_test_1700_1910.csv outputs/test/room_log.csv

# 4. Behaviour: features per labelled piece (with the detection fallback), train 5 nets on every label, score the test piece
python tools/train_behaviour.py extract --cache outputs/cache_yolo --fallback --features outputs/behaviour/feat_fallback
python tools/train_behaviour.py train --features outputs/behaviour/feat_fallback --hold 2 --seeds 5 --all-data --epochs 6 --out models/behaviour_v3.pt
python tools/train_behaviour.py eval --piece test_2300_2330 --features outputs/behaviour/feat_fallback --results outputs/behaviour/test_v3 --out models/behaviour_v3.pt

# 5. Keeper report (hourly activity, time budget, room use, timeline) from the step-3 replay
python tools/behaviour_report.py --db outputs/test/events.sqlite --room-log outputs/test/room_log.csv --start 17:00 --end 23:59:59 --out outputs/report

# 6. Live / real-time replay of the 75-min sample (config/system.yaml)
python -m civetwatch.run
```

### Rerun the results from the included `outputs/` (minutes, no footage or GPU needed)

```powershell
# Room tracking on the test block from the stored detections, then the crossing score
python -m civetwatch.run --config config/april5_24h.yaml --offline --cache outputs/cache_yolo_test --start 61200 --duration 25200 --room-log outputs/rerun/room_log.csv --set storage.db_path=outputs/rerun/rooms.sqlite
python tools/score_crossings.py labelling/crossings_test_1700_1910.csv outputs/rerun/room_log.csv

# Behaviour: test score of the included model, then the validation scores, from the stored features
python tools/train_behaviour.py eval --piece test_2300_2330 --features outputs/behaviour/feat_fallback --results outputs/rerun --out models/behaviour_v3.pt
python tools/train_behaviour.py val --features outputs/behaviour/feat_fallback --out outputs/behaviour/exp/ens5_fallback_hold2.pt
# Retrain: with validation (early stopping; prints the validation scores), then the final fit on every label
python tools/train_behaviour.py train --features outputs/behaviour/feat_fallback --hold 2 --seeds 5 --out outputs/rerun/val_model.pt
python tools/train_behaviour.py train --features outputs/behaviour/feat_fallback --hold 2 --seeds 5 --all-data --epochs 6 --out outputs/rerun/behaviour.pt

# Keeper report: replay 17:00-24:00 with boxes + keypoints, then tables and charts
python -m civetwatch.run --config config/april5_24h.yaml --offline --cache outputs/report_kpts/replay --start 61200 --duration 25200 --room-log outputs/rerun/report_room_log.csv --set storage.db_path=outputs/rerun/events.sqlite
python tools/behaviour_report.py --db outputs/rerun/events.sqlite --room-log outputs/rerun/report_room_log.csv --start 17:00 --end 23:59:59 --out outputs/rerun/report
```

The scored room log (`outputs/yolo_test/room_log.csv`, 10 errors) was made before the `door_track`
rule was added; a rerun with the current settings uses it and can differ slightly.
`outputs/report_kpts/replay/` is the test-block box cache (`outputs/cache_yolo_test`) with keypoints
filled in where a camera saw the civet (windows in `outputs/report_kpts/windows_CAM0N.csv`, cached
with `cache_dets.py --windows`); a full-day `cache_dets.py --detector yolo` run gives the same kind of cache.

Other tools: `tools/mosaic_video.py --start 23:00:00 --end 23:30:00 --out x.mp4` (4-camera video for
labelling), `tools/zone_editor.py --cam CAM03 --video ../civetwatch_data/cam03_20260405_24h.mp4 --at 3600`
(edit zone polygons), `tools/find_gate_visits.py` (gate visits + clips), `tools/extract_frames.py`,
`tools/prelabel_model.py` (frames and pre-labels for CVAT).

Faster step 2: cache only the labelled pieces with
`--windows windows/behaviour_train.csv --pad 0` (and `windows/behaviour_test.csv` into a second folder),
then `extract --cache <that folder> --fallback --features outputs/behaviour/feat_fallback`. Delete an events `.sqlite` before re-running a replay: the store appends.

## Files

| Path | Purpose |
|---|---|
| `civetwatch/` | the pipeline: stream reader, preprocessing, detector, tracker, zones, handoff, behaviour, SQLite |
| `config/system.yaml` | all settings, pointed at the 75-min sample |
| `config/april5_24h.yaml` | the same settings, pointed at the 24 h files |
| `config/zones.json` | floor, door, gate, station and see-through polygons per camera |
| `config/train_pose.yaml` | pose training settings |
| `models/civet_yolo26n.pt` | final pose model (box + 6 keypoints) |
| `models/behaviour_v3.pt` | behaviour model (5 nets, detection fallback, 2 s hold), set in both configs |
| `models/yolo26n-pose.pt`, `models/yolo26s.pt` | COCO starting weights for training / box pre-labelling |
| `labelling/images/`, `labelling/labels/` | frames and YOLO-pose labels (train, val); test and gate frames not labelled yet |
| `labelling/manifest.csv` | every frame: file, camera, time, split, night/day |
| `labelling/prelabels/` | CVAT 1.1 annotation zips used for pre-labelling |
| `labelling/behaviour_*.csv` | behaviour lists, one row per change (5 train pieces, 1 test piece) |
| `labelling/crossings_*.csv` | hand-counted room crossings (test 17:00–19:10, train 00:35–01:14, gate visits) |
| `samples/room_log_20260405*.csv` | reference room logs for the 75-min sample (motion and YOLO detector) |
| `windows/` | clock windows of the labelled behaviour pieces, for a faster keypoint cache |
| `outputs/cache_yolo*/` | stored detections per camera: 75-min sample, train block 01:13–05:00, test block 17:00–24:00 |
| `outputs/behaviour_kpts*/`, `outputs/report_kpts/` | stored detections with keypoints for the behaviour pieces and the 17:00–24:00 report |
| `outputs/behaviour/feat_fallback/` | behaviour features per labelled piece (train pieces and the test piece) and their room logs |
| `outputs/behaviour/test_v3/` | the model on the test piece: events CSV / SQLite, timeline |
| `outputs/behaviour/exp/` | settings tried (`results.csv`, `results2.csv`; 5 seeds each) and `ens5_fallback_hold2.pt`, the model trained without the validation parts (gives the validation scores) |
| `outputs/report_v3/` | keeper report 17:00–24:00: `hourly_activity.csv`, `time_budget.csv`, `room_use.csv`, charts, events |
| `outputs/yolo_test/room_log.csv` | room log of the test block (frozen settings), scored against the crossing list |
| `outputs/crossings/` | 4-camera videos used to write the crossing and behaviour lists; `score_1700_1910.csv` = scored crossings |
| `outputs/gate_visits/` | 66 D23 gate visits (train block) with clips, for the gate crossing list |
| `outputs/remux/` | the 75-min sample remuxed to standard MP4 (plays in normal players) |
| `outputs/train/civet_pose_v1/` | pose training curves, confusion matrix, validation predictions |
| `tests/` | handoff and bout-builder tests |
| `tools/` | caching, training, scoring, labelling helpers, report, zone editor, 4-camera video |
| `../civetwatch_data/` | footage, not included: `cam0N_20260405_24h.mp4` (24 h) and `cam0N_20260405_0000.mp4` (75-min sample) |

## Dataset label distribution

Pose: 1,023 train frames (762 civet, 261 empty), 192 validation (151 civet, 41 empty); 313 test and
181 gate frames picked and pre-labelled, not yet labelled by hand.

Behaviour, train (5 lists, 215 min, 756 bouts; 40 min daylight): Walking 106.8 min · Standing 42.6 · Resting 26.7 ·
Unknown 12.6 · Grooming 11.9 · Drinking 9.2 · Eating 4.2 · Marking 0.5 · Defecation 0.5 · Urination 0.1 · Running 0.1.

Behaviour, test (30 min, 113 bouts): Walking 16.5 min · Standing 7.6 · Grooming 2.6 · Drinking 1.1 ·
Resting 0.9 · Marking 0.7 · Unknown 0.5 · Defecation 0.1.

Marking, urination, defecation and running have one or two training bouts each — their per-class
metrics are noisy and the model rarely predicts them.

## Limitations (state these in the presentation)

1. **One day of footage**: training and test come from 5 April 2026 only; scores may not hold on other days.
2. **Rare behaviours are thin**: one or two bouts of marking, urination, defecation and running; marking is
   read as walking. Short night rests are read as grooming (test F1 0.09), though long daylight rests are
   recognised (validation 0.88).
3. **Little daylight**: one 40-min daylight list, scored on validation only; the test piece is at night.
4. **The D23 gate**: all room-tracking test errors happen there; a box alone cannot tell a civet
   standing in the open gate from one passing through.
5. **Bout edges**: starts and ends are often a few seconds off, so the strict 2 s bout score is low.
6. One labeller, no agreement score; four of the five train lists began as model drafts corrected by hand
   (the test list was labelled blind).
7. Four live streams with the YOLO model have not been timed together yet.
8. **One behaviour test piece**: 30 min at night, already scored once with an earlier model (see Evaluation
   protocol); a fresh blind piece, with daylight, would be the next check.
