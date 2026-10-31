# Civet Behavior Detection

AI pipeline for the large-spotted civet (ชะมดแผงสันหางดำ) project: 4 cameras × 3 videos each
(~55 hours of HEVC footage). The system detects the civet with YOLO, classifies its behavior with
a fine-tuned EfficientNet-B0, and presents results in Streamlit dashboards.

## Results (final, 4-fold group cross-validation)

| Metric | Value |
|---|---|
| Labeled frames | 711 (592 behavior-labeled + 119 "No Civet"; 587 used for training) |
| CV accuracy (unseen videos) | **69.0% ± 5.1** |
| CV macro F1 | 0.248 ± 0.065 (pooled out-of-fold: **0.303**) |
| Walking precision / recall | 0.78 / 0.85 (pooled) |
| Presence gate accuracy | **97.5%** (friend's civet_yolo26n; old COCO gate: 83.3%) |
| Predictions generated | 39,932 frames (every 5s across all 12 videos) |
| Civet present | 15.5% of sampled frames (CAM4_Dataset1 26% → CAM1_Dataset2 1.4%; CAM4_Dataset2 0%) |
| Inference speed | ~114 ms/frame (gate 640 + crop detector 1280, RTX 5060 Ti) |

Evaluation protocol: every video is used as test set exactly once (group CV by source video);
the classifier never trains on frames from a video it is tested on. Reported metrics are pooled
out-of-fold predictions over all 587 training frames. These metrics are on the human-corrected
labels (6 frames were relabeled to harder classes; the earlier 73% score was on pre-correction labels).

## Architecture

```
videos ──scan_all_videos.py──► YOLO candidates ──annotate_app.py──► annotations.csv
                                                    (agent-assisted labels + human review)
annotations + bboxes ──train.py──► behavior_classifier.pt + results/metrics.json (group CV)
videos ──predict_video.py --all──► results/predictions.csv ──dashboard.py──► monitoring UI
```

- **Detection**: presence gate = `civet-monitor/models/civet_yolo26n.pt`, a fine-tuned binary civet
  detector (conf 0.25, imgsz 640) with **97.5% presence accuracy** on our labeled frames, versus 83.3%
  for the old COCO "any animal class" gate. Crop boxes still come from YOLOv8n COCO — the same
  distribution the classifier was trained on (friend's tight crops scored 5pp worse in CV).
- **Classifier**: EfficientNet-B0 (ImageNet pretrained), trained on the padded YOLO box crop
  (falls back to full frame when no box was found). 2-phase training (head 3 epochs, then
  unfreeze backbone), cosine LR decay after unfreezing (+4pp accuracy over constant LR),
  sqrt-inverse class weights (capped at 3), label smoothing 0.1,
  augmentation (flip/rotate/affine/jitter), best epoch by validation macro F1.
- **No detection → "Not Visible" / Idle**: the classifier only runs when the gate fires, so
  empty-enclosure frames are never classified.

## Pipeline commands

```powershell
# 1. Sample candidate frames (every 10s, conf 0.10) and deduplicate >=30s apart
python scan_all_videos.py --interval 10 --conf 0.10

# 2. Label frames (Streamlit UI: camera/dataset picker, keys 1-9, gallery browser)
streamlit run annotate_app.py          # http://localhost:8511

# 2b. Enrich rare behaviors (Eating/Drinking/...) with +/-15/30s neighbor frames
python extract_neighbors.py
python extract_bboxes.py               # YOLO boxes for classifier crops

# 3. Train with 4-fold group CV; saves model + results/
python train.py --epochs 60

# 4. Predict all videos (every 5s) -> results/predictions.csv
python predict_video.py --all --every-seconds 5

# 5. Monitoring dashboard (overview / behaviors / timeline / model performance)
streamlit run dashboard.py
```

## Files

| Path | Purpose |
|---|---|
| `data/videos/` | 12 source videos (CAM1–4 × Dataset1–3) |
| `data/review_queue.csv` | 711 candidate frames (image_path, video, camera, timestamp) |
| `data/annotations.csv` | ground-truth labels (written by annotate_app / labeling agents) |
| `data/bboxes.csv` | YOLO boxes per labeled frame (classifier crops) |
| `models/behavior_classifier.pt` | final model (trained on all videos; epoch from internal val split) |
| `results/metrics.json` | CV metrics, per-fold table, pooled per-class report |
| `results/confusion_matrix.png` | pooled out-of-fold confusion matrix |
| `results/predictions.csv` | 39,932 frame-level predictions |
| `train.py` | training + group CV evaluation |
| `predict_video.py` | YOLO gate + classifier over videos |
| `annotate_app.py` | labeling UI |
| `dashboard.py` | results dashboard |
| `eval_gate.py` | presence-gate A/B evaluation on the labeled frames |
| `civet-monitor/` | friend's full pipeline (local, git-ignored); only its detector is used |

## Dataset label distribution (711 frames)

Walking 421 · No Civet 119 · Eating 89 · Drinking 47 · Standing 16 · Grooming 7 · Other 7 ·
Running 2 · Resting 3.

Classes with <4 frames are excluded from training (Running 2, Resting 3 — Resting had CV F1 0 and
cost ~2 points of macro F1 while it was included). Grooming (7) and Other (7–8) remain thin.

## Limitations (state these in the presentation)

1. **Still images cannot perfectly separate Walking vs Standing** — motion cues need video
   (temporal models: LSTM/transformer over frame sequences are the natural next step).
2. **Rare classes are thin**: Grooming (7), Other (7), Standing (16), Drinking (47) — collect more
   examples by labeling dense clips around known events (`extract_neighbors.py`); Resting (3) and
   Running (2) are too thin to train at all.
3. **Cross-video generalization is the honest metric**: accuracy on an unseen video is 69%,
   lower than a random frame split would suggest (which would leak between adjacent frames).
   After the human label corrections this is below the pre-correction score — corrections
   moved frames into harder, visually similar classes.
   lower than a random frame split would suggest (which would leak between adjacent frames).
4. YOLO rarely classifies the civet correctly (mostly "bird"); the gate accepts any COCO
   animal-class detection, which may rare-fire on non-civet animals outdoors.
5. Predictions are frame-level estimates (5s sampling), not verified ethogram events.
6. The friend's civet-monitor system claims 80% behaviour accuracy, but it scores 0.53 s *temporal
   steps* from 5.3 s pose windows on one day (macro F1 0.48) — not directly comparable to our
   per-frame video-split score. Its detector is integrated; integrating its temporal behaviour model
   means running pose + tracking over full videos with an adapted zone config (next step).

## Deadline-ready checklist

- [x] All 12 videos scanned; 425 + 286 frames labeled (agent-assisted, human-audited sample)
- [x] Group-CV evaluation with confusion matrix + per-class report
- [x] Friend's civet detector adopted as presence gate (97.5% vs 83.3% COCO)
- [x] Full-corpus predictions (39,932 frames, re-run with the new gate)
- [x] Labeling UI + monitoring dashboard (dark theme, keys 1–9, gallery)
- [ ] Presentation slides (use `results/metrics.json`, `results/confusion_matrix.png`, dashboard)
