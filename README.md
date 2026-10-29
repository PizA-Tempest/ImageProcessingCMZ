# Civet Behavior Detection

AI pipeline for the large-spotted civet (ชะมดแผงสันหางดำ) project: 4 cameras × 3 videos each
(~55 hours of HEVC footage). The system detects the civet with YOLO, classifies its behavior with
a fine-tuned EfficientNet-B0, and presents results in Streamlit dashboards.

## Results (final, 4-fold group cross-validation)

| Metric | Value |
|---|---|
| Labeled frames | 711 (592 behavior-labeled + 119 "No Civet"; 588 used for training) |
| CV accuracy (unseen videos) | **73.4% ± 3.2** |
| CV macro F1 | 0.280 ± 0.090 (pooled out-of-fold: **0.378**) |
| Walking precision / recall | 0.79 / 0.87 (pooled) |
| Predictions generated | 39,932 frames (every 5s across all 12 videos) |
| Civet present | 13.9% of sampled frames (CAM2 26% → CAM4 5%) |
| Inference speed | ~33 ms/frame (RTX 5060 Ti: YOLO 1280 + classifier crop) |

Evaluation protocol: every video is used as test set exactly once (group CV by source video);
the classifier never trains on frames from a video it is tested on. Reported metrics are pooled
out-of-fold predictions over all 588 training frames.

## Architecture

```
videos ──scan_all_videos.py──► YOLO candidates ──annotate_app.py──► annotations.csv
                                                    (agent-assisted labels + human review)
annotations + bboxes ──train.py──► behavior_classifier.pt + results/metrics.json (group CV)
videos ──predict_video.py --all──► results/predictions.csv ──dashboard.py──► monitoring UI
```

- **Detection**: YOLOv8n, COCO animal classes, conf ≥ 0.10 at imgsz 1280 (the civet is often
  misclassified as "bird"/"cat" — any animal-class detection is treated as a civet presence gate).
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

## Dataset label distribution (711 frames)

Walking 421 · No Civet 119 · Eating 89 · Drinking 47 · Standing 17 · Grooming 7 · Other 7 ·
Running 2 · Resting 2.

Classes with <3 frames (Running, Resting) are excluded from training; "No Civet" is handled by
the detection gate. Grooming/Other remain thin (7 examples each) — their per-class metrics are noisy.

## Limitations (state these in the presentation)

1. **Still images cannot perfectly separate Walking vs Standing** — motion cues need video
   (temporal models: LSTM/transformer over frame sequences are the natural next step).
2. **Rare classes are thin**: Grooming (7), Other (7), Standing (17) — collect more examples by
   labeling dense clips around known events (`extract_neighbors.py`).
3. **Cross-video generalization is the honest metric**: accuracy on an unseen video is 73%,
   lower than a random frame split would suggest (which would leak between adjacent frames).
4. YOLO rarely classifies the civet correctly (mostly "bird"); the gate accepts any COCO
   animal-class detection, which may rare-fire on non-civet animals outdoors.
5. Predictions are frame-level estimates (5s sampling), not verified ethogram events.

## Deadline-ready checklist

- [x] All 12 videos scanned; 425 + 286 frames labeled (agent-assisted, human-audited sample)
- [x] Group-CV evaluation with confusion matrix + per-class report
- [x] Full-corpus predictions (39,932 frames)
- [x] Labeling UI + monitoring dashboard (dark theme, keys 1–9, gallery)
- [ ] Presentation slides (use `results/metrics.json`, `results/confusion_matrix.png`, dashboard)
