"""Extract neighbor frames around rare-behavior events to enrich the review queue."""
from pathlib import Path
import cv2
import pandas as pd

RARE = {'Eating', 'Drinking', 'Standing', 'Grooming', 'Other'}
OFFSETS = [-30, -15, 15, 30]          # seconds around each rare frame
OFFSET_RARE_SHORT = [-15, 15]          # for Standing/Grooming/Other (fewer, noisier)

q = pd.read_csv('data/review_queue.csv').fillna('')
ann = pd.read_csv('data/annotations.csv').fillna('') if Path('data/annotations.csv').exists() else None
if ann is None:
    raise SystemExit('annotations.csv required to know rare labels')
label_by_path = dict(zip(ann['image_path'], ann['label']))

known = set(q['image_path'])
durations = {}
for vp in Path('data/videos').glob('*.mp4'):
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    durations[vp.name] = n / fps if fps else 0
    cap.release()

# group existing queue rows by video to find rare frames
q['label'] = q['image_path'].map(label_by_path).fillna('')
new_rows = []
seen = set()

for _, r in q.iterrows():
    lab = r['label']
    if lab not in RARE:
        continue
    offs = OFFSET_RARE_SHORT if lab in {'Standing', 'Grooming', 'Other'} else OFFSETS
    dur = durations.get(r['source_video'], 0)
    for off in offs:
        t = r['timestamp_seconds'] + off
        if t < 1 or t > dur - 1:
            continue
        cam_ds = r['source_video'].replace('.mp4', '')
        rel = f"data/scan_candidates/{cam_ds}/{cam_ds}_{int(round(t * 1000)):010d}ms.jpg"
        if rel in known or rel in seen:
            continue
        seen.add(rel)
        new_rows.append({
            'image_path': rel, 'source_video': r['source_video'], 'camera': r['camera'],
            'timestamp_seconds': float(t), 'top_class': '', 'top_conf': 0.0,
            'is_animal_class': True, 'num_detections': 0, 'from_label': lab,
        })

new = pd.DataFrame(new_rows)
print('candidate neighbor frames:', len(new))

# extract from videos
missing = 0
per_video = {}
for _, r in new.sort_values(['source_video', 'timestamp_seconds']).iterrows():
    out_path = Path(r['image_path'])
    if out_path.exists():
        continue
    vcap = cv2.VideoCapture(str(Path('data/videos') / r['source_video']))
    fps = vcap.get(cv2.CAP_PROP_FPS) or 1
    idx = int(round(r['timestamp_seconds'] * fps))
    vcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = vcap.read()
    vcap.release()
    if not ok:
        missing += 1
        continue
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    per_video[r['source_video']] = per_video.get(r['source_video'], 0) + 1

print('extracted:', sum(per_video.values()), '| seek failures:', missing)
print('per video:', per_video)

# keep only frames that exist on disk; append to review queue with unique row ids
new = new[new['image_path'].map(lambda p: Path(p).exists())].drop(columns=['from_label'])
q2 = pd.concat([q.drop(columns=['label'], errors='ignore'), new], ignore_index=True)
q2['row_id'] = range(len(q2))
q2.to_csv('data/review_queue.csv', index=False)
print('review_queue now:', len(q2), 'frames')
