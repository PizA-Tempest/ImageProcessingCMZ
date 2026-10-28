"""Extract YOLO bounding boxes for the labeled review frames (used for cropping)."""
from pathlib import Path
import pandas as pd
import cv2
from ultralytics import YOLO

ANIMAL_CLASSES = list(range(14, 24))

q = pd.read_csv('data/review_queue.csv')
model = YOLO('yolov8n.pt')
out = []
for p in q['image_path'].tolist():
    img = cv2.imread(p)
    if img is None:
        continue
    res = model.predict(img, conf=0.05, classes=ANIMAL_CLASSES, imgsz=1280, verbose=False, device=0)
    r = res[0]
    best = None
    if len(r.boxes):
        confs = r.boxes.conf.cpu().numpy()
        i = int(confs.argmax())
        if confs[i] >= 0.10:  # a wrong box is worse than no box; ignore noise below 0.10
            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
            best = (int(x1), int(y1), int(x2), int(y2), float(confs[i]))
    out.append({'image_path': p, 'bbox': None if best is None else f'{best[0]},{best[1]},{best[2]},{best[3]}',
                'bbox_conf': None if best is None else best[4]})

df = pd.DataFrame(out)
Path('data').mkdir(exist_ok=True)
df.to_csv('data/bboxes.csv', index=False)
print('boxes found:', df.bbox.notna().sum(), '/', len(df))
