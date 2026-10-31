import pandas as pd, numpy as np
from pathlib import Path
from ultralytics import YOLO

df = pd.read_csv('data/annotations.csv').fillna('')
df = df[df.image_path.map(lambda p: Path(p).exists())].reset_index(drop=True)
truth = (df.label != 'No Civet').astype(int).values
paths = df.image_path.tolist()

def score(pred, name):
    pred = np.array(pred).astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum()); tn = int(((pred == 0) & (truth == 0)).sum())
    acc = (tp + tn) / len(truth); prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f'{name:38s} acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} (tp={tp} fp={fp} fn={fn})')
    return acc

print(f'frames={len(truth)}  civet present={truth.sum()}  empty={len(truth)-truth.sum()}')
score(df.is_animal_class.astype(int), 'current gate (COCO animal classes)')

m = YOLO('civet-monitor/models/civet_yolo26n.pt')
for conf in [0.10, 0.25, 0.40]:
    for imgsz in [640, 1280]:
        preds = []
        for i in range(0, len(paths), 64):
            r = m.predict(paths[i:i+64], conf=conf, imgsz=imgsz, device=0, verbose=False)
            preds += [1 if len(x.boxes) else 0 for x in r]
        score(preds, f'civet_yolo26n conf={conf} imgsz={imgsz}')
