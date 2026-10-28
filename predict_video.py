import argparse, time
from pathlib import Path
import cv2, pandas as pd, torch
from PIL import Image
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO

ANIMAL_CLASSES = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
ROOMS = {'CAM1': 'Room 1', 'CAM2': 'Room 2', 'CAM3': 'Room 3', 'CAM4': 'Room 4'}


def load_behavior_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    classes = ckpt['classes']
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, classes, bool(ckpt.get('crop_to_bbox', False))


def crop_bbox(frame, xyxy, pad=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
    l, t, r, b = max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)
    crop = frame[t:b, l:r]
    return crop if crop.size > 0 else frame


def process_video(cap_path, camera, room, every_s, device, yolo, clf, classes, tf, do_crop=False, conf_gate=0.10):
    cap = cv2.VideoCapture(str(cap_path))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {cap_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    step = max(1, int(fps * every_s))
    rows, idx = [], 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        sec = idx / fps
        det = yolo.predict(frame, conf=conf_gate, classes=list(ANIMAL_CLASSES),
                           imgsz=1280, verbose=False, device=device)[0]
        if len(det.boxes):
            best = det.boxes[int(det.boxes.conf.argmax())]
            yolo_conf = float(best.conf[0])
            region = crop_bbox(frame, best.xyxy[0].cpu().numpy()) if do_crop else frame
            rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            x = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(clf(x), 1)[0].cpu().numpy()
            j = int(probs.argmax())
            rows.append({'camera': camera, 'room': room, 'source_video': Path(cap_path).name,
                         'timestamp_seconds': round(sec, 2), 'behavior': classes[j],
                         'confidence': round(float(probs[j]), 4),
                         'camera_status': 'Active', 'civet_detected': True,
                         'detection_confidence': round(yolo_conf, 3), 'frame_read_ok': True})
        else:
            rows.append({'camera': camera, 'room': room, 'source_video': Path(cap_path).name,
                         'timestamp_seconds': round(sec, 2), 'behavior': 'Not Visible',
                         'confidence': 0.0, 'camera_status': 'Idle',
                         'civet_detected': False, 'detection_confidence': 0.0,
                         'frame_read_ok': True})
        idx += step
    cap.release()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', help='single video path')
    p.add_argument('--all', action='store_true', help='process every video in data/videos')
    p.add_argument('--camera', default='CAM1')
    p.add_argument('--room', default=None)
    p.add_argument('--every-seconds', type=float, default=5)
    p.add_argument('--conf', type=float, default=0.10, help='YOLO civet confidence gate (matches extract_bboxes.py)')
    p.add_argument('--output', default='results/predictions.csv')
    args = p.parse_args()

    ckpt_path = Path('models/behavior_classifier.pt')
    if not ckpt_path.exists():
        raise SystemExit('Model missing. Run train.py first.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    clf, classes, do_crop = load_behavior_model(ckpt_path, device)
    print(f'Crop to detection box: {do_crop}')
    yolo = YOLO('yolov8n.pt')
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    if args.all:
        jobs = []
        for vp in sorted(Path('data/videos').glob('*.mp4')):
            cam = vp.stem.split('_')[0]
            jobs.append((vp, cam, ROOMS.get(cam, cam)))
    elif args.video:
        jobs = [(Path(args.video), args.camera, args.room or ROOMS.get(args.camera, 'Room 1'))]
    else:
        raise SystemExit('Provide --video PATH or --all')

    all_rows, start = [], time.perf_counter()
    for vp, cam, room in jobs:
        print(f'Processing {vp.name} ({cam})...')
        rows = process_video(vp, cam, room, args.every_seconds, device, yolo, clf, classes, tf, do_crop, args.conf)
        n_active = sum(r['civet_detected'] for r in rows)
        print(f'  {len(rows)} frames sampled, civet in {n_active} ({100*n_active/max(1,len(rows)):.1f}%)')
        all_rows.extend(rows)
    elapsed = time.perf_counter() - start

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(all_rows)
    if out.exists():
        old = pd.read_csv(out)
        done = set(new['source_video'].unique())
        old = old[~old.source_video.isin(done)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f'Wrote {len(all_rows)} new predictions to {out}. Time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
