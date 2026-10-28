from pathlib import Path
import argparse
import csv
import cv2
from ultralytics import YOLO

ANIMAL_CLASSES = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
               'toothbrush']


def scan_video(model, video_path, out_dir, interval_s, conf_thresh, cam_label):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  Cannot open {video_path.name}')
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    step = max(1, int(round(fps * interval_s)))
    results = []
    frame_idx = 0
    saved = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / fps
        r = model.predict(frame, conf=conf_thresh, verbose=False, imgsz=640)[0]
        if len(r.boxes) > 0:
            best = r.boxes[int(r.boxes.conf.argmax())]
            cls_id = int(best.cls[0])
            conf = float(best.conf[0])
            is_animal = cls_id in ANIMAL_CLASSES
            ts_ms = int(ts * 1000)
            fname = f'{video_path.stem}_{ts_ms:010d}ms.jpg'
            out_path = out_dir / fname
            if not out_path.exists():
                cv2.imwrite(str(out_path), frame)
            results.append({
                'image_path': f'data/scan_candidates/{video_path.stem}/{fname}',
                'source_video': video_path.name,
                'camera': cam_label,
                'timestamp_seconds': round(ts, 1),
                'top_class': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                'top_conf': round(conf, 3),
                'is_animal_class': is_animal,
                'num_detections': len(r.boxes),
            })
            saved += 1
        frame_idx += step
        if frame_idx // step % 50 == 0:
            print(f'    {video_path.name}: {int(ts)}s / {int(duration)}s ({saved} hits so far)')

    cap.release()
    print(f'  {video_path.name}: done, {saved} detection frames of {int(duration/interval_s)} sampled')
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=float, default=10.0, help='seconds between samples')
    ap.add_argument('--conf', type=float, default=0.10, help='YOLO confidence threshold')
    args = ap.parse_args()

    videos_dir = Path('data/videos')
    out_root = Path('data/scan_candidates')
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(v for v in videos_dir.glob('*.mp4'))
    if not videos:
        print('No videos found in data/videos')
        return

    print(f'Loading YOLO (conf={args.conf}, interval={args.interval}s)...')
    model = YOLO('yolo11n.pt')

    all_rows = []
    for vp in videos:
        cam = vp.stem.split('_')[0]
        cam_dir = out_root / vp.stem
        cam_dir.mkdir(parents=True, exist_ok=True)
        all_rows.extend(scan_video(model, vp, cam_dir, args.interval, args.conf, cam))

    out_csv = Path('data/scan_candidates.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['image_path', 'source_video', 'camera',
                                          'timestamp_seconds', 'top_class', 'top_conf',
                                          'is_animal_class', 'num_detections'])
        w.writeheader()
        w.writerows(all_rows)

    print(f'\n=== TOTAL: {len(all_rows)} detection frames -> {out_csv} ===')
    animal_rows = [r for r in all_rows if r['is_animal_class']]
    print(f'Animal-class hits: {len(animal_rows)}')
    by_cam = {}
    for r in all_rows:
        by_cam[r['camera']] = by_cam.get(r['camera'], 0) + 1
    print('By camera (all detections):', by_cam)


if __name__ == '__main__':
    main()
