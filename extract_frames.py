import argparse, csv
from pathlib import Path
import cv2

EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}

def main():
    p = argparse.ArgumentParser(description='Sample frames from selected video clips.')
    p.add_argument('--input', default='data/videos')
    p.add_argument('--output', default='data/frames')
    p.add_argument('--every-seconds', type=float, default=10)
    p.add_argument('--max-frames-per-video', type=int, default=500)
    args = p.parse_args()
    src, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(x for x in src.rglob('*') if x.suffix.lower() in EXTS)
    if not videos:
        raise SystemExit(f'No videos found in {src.resolve()}')
    manifest = out / 'frames.csv'
    rows = []
    for vi, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f'[WARN] Cannot open: {video}'); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps > 0 else 0
        step = max(1, int((fps if fps > 0 else 1) * args.every_seconds))
        clip_dir = out / video.stem
        clip_dir.mkdir(parents=True, exist_ok=True)
        print(f'[{vi}/{len(videos)}] {video.name} duration~{duration/60:.1f} min')
        idx, saved = 0, 0
        while saved < args.max_frames_per_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok: break
            sec = idx / fps if fps > 0 else idx
            name = f'{video.stem}_{int(sec*1000):010d}ms.jpg'
            target = clip_dir / name
            if cv2.imwrite(str(target), frame):
                # Infer camera from leading CAMxx or Axx token, otherwise use video stem.
                prefix = video.stem.split('_')[0]
                if prefix.lower().startswith('cam'):
                    cam = prefix
                elif prefix.upper().startswith('A') and prefix[1:].isdigit():
                    cam = 'CAM' + prefix[1:].zfill(2)
                else:
                    cam = 'Unknown'
                rows.append({'image_path': str(target.as_posix()), 'source_video': video.name,
                             'camera': cam, 'timestamp_seconds': round(sec, 3), 'label': ''})
                saved += 1
            idx += step
        cap.release()
        print(f'  saved {saved} frames')
    with manifest.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['image_path','source_video','camera','timestamp_seconds','label'])
        w.writeheader(); w.writerows(rows)
    print(f'Wrote {manifest} with {len(rows)} frames. Open the annotation app to label them.')

if __name__ == '__main__': main()
