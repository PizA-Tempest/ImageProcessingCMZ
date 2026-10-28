from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import torch

def main():
    manifest = Path('data/frames/frames.csv')
    outfile = Path('data/civet_frames.csv')

    if not manifest.exists():
        print('No frames found.')
        return

    frames = pd.read_csv(manifest).fillna('')
    print(f'Total frames: {len(frames)}')

    # Load YOLO model (pretrained on COCO - detects animals, people, etc.)
    print('Loading YOLO model...')
    model = YOLO('yolo11n.pt')  # small, fast model

    # Classes that could be civet or related
    # COCO animal classes: bird(14), cat(15), dog(16), horse(17), sheep(18),
    # cow(19), elephant(20), bear(21), zebra(22), giraffe(23)
    # Also consider any detection as potential civet
    ANIMAL_CLASSES = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

    results = []
    batch_size = 16

    print('Running YOLO detection on all frames...')
    for i in range(0, len(frames), batch_size):
        batch = frames.iloc[i:i+batch_size]
        paths = batch['image_path'].tolist()

        # Run detection
        detections = model(paths, verbose=False, conf=0.3)

        for j, det in enumerate(detections):
            row = batch.iloc[j]
            has_detection = len(det.boxes) > 0
            has_animal = False
            max_conf = 0

            if has_detection:
                for box in det.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    max_conf = max(max_conf, conf)
                    if cls_id in ANIMAL_CLASSES:
                        has_animal = True

            results.append({
                'image_path': row['image_path'],
                'source_video': row['source_video'],
                'camera': row['camera'],
                'timestamp_seconds': row['timestamp_seconds'],
                'has_detection': has_detection,
                'has_animal': has_animal,
                'max_confidence': round(max_conf, 3),
                'num_detections': len(det.boxes)
            })

        if (i // batch_size) % 10 == 0:
            print(f'  Processed {i+len(paths)}/{len(frames)} frames')

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(outfile, index=False, encoding='utf-8-sig')

    # Summary
    print('\n=== RESULTS ===')
    print(f'Total frames: {len(df_results)}')
    print(f'Frames with any detection: {df_results["has_detection"].sum()}')
    print(f'Frames with animal detection: {df_results["has_animal"].sum()}')

    # Save filtered frames with animals
    animal_frames = df_results[df_results['has_animal']].copy()
    if len(animal_frames) > 0:
        animal_frames['has_civet'] = ''  # empty for user to confirm
        animal_frames.to_csv('data/animal_candidates.csv', index=False, encoding='utf-8-sig')
        print(f'\nSaved {len(animal_frames)} animal candidate frames to data/animal_candidates.csv')
        print('Open find_civet.py to review and confirm.')

        # Show breakdown by camera
        print('\nBy camera:')
        print(animal_frames['camera'].value_counts().to_string())
    else:
        print('\nNo animal detections found. May need lower confidence threshold.')

if __name__ == '__main__':
    main()
