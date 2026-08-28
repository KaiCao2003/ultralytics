from ultralytics import YOLO
from pathlib import Path
import csv
import json
import numpy as np


MODEL_PATH = "runs/pose/train/weights/best.pt"

file_list = [
    (
        "data/videos/bright.avi",
        "runs/pose/bright_pose.csv",
        "runs/pose/bright_hd.json",
    ),
    (
        "data/videos/dark.avi",
        "runs/pose/dark_pose.csv",
        "runs/pose/dark_hd.json",
    ),
]

model = YOLO(MODEL_PATH)


for video_path, csv_path, json_path in file_list:
    print(f"Processing: {video_path}")

    results = model.predict(
        source=video_path,
        conf=0.5,
        imgsz=1024,
        # device="mps",
        save=True,
        stream=True,
    )

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    json_frames = []
    json_hd = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "center_x",
            "center_y",
            "front_x",
            "front_y",
            "back_x",
            "back_y",
            "hd_deg",
            "det_conf",
        ])

        for frame_idx, r in enumerate(results):

            # No detection
            if (
                r.keypoints is None
                or len(r.keypoints) == 0
                or r.boxes is None
                or len(r.boxes) == 0
            ):
                writer.writerow([
                    frame_idx,
                    np.nan, np.nan,
                    np.nan, np.nan,
                    np.nan, np.nan,
                    np.nan,
                    np.nan,
                ])
                continue

            # ----------------------------------------
            # If multiple headplates detected,
            # use the one with highest confidence
            # ----------------------------------------

            confs = r.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))

            xy = r.keypoints.xy[best_idx].cpu().numpy()
            det_conf = float(confs[best_idx])

            front_x, front_y = xy[0]
            back_x, back_y = xy[1]

            # Treat missing/invalid keypoints as no HD
            if not np.all(np.isfinite([
                front_x,
                front_y,
                back_x,
                back_y,
            ])):
                writer.writerow([
                    frame_idx,
                    np.nan, np.nan,
                    np.nan, np.nan,
                    np.nan, np.nan,
                    np.nan,
                    det_conf,
                ])
                continue

            center_x = (front_x + back_x) / 2
            center_y = (front_y + back_y) / 2

            # CCW:
            # up = 0°
            # left = 90°
            # down = 180°
            # right = 270°
            dx = front_x - back_x
            dy = front_y - back_y

            hd_deg = np.degrees(
                np.arctan2(-dx, -dy)
            ) % 360

            # CSV: every frame
            writer.writerow([
                frame_idx,
                float(center_x),
                float(center_y),
                float(front_x),
                float(front_y),
                float(back_x),
                float(back_y),
                float(hd_deg),
                det_conf,
            ])

            # JSON: only valid HD frames
            json_frames.append(frame_idx)
            json_hd.append(float(hd_deg))

    # ----------------------------------------
    # Save compact HD JSON
    # ----------------------------------------

    json_data = {
        "hd": {
            "frame": json_frames,
            "hd": json_hd,
        }
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")