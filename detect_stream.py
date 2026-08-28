from ultralytics import YOLO
from pathlib import Path
import csv
import numpy as np


MODEL_PATH = "runs/pose/train/weights/best.pt"

file_list = [
    ("data/videos/bright.avi", "runs/pose/bright_pose.csv"),
    ("data/videos/dark.avi",   "runs/pose/dark_pose.csv"),
]

model = YOLO(MODEL_PATH)

for video_path, csv_path in file_list:
    print(f"Processing: {video_path}")

    results = model.predict(
        source=video_path,
        conf=0.5,
        imgsz=1024,
        device="mps",
        save=True,
        stream=True,
    )

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

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

            # One headplate per frame
            xy = r.keypoints.xy[0].cpu().numpy()

            front_x, front_y = xy[0]
            back_x, back_y = xy[1]

            center_x = (front_x + back_x) / 2
            center_y = (front_y + back_y) / 2

            # Image convention:
            # up = 0°, right = 90°, down = 180°, left = 270°
            dx = front_x - back_x
            dy = front_y - back_y
            hd_deg = np.degrees(np.arctan2(dx, -dy)) % 360

            det_conf = float(r.boxes.conf[0].cpu())

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

    print(f"Saved: {csv_path}")