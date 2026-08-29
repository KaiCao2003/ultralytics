import csv
import json
from pathlib import Path

import numpy as np

from ultralytics import YOLO

MODEL_PATH = "runs/pose/headplate_pose_v2/weights/best.pt"

file_list = [
    (
        "data/videos/bright.avi",
        "runs/pose/bright_pose.csv",
        "runs/pose/bright_position.csv",
        "runs/pose/bright_hd.json",
    ),
    (
        "data/videos/dark.avi",
        "runs/pose/dark_pose.csv",
        "runs/pose/dark_position.csv",
        "runs/pose/dark_hd.json",
    ),
]

model = YOLO(MODEL_PATH)


for video_path, csv_path, position_path, json_path in file_list:
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

    json_frames = []
    json_hd = []
    last_pose = [None] * 7

    with open(csv_path, "w", newline="") as f, open(position_path, "w", newline="") as position_file:
        writer = csv.writer(f)
        position_writer = csv.writer(position_file)

        writer.writerow(
            [
                "frame",
                "center_x",
                "center_y",
                "front_x",
                "front_y",
                "back_x",
                "back_y",
                "hd_deg",
                "det_conf",
            ]
        )
        position_writer.writerow(["frame", "center_x", "center_y", "det_conf"])

        for frame_idx, r in enumerate(results):
            det_conf = np.nan

            if len(r.boxes):
                confs = r.boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                front, back = r.keypoints.xy[best_idx].cpu().numpy()
                center = (front + back) / 2
                hd_deg = np.degrees(np.arctan2(back[0] - front[0], back[1] - front[1])) % 360
                last_pose = [float(value) for value in (*center, *front, *back, hd_deg)]
                det_conf = float(confs[best_idx])

            writer.writerow([frame_idx, *last_pose, det_conf])
            position_writer.writerow([frame_idx, *last_pose[:2], det_conf])
            json_frames.append(frame_idx)
            json_hd.append(last_pose[-1])

    # ----------------------------------------
    # Save compact HD JSON
    # ----------------------------------------

    json_data = {
        "hp4": {
            "frames": json_frames,
            "hd": json_hd,
        }
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved CSV:  {position_path}")
    print(f"Saved JSON: {json_path}")
