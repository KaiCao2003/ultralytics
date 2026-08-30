import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

MODEL_PATH = "runs/pose/headplate_pose_v2/weights/best.pt"
RIGID_BODY = "hp4"
FRONT_MARKER = f"{RIGID_BODY}:front"
BACK_MARKER = f"{RIGID_BODY}:back"
ARROW_LENGTH = 80

file_list = [
    (
        "data/videos/bright.avi",
        "runs/pose/bright_pose.csv",
        "runs/pose/bright_position.csv",
        "runs/pose/bright_hd.json",
        "runs/pose/bright_hd.avi",
    ),
    (
        "data/videos/dark.avi",
        "runs/pose/dark_pose.csv",
        "runs/pose/dark_position.csv",
        "runs/pose/dark_hd.json",
        "runs/pose/dark_hd.avi",
    ),
]

model = YOLO(MODEL_PATH)


for video_path, csv_path, position_path, json_path, output_path in file_list:
    print(f"Processing: {video_path}")

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    results = model.track(
        source=video_path,
        conf=0.25,
        imgsz=1024,
        # device="mps",
        stream=True,
    )

    json_frames = []
    json_hd = []
    last_pose = [None] * 7
    target_id = None
    position_rows = []

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
                "track_id",
                "det_conf",
            ]
        )
        position_writer.writerows(
            [
                [
                    "Format Version",
                    "1.25",
                    "Take Name",
                    Path(video_path).stem,
                    "Take Notes",
                    "YOLO26 pose tracking",
                    "Capture Frame Rate",
                    fps,
                    "Export Frame Rate",
                    fps,
                    "Capture Start Time",
                    "",
                    "Capture Start Frame",
                    0,
                    "Total Frames in Take",
                    total_frames,
                    "Total Exported Frames",
                    total_frames,
                    "Rotation Type",
                    "XYZ",
                    "Length Units",
                    "Pixels",
                    "Coordinate Space",
                    "Image",
                ],
                [],
                ["", "Type", *["Rigid Body"] * 6, *["Marker"] * 6],
                ["", "Name", *[RIGID_BODY] * 6, *[FRONT_MARKER] * 3, *[BACK_MARKER] * 3],
                [
                    "",
                    "ID",
                    *[f"YOLO26:{RIGID_BODY}"] * 6,
                    *[f"YOLO26:{FRONT_MARKER}"] * 3,
                    *[f"YOLO26:{BACK_MARKER}"] * 3,
                ],
                ["", "Parent", *[""] * 12],
                ["", "", *["Rotation"] * 3, *["Position"] * 3, *["Position"] * 6],
                ["Frame", "Time (Seconds)", *["X", "Y", "Z"] * 4],
            ]
        )

        for frame_idx, r in enumerate(results):
            det_conf = np.nan

            if len(r.boxes) and r.boxes.id is not None:
                confs = r.boxes.conf.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().numpy()

                if target_id is None:
                    best_idx = int(np.argmax(confs))
                elif target_id in track_ids:
                    best_idx = int(np.flatnonzero(track_ids == target_id)[0])
                else:
                    best_idx = None

                if best_idx is not None:
                    target_id = int(track_ids[best_idx])
                    front, back = r.keypoints.xy[best_idx].cpu().numpy()
                    center = (front + back) / 2
                    hd_deg = np.degrees(np.arctan2(back[0] - front[0], back[1] - front[1])) % 360
                    last_pose = [float(value) for value in (*center, *front, *back, hd_deg)]
                    det_conf = float(confs[best_idx])

            writer.writerow([frame_idx, *last_pose, target_id, det_conf])
            position_rows.append(
                [
                    frame_idx,
                    frame_idx / fps,
                    0.0,
                    0.0,
                    last_pose[-1],
                    *last_pose[:2],
                    0.0,
                    *last_pose[2:4],
                    0.0,
                    *last_pose[4:6],
                    0.0,
                ]
            )
            json_frames.append(frame_idx)
            json_hd.append(last_pose[-1])

            frame = r.orig_img.copy()
            if last_pose[-1] is not None:
                center_x, center_y = last_pose[:2]
                theta = np.radians(last_pose[-1])
                start = (round(center_x), round(center_y))
                end = (
                    round(center_x - ARROW_LENGTH * np.sin(theta)),
                    round(center_y - ARROW_LENGTH * np.cos(theta)),
                )
                cv2.arrowedLine(frame, start, end, (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.25)
                cv2.circle(frame, start, 5, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"{last_pose[-1]:.1f} deg",
                    (end[0] + 10, end[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            video_writer.write(frame)

        first_pose = next(row[2:] for row in position_rows if row[4] is not None)
        for row in position_rows:
            if row[4] is None:
                row[2:] = first_pose
        position_writer.writerows(position_rows)

    json_hd = [first_pose[2] if hd is None else hd for hd in json_hd]

    video_writer.release()

    # ----------------------------------------
    # Save compact HD JSON
    # ----------------------------------------

    json_data = {
        RIGID_BODY: {
            "frames": json_frames,
            "hd": json_hd,
        }
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved CSV:  {position_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved AVI:  {output_path}")
