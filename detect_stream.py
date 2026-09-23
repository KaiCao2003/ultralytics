import argparse
import csv
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np
import torch

from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "runs/pose/headplate_260921/headplate_260921/weights/best.pt"

RIGID_BODY = "hp4"
FRONT_MARKER = f"{RIGID_BODY}:front"
BACK_MARKER = f"{RIGID_BODY}:back"
ARROW_LENGTH = 80

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=Path)
parser.add_argument("date")
parser.add_argument("session_id", type=int)
parser.add_argument("--model", type=Path, default=MODEL_PATH)
args = parser.parse_args()

date = args.date
session_dir = args.base_dir / date / f"{date}_{args.session_id}"
video_path = session_dir / f"{date}.avi"
data_dir = session_dir / "data"
remote_paths = [
    session_dir / f"{date}.csv",
    data_dir / f"{date}.csv",
    data_dir / "processed/head_direction.json",
    data_dir / f"{date}_hd.avi",
]

model = YOLO(args.model)


print(f"Processing: {video_path}")

# Keep local outputs available if processing or transfer fails.
local_dir = Path(mkdtemp(prefix="detect-stream-"))
csv_path = local_dir / "pose.csv"
position_path = local_dir / "position.csv"
json_path = local_dir / "head_direction.json"
output_path = local_dir / "head_direction.avi"
print(f"Local outputs: {local_dir}")
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
    raise RuntimeError(f"Could not read valid video metadata: {video_path}")

video_writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (width, height),
)
if not video_writer.isOpened():
    raise RuntimeError(f"Could not open video writer: {output_path}")

json_frames = []
json_hd = []
last_pose = [None] * 7
target_id = None
first_pose = None
next_progress = 10

with ExitStack() as cleanup:
    cleanup.callback(video_writer.release)
    model.add_callback("on_predict_start", lambda predictor: cleanup.callback(predictor.dataset.cap.release))
    results = model.track(
        source=str(video_path),
        conf=0.25,
        imgsz=1024,
        device="mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else None,
        stream=True,
        verbose=False,
    )
    cleanup.callback(results.close)
    f = cleanup.enter_context(open(csv_path, "w", newline="", buffering=1))
    position_file = cleanup.enter_context(open(position_path, "w", newline="", buffering=1))
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
                "",
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

            best_idx = int(np.argmax(confs))
            if target_id is not None and target_id in track_ids:
                best_idx = int(np.flatnonzero(track_ids == target_id)[0])

            target_id = int(track_ids[best_idx])
            front, back = r.keypoints.xy[best_idx].cpu().numpy()
            center = (front + back) / 2
            hd_deg = np.degrees(np.arctan2(back[0] - front[0], back[1] - front[1])) % 360
            last_pose = [float(value) for value in (*center, *front, *back, hd_deg)]
            det_conf = float(confs[best_idx])

        writer.writerow([frame_idx, *last_pose, target_id, det_conf])
        if last_pose[-1] is not None:
            row = [
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
            if first_pose is None:
                first_pose = row[2:]
                position_writer.writerows([i, i / fps, *first_pose] for i in range(frame_idx))
            position_writer.writerow(row)
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
        if total_frames > 0:
            progress = min(90, (frame_idx + 1) * 10 // total_frames * 10)
            if progress >= next_progress:
                print(f"{progress}%: {frame_idx + 1}/{total_frames} frames", flush=True)
                next_progress = progress + 10


if first_pose is None:
    raise RuntimeError(f"No tracked poses found in {video_path}")

# An unfinished local CSV has a blank exported count; finalize it only at EOF.
completed_position = local_dir / "position.complete.csv"
with open(position_path, newline="") as source, open(completed_position, "w", newline="") as destination:
    metadata = next(csv.reader(source))
    metadata[metadata.index("Total Exported Frames") + 1] = len(json_frames)
    csv.writer(destination).writerow(metadata)
    shutil.copyfileobj(source, destination)
completed_position.replace(position_path)

json_hd = [first_pose[2] if hd is None else hd for hd in json_hd]

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

for local_path, remote_path in zip((csv_path, position_path, json_path, output_path), remote_paths, strict=True):
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy beside the destination so a failed transfer cannot truncate its prior output.
    temporary_path = remote_path.with_name(f".{remote_path.name}.{local_dir.name}.tmp")
    shutil.copy2(local_path, temporary_path)
    temporary_path.replace(remote_path)
shutil.rmtree(local_dir)
print(f"100%: Saved outputs to: {data_dir}")
