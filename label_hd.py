import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


file_list = [
    (
        "data/videos/bright.avi",
        "runs/pose/bright_pose_ccw.csv",
        "runs/pose/bright_hd_ccw.avi",
    ),
    (
        "data/videos/dark.avi",
        "runs/pose/dark_pose_ccw.csv",
        "runs/pose/dark_hd_ccw.avi",
    ),
]

ARROW_LENGTH = 80


for video_path, csv_path, output_path in file_list:
    print(f"Processing: {video_path}")

    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    for frame_idx in tqdm(
        range(total_frames),
        desc=Path(video_path).name,
        unit="frame"
    ):
        ok, frame = cap.read()

        if not ok:
            break

        if frame_idx < len(df):
            row = df.iloc[frame_idx]

            if not np.isnan(row["hd_deg"]):
                cx = float(row["center_x"])
                cy = float(row["center_y"])
                hd = float(row["hd_deg"])

                theta = np.radians(hd)

                # CCW convention:
                # up = 0°, left = 90°, down = 180°, right = 270°
                dx = -ARROW_LENGTH * np.sin(theta)
                dy = -ARROW_LENGTH * np.cos(theta)

                start = (
                    int(round(cx)),
                    int(round(cy)),
                )

                end = (
                    int(round(cx + dx)),
                    int(round(cy + dy)),
                )

                cv2.arrowedLine(
                    frame,
                    start,
                    end,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )

                cv2.circle(
                    frame,
                    start,
                    5,
                    (0, 255, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"{hd:.1f} deg",
                    (end[0] + 10, end[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Saved: {output_path}")