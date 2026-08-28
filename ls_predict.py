import cv2
import pandas as pd
import numpy as np
import json
from pathlib import Path


ROOT = Path("/Users/vxf1610/Developer/yolov26")

FILES = [
    (
        "bright",
        ROOT / "data/videos/bright.avi",
        ROOT / "runs/pose/bright_pose.csv",
    ),
    (
        "dark",
        ROOT / "data/videos/dark.avi",
        ROOT / "runs/pose/dark_pose.csv",
    ),
]


# Must be inside LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
OUT_DIR = ROOT / "data/labelstudio_round2"
FRAME_DIR = OUT_DIR / "frames"
JSON_PATH = OUT_DIR / "label_studio_import.json"

# Relative to:
# /Users/vxf1610/Developer/yolov26/data
LS_PREFIX = "/data/local-files/?d=labelstudio_round2/frames/"


# How many new frames per video
LOW_CONF_N = 100
HD_JUMP_N = 100
RANDOM_N = 100

RANDOM_SEED = 42


FRAME_DIR.mkdir(parents=True, exist_ok=True)


def circular_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


tasks = []


for name, video_path, csv_path in FILES:

    print(f"Processing {name}")

    df = pd.read_csv(csv_path)

    # --------------------------------------------
    # HD jump
    # --------------------------------------------

    df["hd_jump"] = np.nan

    valid = (
        df["hd_deg"].notna()
        & df["hd_deg"].shift(1).notna()
    )

    df.loc[valid, "hd_jump"] = [
        circular_diff(a, b)
        for a, b in zip(
            df.loc[valid, "hd_deg"],
            df["hd_deg"].shift(1)[valid],
        )
    ]

    # --------------------------------------------
    # Select frames
    # --------------------------------------------

    selected = set()

    # Lowest confidence, including failed detections
    low = (
        df.sort_values("det_conf", na_position="first")
        .head(LOW_CONF_N)
    )

    selected.update(low["frame"].astype(int))

    # Biggest HD jumps
    jumps = (
        df.dropna(subset=["hd_jump"])
        .sort_values("hd_jump", ascending=False)
        .head(HD_JUMP_N)
    )

    selected.update(jumps["frame"].astype(int))

    # Random coverage
    remaining = df[
        ~df["frame"].astype(int).isin(selected)
    ]

    random_rows = remaining.sample(
        n=min(RANDOM_N, len(remaining)),
        random_state=RANDOM_SEED,
    )

    selected.update(random_rows["frame"].astype(int))

    selected = sorted(selected)

    print(f"Selected {len(selected)} frames")

    # --------------------------------------------
    # Video
    # --------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = {
        int(row.frame): row
        for row in df.itertuples()
    }

    for frame_idx in selected:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ok, frame = cap.read()

        if not ok:
            print(f"Failed frame: {frame_idx}")
            continue

        row = rows[frame_idx]

        image_name = f"{name}_{frame_idx:06d}.jpg"

        cv2.imwrite(
            str(FRAME_DIR / image_name),
            frame,
        )

        prediction_results = []

        # ----------------------------------------
        # Add predicted front/back if available
        # ----------------------------------------

        if (
            np.isfinite(row.front_x)
            and np.isfinite(row.front_y)
            and np.isfinite(row.back_x)
            and np.isfinite(row.back_y)
        ):

            front_x = row.front_x / width * 100
            front_y = row.front_y / height * 100

            back_x = row.back_x / width * 100
            back_y = row.back_y / height * 100

            prediction_results = [
                {
                    "from_name": "keypoints",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": front_x,
                        "y": front_y,
                        "width": 0.5,
                        "keypointlabels": ["front"],
                    },
                },

                {
                    "from_name": "keypoints",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": back_x,
                        "y": back_y,
                        "width": 0.5,
                        "keypointlabels": ["back"],
                    },
                },
            ]

        score = (
            float(row.det_conf)
            if np.isfinite(row.det_conf)
            else 0.0
        )

        tasks.append(
            {
                "data": {
                    "image": LS_PREFIX + image_name
                },

                "predictions": [
                    {
                        "model_version": "yolo26-round1",
                        "score": score,
                        "result": prediction_results,
                    }
                ],

                "meta": {
                    "source_video": name,
                    "frame": frame_idx,
                },
            }
        )

    cap.release()


with open(JSON_PATH, "w") as f:
    json.dump(tasks, f, indent=2)


print()
print(f"Frames: {FRAME_DIR}")
print(f"JSON:   {JSON_PATH}")
print(f"Tasks:  {len(tasks)}")