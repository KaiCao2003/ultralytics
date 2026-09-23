"""Select suspicious and apparently OK video frames for another Label Studio round."""

import argparse
import bisect
import json
import math
import random
from collections import deque
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd

from local_cli import ROOT


def _sample_frames(rows, amount, gap, rng):
    """Balance bad reasons and separate selected frames within this review pool."""
    pools = {}
    for row in rows:
        pools.setdefault(row["reason"].split(";")[0], []).append(int(row["frame"]))
    for reason, frames in pools.items():
        rng.shuffle(frames)
        pools[reason] = deque(frames)
    selected = []
    while len(selected) < amount and any(pools.values()):
        for frames in pools.values():
            while frames:
                frame = frames.popleft()
                index = bisect.bisect_left(selected, frame)
                if (index and frame - selected[index - 1] < gap) or (
                    index < len(selected) and selected[index] - frame < gap
                ):
                    continue
                selected.insert(index, frame)
                break
            if len(selected) >= amount:
                break
    return selected


def prepare_review(
    video_path,
    csv_path,
    output_dir,
    local_files_root,
    bad_n,
    ok_n,
    min_gap_seconds,
    confidence_threshold,
    jump_threshold,
):
    """Export heuristic bad/OK review sets from raw detections without running inference."""
    video_path, csv_path = Path(video_path).resolve(), Path(csv_path).resolve()
    output_dir, local_files_root = Path(output_dir).resolve(), Path(local_files_root).resolve()
    relative_frames = (output_dir / "frames").relative_to(local_files_root).as_posix()
    if bad_n < 0 or ok_n < 0 or min_gap_seconds < 0:
        raise ValueError("Sample counts and minimum gap must be nonnegative")
    if not 0 <= confidence_threshold <= 1 or not 0 <= jump_threshold <= 180:
        raise ValueError("Confidence must be in [0, 1] and heading jump in [0, 180] degrees")
    if csv_path == output_dir / "review.csv":
        raise ValueError("The source CSV must differ from the generated review.csv")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has no valid frame rate: {video_path}")
        rows = pd.read_csv(csv_path)
        # Keep every detection in the raw CSV; only review uses one representative per frame.
        review = rows.sort_values("det_conf", ascending=False, kind="stable").drop_duplicates("frame")
        review = review.sort_values("frame").reset_index(drop=True)
        if "detection_count" not in review:
            # Legacy tracking CSVs can contain held coordinates with missing detection confidence.
            review["detection_count"] = review["det_conf"].notna().astype(int)
        if "time_s" not in review:
            review["time_s"] = review["frame"] / fps
        detected = (review["detection_count"] > 0) & np.isfinite(review["det_conf"])
        points = review[["front_x", "front_y", "back_x", "back_y"]]
        valid_points = np.isfinite(points).all(axis=1)
        valid_points &= points[["front_x", "back_x"]].ge(0).all(axis=1)
        valid_points &= points[["front_x", "back_x"]].le(width).all(axis=1)
        valid_points &= points[["front_y", "back_y"]].ge(0).all(axis=1)
        valid_points &= points[["front_y", "back_y"]].le(height).all(axis=1)
        valid_points &= (points.front_x != points.back_x) | (points.front_y != points.back_y)
        low_confidence = review["det_conf"] < confidence_threshold
        for column in ("front_conf", "back_conf"):
            if column in review:
                # Two-coordinate pose models do not produce keypoint confidence.
                low_confidence |= review[column].notna() & (review[column] < confidence_threshold)
        valid_heading = detected & valid_points & np.isfinite(review["hd_deg"])
        adjacent = (review["frame"].diff() == 1) & valid_heading & valid_heading.shift(1, fill_value=False)
        review["hd_jump_deg"] = ((review["hd_deg"].diff() + 180) % 360 - 180).abs().where(adjacent)
        review["reason"] = ""
        for reason, mask in (
            ("no_detection", ~detected),
            ("multiple_detections", review["detection_count"] > 1),
            ("invalid_keypoints", detected & ~valid_points),
            ("low_confidence", detected & low_confidence),
            ("heading_jump", review["hd_jump_deg"] > jump_threshold),
        ):
            review.loc[mask, "reason"] += reason + ";"
        review["reason"] = review["reason"].str.rstrip(";")
        review["group"] = np.where(review["reason"] == "", "ok", "bad")
        review.loc[review["group"] == "ok", "reason"] = "ok"
        rng = random.Random(42)
        gap = math.ceil(min_gap_seconds * fps)
        selected = []
        for group, amount in (("bad", bad_n), ("ok", ok_n)):
            candidates = review.loc[review["group"] == group, ["frame", "reason"]].to_dict("records")
            selected.extend(_sample_frames(candidates, amount, gap, rng))
        review["selected"] = review["frame"].isin(selected)
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for index, row in review[review["selected"]].iterrows():
            frame_index = int(row["frame"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
            image_name = f"{video_path.stem}_{frame_index:08d}.png"
            image_path = frame_dir / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Cannot write image: {image_path}")
            prediction = []
            if detected.iloc[index] and valid_points.iloc[index]:
                for label in ("front", "back"):
                    prediction.append(
                        {
                            "from_name": "keypoints",
                            "to_name": "image",
                            "type": "keypointlabels",
                            "original_width": width,
                            "original_height": height,
                            "image_rotation": 0,
                            "value": {
                                "x": float(row[f"{label}_x"]) / width * 100,
                                "y": float(row[f"{label}_y"]) / height * 100,
                                "width": 0.5,
                                "keypointlabels": [label],
                            },
                        }
                    )
            tasks.append(
                {
                    "data": {"image": "/data/local-files/?d=" + quote(f"{relative_frames}/{image_name}", safe="/")},
                    "predictions": [
                        {
                            "model_version": csv_path.stem,
                            "score": float(row["det_conf"]) if detected.iloc[index] else 0.0,
                            "result": prediction,
                        }
                    ],
                    "meta": {
                        "source_video": str(video_path),
                        "frame": frame_index,
                        "time_s": float(row["time_s"]),
                        "group": row["group"],
                        "reason": row["reason"],
                    },
                }
            )
        columns = ["frame", "time_s", "detection_count", "det_conf", "hd_jump_deg", "group", "reason", "selected"]
        review[columns].to_csv(output_dir / "review.csv", index=False)
        for name, group in (("import", None), ("bad", "bad"), ("ok", "ok")):
            selected_tasks = tasks if group is None else [task for task in tasks if task["meta"]["group"] == group]
            (output_dir / f"label_studio_{name}.json").write_text(
                json.dumps(selected_tasks, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        (output_dir / "label_config.xml").write_text(
            '<View>\n  <Image name="image" value="$image"/>\n'
            '  <KeyPointLabels name="keypoints" toName="image">\n'
            '    <Label value="front" background="#ef4444"/>\n'
            '    <Label value="back" background="#3b82f6"/>\n'
            "  </KeyPointLabels>\n</View>\n",
            encoding="utf-8",
        )
        counts = review[review["selected"]]["group"].value_counts()
        print(f"Review: {counts.get('bad', 0)} suspicious, {counts.get('ok', 0)} apparently OK frames: {output_dir}")
        print("OK is a confidence/continuity heuristic; confirm every selected frame when labeling.")
        return output_dir / "label_studio_import.json"
    finally:
        capture.release()


def main():
    """Reselect review frames from an existing detection CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-root", type=Path, default=ROOT)
    parser.add_argument("--bad", type=int, default=100)
    parser.add_argument("--ok", type=int, default=100)
    parser.add_argument("--min-gap-seconds", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--jump-threshold", type=float, default=45.0)
    args = parser.parse_args()
    prepare_review(
        args.video,
        args.csv,
        args.output,
        args.local_files_root,
        args.bad,
        args.ok,
        args.min_gap_seconds,
        args.confidence_threshold,
        args.jump_threshold,
    )


if __name__ == "__main__":
    main()
