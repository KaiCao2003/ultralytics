"""Record per-frame pose predictions, then select bad and OK candidates for another labeling round."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from local_cli import ROOT
from ultralytics import YOLO
from ultralytics.utils.files import increment_path

MODEL_PATH = ROOT / "runs/pose/headplate_260921/headplate_260921/weights/best.pt"
VIDEO_PATH = Path("/mnt/senzailab/Kai/#Recording/m20/260918/260918_11/260918.avi")
FIELDS = [
    "frame",
    "time_s",
    "detection_index",
    "detection_count",
    "class_id",
    "det_conf",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "front_x",
    "front_y",
    "back_x",
    "back_y",
    "front_conf",
    "back_conf",
    "center_x",
    "center_y",
    "hd_deg",
]


def main():
    """Save every decoded frame's detections without tracking, filling, or smoothing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=VIDEO_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/pose/predict_raw", help="Output directory; increment if it exists."
    )
    parser.add_argument("--local-files-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else None)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--conf", type=float, default=0.05, help="Inference threshold; retain low-confidence detections."
    )
    parser.add_argument("--bad", type=int, default=100)
    parser.add_argument("--ok", type=int, default=100)
    parser.add_argument("--min-gap-seconds", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--jump-threshold", type=float, default=45.0)
    parser.add_argument("--raw-only", action="store_true", help="Save predictions now and run ls_predict.py later.")
    args = parser.parse_args()
    source, model_path = args.source.expanduser().resolve(), args.model.expanduser().resolve()
    if not source.is_file() or not model_path.is_file():
        parser.error(f"Source and model must exist: {source}, {model_path}")
    if not 0 <= args.conf <= 1 or not 0 <= args.confidence_threshold <= 1:
        parser.error("Confidence thresholds must be between 0 and 1")
    if min(args.bad, args.ok, args.min_gap_seconds, args.jump_threshold) < 0 or args.imgsz <= 0:
        parser.error("Sample counts, gap and jump threshold must be nonnegative; imgsz must be positive")
    if args.jump_threshold > 180:
        parser.error("Heading jump threshold must not exceed 180 degrees")

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    opened = cap.isOpened()
    cap.release()
    if not opened or not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        raise RuntimeError(f"Cannot read video metadata: {source}")
    model = YOLO(str(model_path))
    if model.task != "pose" or model.model.kpt_shape[0] != 2:
        raise ValueError("Use a two-keypoint pose model with keypoint order front, back")

    local_root = args.local_files_root.expanduser().resolve()
    output = increment_path(args.output.expanduser().resolve())
    if not args.raw_only:
        output.relative_to(local_root)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "raw.csv"
    metadata = {
        "source_video": str(source),
        "model": str(model_path),
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": total_frames,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "keypoint_order": ["front", "back"],
        "heading_convention": "Image pixels: up=0, left=90, down=180, right=270 degrees",
    }
    metadata_path = output / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Raw predictions: {csv_path}", flush=True)
    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        vid_stride=1,
        save=False,
        verbose=False,
    )
    frame_count = detection_count = 0
    next_progress = 10
    try:
        # Line buffering keeps completed rows readable if inference is interrupted.
        with csv_path.open("x", newline="", buffering=1) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for frame_index, result in enumerate(results):
                boxes = result.boxes.cpu().numpy()
                points = result.keypoints.cpu().numpy()
                common = {"frame": frame_index, "time_s": frame_index / fps, "detection_count": len(boxes)}
                if not len(boxes):
                    writer.writerow({**common, "detection_index": -1})
                for index, (box, keypoints) in enumerate(zip(boxes.data, points.data)):
                    front, back = keypoints[:, :2]
                    center = (front + back) / 2
                    # Derived values never replace the original predicted coordinates.
                    heading = np.degrees(np.arctan2(back[0] - front[0], back[1] - front[1])) % 360
                    row = {
                        **common,
                        "detection_index": index,
                        "class_id": int(box[5]),
                        "det_conf": float(box[4]),
                        **dict(zip(FIELDS[6:10], map(float, box[:4]))),
                        **dict(zip(FIELDS[10:14], map(float, keypoints[:, :2].ravel()))),
                        "center_x": float(center[0]),
                        "center_y": float(center[1]),
                        "hd_deg": float(heading),
                    }
                    if keypoints.shape[1] == 3:
                        row.update(front_conf=float(keypoints[0, 2]), back_conf=float(keypoints[1, 2]))
                    writer.writerow(row)
                frame_count += 1
                detection_count += len(boxes)
                if total_frames > 0:
                    progress = min(90, frame_count * 10 // total_frames * 10)
                    if progress >= next_progress:
                        print(f"{progress}%: {frame_count}/{total_frames} frames", flush=True)
                        next_progress = progress + 10
    finally:
        results.close()
        if model.predictor.dataset is not None and model.predictor.dataset.cap is not None:
            model.predictor.dataset.cap.release()
    if not frame_count:
        raise RuntimeError(f"No decodable frames: {source}")
    metadata.update(frames_processed=frame_count, detections=detection_count)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"100%: Saved {frame_count} frames and {detection_count} detections to {csv_path}")
    if not args.raw_only:
        from ls_predict import prepare_review

        prepare_review(
            source,
            csv_path,
            output,
            local_root,
            args.bad,
            args.ok,
            args.min_gap_seconds,
            args.confidence_threshold,
            args.jump_threshold,
        )


if __name__ == "__main__":
    main()
