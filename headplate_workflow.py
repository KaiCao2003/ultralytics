# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Filesystem-backed active-learning workflow for headplate pose estimation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np
from filelock import FileLock

VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
WORKSPACE_NAME = "headplate-yolo"
STATE_NAME = "state.json"
BOX_SCALE = 2.5
LABEL_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <KeyPointLabels name="keypoints" toName="image">
    <KeyPointLabel value="front" background="#ef4444"/>
    <KeyPointLabel value="back" background="#3b82f6"/>
  </KeyPointLabels>
</View>
"""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def workflow_root() -> Path:
    """Return the configured data root used by the web app."""
    return Path(os.environ.get("YOLO_WORKFLOW_ROOT", "/mnt/senzailab")).expanduser().resolve()


def resolve_under(root: Path, value: str | Path, *, create: bool = False) -> Path:
    """Resolve a path and require it to remain under root."""
    root = root.expanduser().resolve()
    path = Path(value).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path must stay under {root}: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise FileNotFoundError(path)
    return path


def workspace(project: Path) -> Path:
    """Return the generated workflow directory for a project."""
    return project / WORKSPACE_NAME


def round_dir(project: Path, round_number: int) -> Path:
    """Return one immutable round directory."""
    return workspace(project) / f"round_{round_number:02d}"


def _atomic_json(path: Path, data: Any) -> None:
    """Atomically write JSON in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_state(project: Path) -> dict[str, Any]:
    """Load a project's workflow state."""
    path = workspace(project) / STATE_NAME
    if not path.exists():
        raise FileNotFoundError(f"Workflow is not initialized: {path}")
    with FileLock(f"{path}.lock"):
        return json.loads(path.read_text(encoding="utf-8"))


def update_state(project: Path, **changes: Any) -> dict[str, Any]:
    """Merge changes into a project's workflow state."""
    path = workspace(project) / STATE_NAME
    with FileLock(f"{path}.lock"):
        state = json.loads(path.read_text(encoding="utf-8"))
        state.update(changes)
        state["updated_at"] = utc_now()
        _atomic_json(path, state)
    return state


def record_round(project: Path, round_number: int, **changes: Any) -> dict[str, Any]:
    """Merge changes into one round's state record."""
    state = load_state(project)
    rounds = state.setdefault("rounds", {})
    record = rounds.setdefault(str(round_number), {})
    record.update(changes)
    return update_state(project, rounds=rounds)


def default_config() -> dict[str, Any]:
    """Return the minimal configuration exposed by the web app."""
    return {
        "round1_frames": 100,
        "review_frames": 100,
        "max_rounds": 2,
        "base_model": "yolo26n-pose.pt",
        "epochs": 2000,
        "imgsz": 1024,
        "batch": -1,
        "device": "0",
        "conf": 0.5,
        "val_fraction": 0.2,
        "json_key": "hp4",
        "seed": 42,
    }


def discover_videos(project: Path) -> list[Path]:
    """Return supported videos placed directly in a project directory."""
    return sorted(path for path in project.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def discover_annotation_exports(project: Path) -> list[Path]:
    """Return Label Studio JSON exports placed directly in a project directory."""
    return sorted(path for path in project.glob("*.json") if path.is_file())


def configure_project(project: Path, videos: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    """Initialize a workflow project without moving its source files."""
    project = resolve_under(workflow_root(), project)
    if not videos:
        raise ValueError("Select at least one video")
    relative_videos = []
    for video in videos:
        video = resolve_under(project, video)
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video: {video}")
        relative_videos.append(video.relative_to(project).as_posix())

    settings = default_config()
    unknown = set(config) - set(settings)
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    settings.update(config)
    try:
        for name in ("round1_frames", "review_frames", "max_rounds", "epochs", "imgsz", "batch", "seed"):
            settings[name] = int(settings[name])
        for name in ("conf", "val_fraction"):
            settings[name] = float(settings[name])
    except (TypeError, ValueError) as error:
        raise ValueError("Numeric settings contain an invalid value") from error
    for name in ("base_model", "device", "json_key"):
        settings[name] = str(settings[name]).strip()
    if settings["round1_frames"] < 1 or settings["review_frames"] < 1:
        raise ValueError("Frame counts must be positive")
    if settings["max_rounds"] != 2:
        raise ValueError("This workflow requires exactly two Label Studio rounds")
    if settings["epochs"] < 1 or settings["imgsz"] < 32 or (settings["batch"] < 1 and settings["batch"] != -1):
        raise ValueError("Training settings are invalid")
    if not settings["base_model"] or not settings["device"]:
        raise ValueError("Base model and device are required")
    if not 0 < settings["conf"] <= 1 or not 0 < settings["val_fraction"] < 1:
        raise ValueError("Confidence and validation fraction must be between 0 and 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", settings["json_key"]):
        raise ValueError("JSON key may contain only letters, numbers, underscores, and hyphens")

    path = workspace(project) / STATE_NAME
    if path.exists():
        raise FileExistsError(f"Project is already initialized: {path}")
    workspace(project).mkdir(parents=True)
    state = {
        "version": 1,
        "project": str(project),
        "videos": relative_videos,
        "config": settings,
        "stage": "CONFIGURED",
        "current_round": 0,
        "progress": 0.0,
        "message": "Ready to prepare Round 1",
        "error": None,
        "active_job": None,
        "rounds": {},
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    _atomic_json(path, state)
    return state


def _video_metadata(video: Path) -> dict[str, Any]:
    """Read the metadata needed by sampling and progress reporting."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    metadata = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if metadata["frames"] < 1 or metadata["fps"] <= 0 or metadata["width"] < 1 or metadata["height"] < 1:
        raise RuntimeError(f"Invalid video metadata: {video}")
    return metadata


def _allocate_samples(frame_counts: list[int], total: int) -> list[int]:
    """Allocate a total sample count proportionally across videos."""
    total = max(total, len(frame_counts))
    available = sum(frame_counts)
    total = min(total, available)
    raw = [total * count / available for count in frame_counts]
    allocation = [min(count, max(1, math.floor(value))) for count, value in zip(frame_counts, raw)]
    while sum(allocation) < total:
        candidates = [
            (raw[i] - math.floor(raw[i]), frame_counts[i] - allocation[i], i)
            for i in range(len(frame_counts))
            if allocation[i] < frame_counts[i]
        ]
        if not candidates:
            break
        allocation[max(candidates)[2]] += 1
    while sum(allocation) > total:
        candidates = [(allocation[i] - raw[i], allocation[i], i) for i in range(len(allocation)) if allocation[i] > 1]
        allocation[max(candidates)[2]] -= 1
    return allocation


def _sample_indices(frame_count: int, amount: int) -> list[int]:
    """Return evenly spaced, unique frame indices."""
    if amount >= frame_count:
        return list(range(frame_count))
    return sorted(set(np.linspace(0, frame_count - 1, amount, dtype=int).tolist()))


def _slug(value: str) -> str:
    """Return a filesystem-safe identifier."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "video"


def _label_studio_url(path: Path) -> str:
    """Build a Label Studio local-files URL relative to the configured root."""
    relative = path.resolve().relative_to(workflow_root()).as_posix()
    return f"/data/local-files/?d={quote(relative, safe='/')}"


def _reset_generated(path: Path, project: Path) -> None:
    """Reset one generated directory while refusing paths outside the workflow workspace."""
    generated_root = workspace(project).resolve()
    path = path.resolve()
    if path == generated_root or generated_root not in path.parents:
        raise ValueError(f"Refusing to reset unsafe path: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def prepare_round1(project: Path) -> Path:
    """Extract the first review set and create Label Studio import files."""
    project = resolve_under(workflow_root(), project)
    state = load_state(project)
    videos = [project / value for value in state["videos"]]
    metadata = [_video_metadata(video) for video in videos]
    allocation = _allocate_samples([item["frames"] for item in metadata], int(state["config"]["round1_frames"]))
    label_dir = round_dir(project, 1) / "label_studio"
    frame_dir = label_dir / "frames"
    _reset_generated(label_dir, project)
    frame_dir.mkdir()

    tasks = []
    manifest = []
    completed = 0
    total = sum(allocation)
    for video_index, (video, meta, amount) in enumerate(zip(videos, metadata, allocation), 1):
        capture = cv2.VideoCapture(str(video))
        for frame_index in _sample_indices(meta["frames"], amount):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f"Cannot read frame {frame_index} from {video}")
            image_name = f"v{video_index:02d}_{_slug(video.stem)}__frame_{frame_index:08d}.jpg"
            image_path = frame_dir / image_name
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                capture.release()
                raise RuntimeError(f"Cannot write image: {image_path}")
            manifest.append(
                {
                    "image_name": image_name,
                    "image_path": image_path.relative_to(project).as_posix(),
                    "video": video.relative_to(project).as_posix(),
                    "frame": frame_index,
                    "timestamp_sec": f"{frame_index / meta['fps']:.6f}",
                    "selection_reason": "uniform",
                }
            )
            tasks.append(
                {
                    "data": {"image": _label_studio_url(image_path)},
                    "meta": {"video": video.name, "frame": frame_index, "round": 1},
                }
            )
            completed += 1
            update_state(project, progress=completed / total, message=f"Extracting Round 1 frames: {completed}/{total}")
        capture.release()

    _write_manifest(label_dir / "frame_manifest.csv", manifest)
    _atomic_json(label_dir / "label_studio_import.json", tasks)
    (label_dir / "label_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")
    record_round(
        project,
        1,
        label_studio_dir=str(label_dir.relative_to(project)),
        label_studio_import=str((label_dir / "label_studio_import.json").relative_to(project)),
        manifest=str((label_dir / "frame_manifest.csv").relative_to(project)),
        sample_count=len(manifest),
    )
    update_state(
        project,
        stage="WAITING_ROUND_01_ANNOTATIONS",
        current_round=1,
        progress=1.0,
        message="Round 1 files are ready for Label Studio",
        active_job=None,
    )
    return label_dir


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a frame manifest."""
    fieldnames = ["image_name", "image_path", "video", "frame", "timestamp_sec", "selection_reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Index a frame manifest by image name."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image_name"]: row for row in csv.DictReader(handle)}


def _task_image_name(task: dict[str, Any]) -> str:
    """Extract an image filename from a Label Studio task."""
    image = str(task.get("data", {}).get("image", ""))
    query = parse_qs(urlparse(image).query)
    candidate = query.get("d", [urlparse(image).path])[-1]
    return Path(unquote(candidate)).name


def _annotation_keypoints(task: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return normalized front/back keypoints from the latest valid annotation."""
    annotations = [item for item in task.get("annotations", []) if not item.get("was_cancelled", False)]
    if not annotations:
        return {}
    keypoints = {}
    for result in annotations[-1].get("result", []):
        if result.get("type") != "keypointlabels":
            continue
        labels = result.get("value", {}).get("keypointlabels", [])
        if not labels or labels[0] not in {"front", "back"}:
            continue
        x = float(result["value"]["x"]) / 100.0
        y = float(result["value"]["y"]) / 100.0
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError(f"Keypoint is outside the image: {(x, y)}")
        keypoints[labels[0]] = (x, y)
    return keypoints


def _pose_label(front: tuple[float, float], back: tuple[float, float]) -> str:
    """Create one YOLO Pose label from front/back keypoints."""
    front_x, front_y = front
    back_x, back_y = back
    distance = math.hypot(front_x - back_x, front_y - back_y)
    if distance <= 0:
        raise ValueError("Front and back keypoints must be different")
    center_x = (front_x + back_x) / 2
    center_y = (front_y + back_y) / 2
    box_size = distance * BOX_SCALE
    x1, y1 = max(0.0, center_x - box_size / 2), max(0.0, center_y - box_size / 2)
    x2, y2 = min(1.0, center_x + box_size / 2), min(1.0, center_y + box_size / 2)
    values = [0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, front_x, front_y, back_x, back_y]
    return " ".join(str(value) if index == 0 else f"{value:.8f}" for index, value in enumerate(values)) + "\n"


def _load_annotations(
    annotation_path: Path, manifest: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load matching, complete Label Studio annotations."""
    tasks = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise TypeError("Label Studio export must be a JSON list")
    samples = {}
    unmatched = incomplete = 0
    for task in tasks:
        image_name = _task_image_name(task)
        if image_name not in manifest:
            unmatched += 1
            continue
        keypoints = _annotation_keypoints(task)
        if "front" not in keypoints or "back" not in keypoints:
            incomplete += 1
            continue
        samples[image_name] = {
            "image_name": image_name,
            "manifest": manifest[image_name],
            "front": keypoints["front"],
            "back": keypoints["back"],
        }
    if len(samples) < 2:
        raise ValueError(f"Only {len(samples)} complete matching annotations were found")
    return list(samples.values()), {"valid": len(samples), "incomplete": incomplete, "unmatched": unmatched}


def build_dataset(project: Path, round_number: int, annotation_path: Path) -> tuple[Path, dict[str, int]]:
    """Build an immutable dataset version from one Label Studio export."""
    project = resolve_under(workflow_root(), project)
    annotation_path = resolve_under(project, annotation_path)
    manifest_path = round_dir(project, round_number) / "label_studio" / "frame_manifest.csv"
    samples, summary = _load_annotations(annotation_path, _read_manifest(manifest_path))
    dataset = round_dir(project, round_number) / "dataset"
    staging = dataset.with_name("dataset.tmp")
    _reset_generated(staging, project)

    if round_number > 1:
        previous = round_dir(project, round_number - 1) / "dataset"
        if not previous.exists():
            raise FileNotFoundError(f"Previous dataset is missing: {previous}")
        shutil.rmtree(staging)
        shutil.copytree(previous, staging)
        splits = {sample["image_name"]: "train" for sample in samples}
    else:
        rng = random.Random(int(load_state(project)["config"]["seed"]))
        names = [sample["image_name"] for sample in samples]
        rng.shuffle(names)
        val_count = min(len(names) - 1, max(1, round(len(names) * load_state(project)["config"]["val_fraction"])))
        validation = set(names[:val_count])
        splits = {name: "val" if name in validation else "train" for name in names}

    for split in ("train", "val"):
        (staging / "images" / split).mkdir(parents=True, exist_ok=True)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        split = splits[sample["image_name"]]
        source = project / sample["manifest"]["image_path"]
        output_name = f"r{round_number:02d}_{sample['image_name']}"
        shutil.copy2(source, staging / "images" / split / output_name)
        (staging / "labels" / split / f"{Path(output_name).stem}.txt").write_text(
            _pose_label(sample["front"], sample["back"]), encoding="utf-8"
        )

    yaml = f"""path: {dataset.resolve()}
train: images/train
val: images/val

names:
  0: headplate

kpt_shape: [2, 2]

kpt_names:
  0:
    - front
    - back
"""
    (staging / "data.yaml").write_text(yaml, encoding="utf-8")
    if dataset.exists():
        shutil.rmtree(dataset)
    staging.replace(dataset)
    (dataset / "data.yaml").write_text(yaml, encoding="utf-8")
    record_round(
        project,
        round_number,
        annotations=str(annotation_path.relative_to(project)),
        dataset=str(dataset.relative_to(project)),
        annotation_summary=summary,
    )
    return dataset, summary


def train_round(project: Path, round_number: int, dataset: Path) -> Path:
    """Train one model version and return its stable best-weight path."""
    from ultralytics import YOLO

    project = resolve_under(workflow_root(), project)
    config = load_state(project)["config"]
    train_dir = round_dir(project, round_number) / "train"
    _reset_generated(train_dir, project)
    model_name = str(config["base_model"])
    model_path = Path(model_name).expanduser()
    if model_path.is_absolute() or (project / model_path).exists():
        model_name = str(resolve_under(project, model_path))
    elif model_path.parent != Path("."):
        raise FileNotFoundError(project / model_path)
    model = YOLO(model_name)

    def on_epoch_end(trainer: Any) -> None:
        progress = 0.05 + 0.45 * (trainer.epoch + 1) / trainer.epochs
        update_state(
            project,
            progress=progress,
            message=f"Training Round {round_number}: epoch {trainer.epoch + 1}/{trainer.epochs}",
        )

    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.train(
        data=str(dataset / "data.yaml"),
        epochs=int(config["epochs"]),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        device=str(config["device"]),
        project=str(train_dir),
        name="run",
        patience=0,
        exist_ok=True,
    )
    trained = Path(model.trainer.best)
    if not trained.exists():
        raise FileNotFoundError(f"Training did not produce best.pt: {trained}")
    stable = round_dir(project, round_number) / "model_best.pt"
    shutil.copy2(trained, stable)
    record_round(project, round_number, model=str(stable.relative_to(project)))
    return stable


def heading_degrees(front: tuple[float, float], back: tuple[float, float]) -> float:
    """Return CCW heading where up=0, left=90, down=180, and right=270."""
    dx, dy = front[0] - back[0], front[1] - back[1]
    return float(np.degrees(np.arctan2(-dx, -dy)) % 360)


def _draw_detection(frame: np.ndarray, detection: dict[str, Any]) -> None:
    """Draw the selected bounding box, keypoints, confidence, and heading."""
    x1, y1, x2, y2 = detection["box"]
    front = tuple(round(value) for value in detection["front"])
    back = tuple(round(value) for value in detection["back"])
    center = tuple(round((a + b) / 2) for a, b in zip(detection["front"], detection["back"]))
    theta = math.radians(detection["hd_deg"])
    end = (round(center[0] - 80 * math.sin(theta)), round(center[1] - 80 * math.cos(theta)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
    cv2.putText(
        frame,
        f"headplate {detection['det_conf']:.2f}",
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(frame, front, 7, (0, 0, 255), -1)
    cv2.circle(frame, back, 7, (255, 80, 0), -1)
    cv2.putText(frame, "front", (front[0] + 8, front[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "back", (back[0] + 8, back[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 2)
    cv2.arrowedLine(frame, center, end, (0, 220, 255), 4, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(
        frame,
        f"HD {detection['hd_deg']:.1f} deg",
        (end[0] + 10, end[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )


def _best_detection(result: Any) -> dict[str, Any] | None:
    """Return the highest-confidence valid pose detection."""
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0 or len(result.keypoints) == 0:
        return None
    confidences = result.boxes.conf.cpu().numpy()
    index = int(np.argmax(confidences))
    points = result.keypoints.xy[index].cpu().numpy()
    if len(points) < 2:
        return None
    front, back = points[0].astype(float), points[1].astype(float)
    if (
        not np.all(np.isfinite([*front, *back]))
        or np.linalg.norm(front) <= 1e-6
        or np.linalg.norm(back) <= 1e-6
        or np.linalg.norm(front - back) <= 1e-6
    ):
        return None
    box = result.boxes.xyxy[index].cpu().numpy().astype(int).tolist()
    return {
        "box": box,
        "front": front.tolist(),
        "back": back.tolist(),
        "hd_deg": heading_degrees(tuple(front), tuple(back)),
        "det_conf": float(confidences[index]),
    }


def infer_round(project: Path, round_number: int, model_path: Path) -> list[dict[str, Any]]:
    """Run inference once per video while writing CSV, JSON, and overlay video."""
    from ultralytics import YOLO

    project = resolve_under(workflow_root(), project)
    state = load_state(project)
    config = state["config"]
    videos = [project / value for value in state["videos"]]
    metadata = [_video_metadata(video) for video in videos]
    total_frames = sum(item["frames"] for item in metadata)
    completed = 0
    output_dir = round_dir(project, round_number) / "results"
    _reset_generated(output_dir, project)
    model = YOLO(str(model_path))
    artifacts = []

    for video_index, (video, meta) in enumerate(zip(videos, metadata), 1):
        stem = f"v{video_index:02d}_{_slug(video.stem)}"
        csv_path = output_dir / f"{stem}_pose.csv"
        json_path = output_dir / f"{stem}_hd.json"
        overlay_path = output_dir / f"{stem}_overlay.mp4"
        writer = cv2.VideoWriter(
            str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"])
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create overlay video: {overlay_path}")
        valid_frames, json_frames, json_hd = 0, [], []
        results = model.predict(
            source=str(video),
            conf=float(config["conf"]),
            imgsz=int(config["imgsz"]),
            device=str(config["device"]),
            stream=True,
            verbose=False,
        )
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(
                ["frame", "center_x", "center_y", "front_x", "front_y", "back_x", "back_y", "hd_deg", "det_conf"]
            )
            for frame_index, result in enumerate(results):
                frame = result.orig_img.copy()
                detection = _best_detection(result)
                if detection is None:
                    csv_writer.writerow([frame_index, *([math.nan] * 8)])
                else:
                    front, back = detection["front"], detection["back"]
                    center = [(front[0] + back[0]) / 2, (front[1] + back[1]) / 2]
                    csv_writer.writerow(
                        [frame_index, *center, *front, *back, detection["hd_deg"], detection["det_conf"]]
                    )
                    json_frames.append(frame_index)
                    json_hd.append(detection["hd_deg"])
                    valid_frames += 1
                    _draw_detection(frame, detection)
                writer.write(frame)
                completed += 1
                if completed % 100 == 0 or completed == total_frames:
                    update_state(
                        project,
                        progress=0.55 + 0.35 * completed / total_frames,
                        message=f"Inference Round {round_number}: {completed}/{total_frames} frames",
                    )
        writer.release()
        _atomic_json(json_path, {str(config["json_key"]): {"frames": json_frames, "hd": json_hd}})
        artifacts.append(
            {
                "video": video.relative_to(project).as_posix(),
                "csv": csv_path.relative_to(project).as_posix(),
                "json": json_path.relative_to(project).as_posix(),
                "overlay": overlay_path.relative_to(project).as_posix(),
                "frames": meta["frames"],
                "valid_frames": valid_frames,
            }
        )
    _atomic_json(output_dir / "summary.json", {"round": round_number, "artifacts": artifacts})
    record_round(project, round_number, results=str(output_dir.relative_to(project)), artifacts=artifacts)
    return artifacts


def _float(value: str) -> float:
    """Parse a CSV float, returning NaN for empty values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def select_review_frames(rows: list[dict[str, str]], amount: int, seed: int) -> list[tuple[int, str]]:
    """Select low-confidence, heading-jump, and random review frames."""
    amount = min(amount, len(rows))
    low_count = round(amount * 0.4)
    jump_count = round(amount * 0.4)
    selected: dict[int, str] = {}
    low = sorted(rows, key=lambda row: -1 if math.isnan(_float(row["det_conf"])) else _float(row["det_conf"]))
    for row in low:
        selected[int(row["frame"])] = "low_confidence"
        if len(selected) >= low_count:
            break

    jumps = []
    previous = None
    for row in rows:
        heading = _float(row["hd_deg"])
        if previous is not None and not math.isnan(heading) and not math.isnan(previous):
            difference = abs((heading - previous + 180) % 360 - 180)
            jumps.append((difference, int(row["frame"])))
        previous = heading
    for _, frame in sorted(jumps, reverse=True):
        if frame not in selected:
            selected[frame] = "heading_jump"
        if sum(reason == "heading_jump" for reason in selected.values()) >= jump_count:
            break

    remaining = [int(row["frame"]) for row in rows if int(row["frame"]) not in selected]
    random.Random(seed).shuffle(remaining)
    for frame in remaining[: amount - len(selected)]:
        selected[frame] = "random"
    return sorted(selected.items())


def _prediction_results(row: dict[str, str], width: int, height: int) -> list[dict[str, Any]]:
    """Build Label Studio keypoint predictions from one inference row."""
    values = [_float(row[key]) for key in ("front_x", "front_y", "back_x", "back_y")]
    if not all(math.isfinite(value) for value in values):
        return []
    front_x, front_y, back_x, back_y = values
    return [
        {
            "from_name": "keypoints",
            "to_name": "image",
            "type": "keypointlabels",
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "x": front_x / width * 100,
                "y": front_y / height * 100,
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
            "value": {"x": back_x / width * 100, "y": back_y / height * 100, "width": 0.5, "keypointlabels": ["back"]},
        },
    ]


def prepare_review_round(project: Path, source_round: int, artifacts: list[dict[str, Any]]) -> Path:
    """Create the next Label Studio review set from inference artifacts."""
    project = resolve_under(workflow_root(), project)
    state = load_state(project)
    target_round = source_round + 1
    label_dir = round_dir(project, target_round) / "label_studio"
    frame_dir = label_dir / "frames"
    _reset_generated(label_dir, project)
    frame_dir.mkdir()
    tasks, manifest = [], []

    for video_index, artifact in enumerate(artifacts, 1):
        video = project / artifact["video"]
        meta = _video_metadata(video)
        with (project / artifact["csv"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        indexed = {int(row["frame"]): row for row in rows}
        selected = select_review_frames(
            rows, int(state["config"]["review_frames"]), int(state["config"]["seed"]) + video_index + source_round
        )
        capture = cv2.VideoCapture(str(video))
        for frame_index, reason in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f"Cannot read frame {frame_index} from {video}")
            image_name = f"v{video_index:02d}_{_slug(video.stem)}__frame_{frame_index:08d}.jpg"
            image_path = frame_dir / image_name
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                capture.release()
                raise RuntimeError(f"Cannot write image: {image_path}")
            row = indexed[frame_index]
            prediction = _prediction_results(row, meta["width"], meta["height"])
            tasks.append(
                {
                    "data": {"image": _label_studio_url(image_path)},
                    "predictions": [
                        {
                            "model_version": f"headplate-round-{source_round}",
                            "score": 0.0 if math.isnan(_float(row["det_conf"])) else _float(row["det_conf"]),
                            "result": prediction,
                        }
                    ],
                    "meta": {"video": video.name, "frame": frame_index, "round": target_round, "reason": reason},
                }
            )
            manifest.append(
                {
                    "image_name": image_name,
                    "image_path": image_path.relative_to(project).as_posix(),
                    "video": video.relative_to(project).as_posix(),
                    "frame": frame_index,
                    "timestamp_sec": f"{frame_index / meta['fps']:.6f}",
                    "selection_reason": reason,
                }
            )
        capture.release()

    _write_manifest(label_dir / "frame_manifest.csv", manifest)
    _atomic_json(label_dir / "label_studio_import.json", tasks)
    (label_dir / "label_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")
    record_round(
        project,
        target_round,
        label_studio_dir=str(label_dir.relative_to(project)),
        label_studio_import=str((label_dir / "label_studio_import.json").relative_to(project)),
        manifest=str((label_dir / "frame_manifest.csv").relative_to(project)),
        sample_count=len(manifest),
    )
    return label_dir


def process_round(project: Path, round_number: int, annotation_path: Path) -> None:
    """Convert annotations, train, infer, and optionally prepare the next review round."""
    project = resolve_under(workflow_root(), project)
    update_state(project, progress=0.01, message=f"Validating Round {round_number} annotations")
    dataset, _ = build_dataset(project, round_number, annotation_path)
    update_state(project, progress=0.05, message=f"Training Round {round_number}")
    model = train_round(project, round_number, dataset)
    update_state(project, progress=0.55, message=f"Running Round {round_number} inference")
    artifacts = infer_round(project, round_number, model)
    if round_number < int(load_state(project)["config"]["max_rounds"]):
        update_state(project, progress=0.92, message=f"Preparing Round {round_number + 1} review frames")
        prepare_review_round(project, round_number, artifacts)
        update_state(
            project,
            stage=f"WAITING_ROUND_{round_number + 1:02d}_ANNOTATIONS",
            current_round=round_number + 1,
            progress=1.0,
            message=f"Round {round_number + 1} files are ready for Label Studio",
            active_job=None,
        )
    else:
        update_state(
            project,
            stage="COMPLETE",
            current_round=round_number,
            progress=1.0,
            message=f"Workflow complete after Round {round_number}",
            active_job=None,
        )


def launch_worker(project: Path, action: str, annotation_path: Path | None = None) -> int:
    """Launch a durable worker process and return its PID."""
    project = resolve_under(workflow_root(), project)
    state = recover_dead_job(project)
    if state.get("active_job"):
        raise RuntimeError("A workflow job is already active")
    if action == "prepare-round1" and state["stage"] != "CONFIGURED":
        raise ValueError("Round 1 can only be prepared from a newly configured project")
    if action == "process-round":
        expected = f"WAITING_ROUND_{int(state['current_round']):02d}_ANNOTATIONS"
        if state["stage"] != expected:
            raise ValueError("Annotations can only be processed while a labeling round is waiting")
        if annotation_path is None or annotation_path.suffix.casefold() != ".json":
            raise ValueError("Select a Label Studio JSON export")
    command = [sys.executable, str(Path(__file__).resolve()), action, str(project)]
    if annotation_path is not None:
        command.append(str(resolve_under(project, annotation_path)))
    log_path = workspace(project) / "workflow.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    update_state(
        project,
        active_job={"pid": process.pid, "action": action, "started_at": utc_now()},
        error=None,
        message=f"Starting {action}",
        progress=0.0,
    )
    return process.pid


def recover_dead_job(project: Path) -> dict[str, Any]:
    """Mark an unexpectedly terminated worker as failed."""
    project = resolve_under(workflow_root(), project)
    state = load_state(project)
    job = state.get("active_job")
    if not job:
        return state
    try:
        os.kill(int(job["pid"]), 0)
    except (OSError, ProcessLookupError):
        return update_state(
            project,
            active_job=None,
            error="The background worker stopped unexpectedly. Check workflow.log and retry.",
        )
    return state


def run_worker(action: str, project: Path, annotation_path: Path | None) -> int:
    """Execute one worker action and persist failures for the web UI."""
    project = resolve_under(workflow_root(), project)
    state = load_state(project)
    previous_stage = state["stage"]
    update_state(
        project,
        stage="RUNNING",
        active_job={"pid": os.getpid(), "action": action, "started_at": utc_now()},
        error=None,
    )
    try:
        if action == "prepare-round1":
            prepare_round1(project)
        elif action == "process-round":
            if annotation_path is None:
                raise ValueError("process-round requires a Label Studio JSON path")
            process_round(project, int(state["current_round"]), annotation_path)
        else:
            raise ValueError(f"Unknown action: {action}")
    except Exception as error:  # noqa: BLE001 - every worker failure must be persisted for recovery
        traceback.print_exc()
        update_state(
            project,
            stage=previous_stage,
            active_job=None,
            error=str(error),
            message=f"{action} failed",
        )
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    """Parse background-worker arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-round1", "process-round"))
    parser.add_argument("project", type=Path)
    parser.add_argument("annotation_path", nargs="?", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_worker(args.action, args.project, args.annotation_path))
