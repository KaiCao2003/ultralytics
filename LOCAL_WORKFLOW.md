# Local YOLO26 workflow

This worktree keeps the official Ultralytics source and adds familiar YOLOv5-style entry scripts for the local mouse
dataset. The dataset is referenced in place, so its images and labels are not duplicated.

## Setup

```bash
cd /Users/vxf1610/developer/yolov26
uv venv --python 3.12
uv pip install --python .venv/bin/python -e . pandas
source .venv/bin/activate
python check_dataset.py
```

## Common commands

Ultralytics arguments use `key=value`. Values given on the command line override each script's defaults.

```bash
# Train the local mouse detector (default: yolo26n.pt, 100 epochs, 640 px)
python train.py batch=8 device=mps

# Resume a run
python train.py resume=runs/train/mouse_yolo26n/weights/last.pt

# Predict with the trained model
python detect.py model=runs/train/mouse_yolo26n/weights/best.pt source=/path/to/video.mp4 save_txt=True

# Validate, export, track, or benchmark
python val.py model=runs/train/mouse_yolo26n/weights/best.pt
python export.py model=runs/train/mouse_yolo26n/weights/best.pt format=onnx
python track.py model=runs/train/mouse_yolo26n/weights/best.pt source=/path/to/video.mp4
python benchmarks.py model=runs/train/mouse_yolo26n/weights/best.pt format=onnx device=cpu
```

The wrappers map the old layout to the current API:

| YOLOv5 habit                             | YOLO26 local entry                |
| ---------------------------------------- | --------------------------------- |
| `train.py --weights ... --data ...`      | `train.py model=... data=...`     |
| `detect.py --weights ... --source ...`   | `detect.py model=... source=...`  |
| `val.py --weights ...`                   | `val.py model=...`                |
| `export.py --weights ... --include onnx` | `export.py model=... format=onnx` |

## Data utilities

```bash
# Extract one frame per second with OpenCV; each video gets its own output folder
python extract.py /path/to/videos --output data/frames --fps 1

# Edit SOURCE, DESTINATION, and AMOUNT in pick_val.py, then copy a random JPG subset
python pick_val.py
```

YOLO detection still requires a separate `val` entry; `test` is optional. `pick_val.py` only copies the requested
random JPGs and does not create dataset splits.

## Tracking exports

The deployed recording script takes the recording root, date, and session number. Both detection scripts default to
the current `headplate_260921` weights; pass `--model` to select a different training run:

```bash
python detect_stream.py '/mnt/senzailab/Kai/#Recording/m20' 260918 11 \
  --model runs/pose/headplate_260921/headplate_260921/weights/best.pt
```

It preserves the deployed layout: `<session>/<date>.csv` is the flat tracking table; `data/<date>.csv` is the eight-line
Motive-format table; `data/processed/head_direction.json` contains the existing `hp4.frames` and `hp4.hd` arrays;
`data/<date>_hd.avi` is the annotated video. Coordinates remain image pixels, not calibrated world coordinates.
Tracking exports retain the existing gap-filling behavior; use `detect_raw.py` below when selecting new labels.

Both CSVs are written incrementally in the printed local staging directory. On interruption, completed rows and the
closed AVI remain there; an unfinished Motive CSV has an empty `Total Exported Frames` value. Leading rows in that CSV
are emitted when the first tracked pose becomes available, preserving the existing initial fill. After successful
inference, each file is copied to a temporary file beside its destination and then replaced. All local outputs remain
until every transfer succeeds. A failed transfer can leave a mixture of completed old/new destination files and a
temporary copy, but the complete local set is retained for recovery.

## Raw detection and another labeling round

Run a two-keypoint model (`front`, then `back`) on every video frame:

```bash
python detect_raw.py /path/to/recording.avi --model /path/to/best.pt
```

With no arguments, it uses `VIDEO_PATH` and `MODEL_PATH` at the top of `detect_raw.py`.

The default output is the local project directory `runs/pose/predict_raw/`, regardless of where the input video is stored.
Repeated runs use `runs/pose/predict_raw-2/`, `runs/pose/predict_raw-3/`, etc.
Pass `--output runs/pose/another_name` to change the name.
The script prints the absolute output directory and saves every returned detection in `raw.csv`,
including bounding boxes, detection confidence, and original keypoint coordinates in image pixels. Multiple detections
produce multiple rows with the same zero-based `frame`; a missed frame has `detection_count=0`, `detection_index=-1`,
and empty measurements. Missing keypoint confidence is left empty for models trained with `kpt_shape: [2, 2]`.
There is no tracking, interpolation, forward filling, or smoothing. `run.json` records the video, model, inference
settings, dimensions, FPS, and completed frame count. CSV rows are written during inference and survive an interruption;
an interrupted run has no `frames_processed` entry in `run.json`.

The default inference threshold is `--conf 0.05` to retain weak predictions for review. These are model predictions
after normal inference filtering, not the network's internal tensors. Use `--conf 0.25` to match the tracking script's
threshold. Center and heading are derived columns: up = 0, left = 90, down = 180, right = 270 degrees in image coordinates.

After inference, the same command selects up to 100 suspected bad frames and 100 OK candidates. Bad reasons include
misses, multiple detections, low confidence, invalid keypoints, and abrupt heading changes between adjacent frames.
OK means no rule fired; it is a review candidate, not verified ground truth. Sampling uses seed 42 and at least one
second between selected frames within each group, so short videos or small pools can produce fewer samples.

```bash
python detect_raw.py /path/to/recording.avi --model /path/to/best.pt \
  --bad 100 --ok 100 --min-gap-seconds 1 --confidence-threshold 0.6 --jump-threshold 45

# Save only the raw predictions now.
python detect_raw.py /path/to/recording.avi --model /path/to/best.pt --raw-only

# Change the selection later without running the model again.
python ls_predict.py --video /path/to/recording.avi --csv runs/pose/predict_raw/raw.csv \
  --output runs/pose/predict_raw --bad 150 --ok 100
```

`review.csv` contains one row per frame with its review group, reasons, and selection flag. The exported PNGs are the
original decoded frames without overlays. Import `label_studio_import.json` for both groups, or import
`label_studio_bad.json` and `label_studio_ok.json` separately. Use the generated `label_config.xml` for the `front` and
`back` point labels. Prelabels use only the current frame's highest-confidence valid pose; all detections remain in
`raw.csv`, and missing or invalid poses have no prelabels.

Enable Label Studio local file serving with `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and set
`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` to the absolute project directory containing `runs/` (for example,
`/home/kai/yolov26`). The imports reference `runs/pose/predict_raw/frames/...` relative to that root. Pass the same directory through
`--local-files-root` if it differs. Review output must be under that directory. Correct the point labels in Label Studio
and export the human annotations for the next dataset round; predictions are not human annotations.

中文：`detect_raw.py` 逐帧保存原始预测，不追踪、不补值。检测结束后，按漏检、多目标、低置信度和方向跳变等规则粗筛，
分别抽取疑似错误和 OK 候选帧。导入生成的 Label Studio JSON 后复核并修改 `front/back` 标注。
用 `--raw-only` 只做检测；之后运行 `ls_predict.py` 可以直接从已有 CSV 重新挑帧，不必重新检测。

## Layout and updates

- `ultralytics/` contains the canonical YOLO26 implementation and model YAMLs.
- `data/mouse_1909.yaml` points to the existing YOLOv5 mouse dataset.
- `runs/` receives local train, validation, prediction, and tracking outputs and is ignored by Git.
- The custom work lives on branch `local/yolov26-workflow`; update it with `git pull --rebase` when upstream changes.
