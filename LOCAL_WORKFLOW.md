# Local YOLO26 workflow

This worktree keeps the official Ultralytics source and adds familiar YOLOv5-style entry scripts for the local mouse
dataset. The dataset is referenced in place, so its images and labels are not duplicated.

## Setup

```bash
cd /Users/vxf1610/developer/yolov26
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
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

# Edit SOURCE, DESTINATION, and AMOUNT in pick_train.py, then copy a random JPG subset
python pick_train.py

# Preview a deterministic 80/20 split, then copy it into images/{train,val} and labels/{train,val}
python pick_val.py /path/to/flat_dataset --val-ratio 0.2 --dry-run
python pick_val.py /path/to/flat_dataset --val-ratio 0.2
```

YOLO detection datasets require `train` and `val` entries; `test` is optional. Label the selected images, then use
`pick_val.py` to create the required training and validation split.

`pick_val.py` copies by default. Add `--move` only when the flat originals should be relocated. Existing destination
files are protected unless `--overwrite` is passed.

## Layout and updates

- `ultralytics/` contains the canonical YOLO26 implementation and model YAMLs.
- `data/mouse_1909.yaml` points to the existing YOLOv5 mouse dataset.
- `runs/` receives local train, validation, prediction, and tracking outputs and is ignored by Git.
- The custom work lives on branch `local/yolov26-workflow`; update it with `git pull --rebase` when upstream changes.
