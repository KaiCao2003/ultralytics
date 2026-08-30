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

# Edit SOURCE, DESTINATION, and AMOUNT in pick_val.py, then copy a random JPG subset
python pick_val.py
```

YOLO detection still requires a separate `val` entry; `test` is optional. `pick_val.py` only copies the requested
random JPGs and does not create dataset splits.

## Layout and updates

- `ultralytics/` contains the canonical YOLO26 implementation and model YAMLs.
- `data/mouse_1909.yaml` points to the existing YOLOv5 mouse dataset.
- `runs/` receives local train, validation, prediction, and tracking outputs and is ignored by Git.
- The custom work lives on branch `local/yolov26-workflow`; update it with `git pull --rebase` when upstream changes.

## Headplate active-learning web app

The Linux web app runs from `/mnt/ssd4.1/apps/headplate-yolo` and is exposed by nginx at `/yolo`. It never uploads or
downloads experiment files through the browser. Users place videos and Label Studio exports directly in a project
folder at any depth under `/mnt/senzailab`, and every generated artifact stays in that folder. The folder browser can
also create a new child folder without exposing any path outside that data root.

### Project flow

1. Open `http://<server>/yolo`, sign in, then browse to any nested folder under `/mnt/senzailab` or create a new folder.
2. Put one or more videos directly inside the chosen folder, press Refresh, and select **Use this folder**.
3. Initialize the project and prepare Round 1. The app creates 100 uniformly sampled frames plus:
   - `headplate-yolo/round_01/label_studio/label_config.xml`
   - `headplate-yolo/round_01/label_studio/label_studio_import.json`
   - `headplate-yolo/round_01/label_studio/frame_manifest.csv`
4. Import the JSON into Label Studio, label `front` and `back`, and put the exported JSON directly in the project folder.
5. Select that export in the app. The background worker converts it to YOLO Pose, trains model v1, analyzes every
   source video with identity-aware tracking, writes pose CSV/position CSV/continuous HD JSON/overlay MP4, and prepares
   pre-labeled Round 2 review frames. It retains the current `track_id` when possible and forward-fills pose, position,
   and HD across missed detections while leaving `det_conf` as `nan`.
6. Repeat the Label Studio export/import step for Round 2. The app creates dataset v2, trains model v2, and writes the
   final results under `headplate-yolo/round_02/results/`.

Label Studio must use `/mnt/senzailab` as its local-files document root:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/mnt/senzailab
```

### Linux installation

```bash
cd /mnt/ssd4.1/apps/headplate-yolo
/home/kai/.local/bin/uv venv --python 3.12
/home/kai/.local/bin/uv pip install --python .venv/bin/python -e ".[workflow]"
```

The user unit and nginx location are in `deploy/headplate-yolo.service` and `deploy/nginx-yolo-location.conf`. The
service reuses `/home/kai/.config/lab-access-gate/pi-first-name.env`, matching the other lab apps. Install the unit at
`~/.config/systemd/user/headplate-yolo.service`, include the nginx location file inside the lab app server block, then
run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now headplate-yolo.service
sudo nginx -t
sudo systemctl reload nginx
```

The service binds only to `127.0.0.1:3007`; nginx is the public entry point. Its default data root is
`/mnt/senzailab`, and the app rejects paths that escape that root.
