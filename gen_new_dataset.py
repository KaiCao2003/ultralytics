import json
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs


ROOT = Path("/Users/vxf1610/Developer/yolov26")

OLD_DATASET = ROOT / "data/datasets/headplate_pose_dataset"
NEW_DATASET = ROOT / "data/datasets/headplate_pose_dataset_v2"

ROUND2_JSON = "/Users/vxf1610/Downloads/project-10-at-2026-08-28-16-54-61125716.json"

# This matches your Label Studio document root
LOCAL_FILES_ROOT = ROOT / "data"

BOX_SCALE = 2.5


# --------------------------------------------------
# Copy old dataset -> v2
# --------------------------------------------------

if NEW_DATASET.exists():
    shutil.rmtree(NEW_DATASET)

shutil.copytree(OLD_DATASET, NEW_DATASET)

train_images = NEW_DATASET / "images/train"
train_labels = NEW_DATASET / "labels/train"

train_images.mkdir(parents=True, exist_ok=True)
train_labels.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load round 2 annotations
# --------------------------------------------------

with open(ROUND2_JSON, "r") as f:
    tasks = json.load(f)


added = 0
skipped = 0


for task in tasks:

    annotations = task.get("annotations", [])

    if not annotations:
        skipped += 1
        continue

    # Use latest human annotation
    annotation = annotations[-1]

    if annotation.get("was_cancelled", False):
        skipped += 1
        continue

    keypoints = {}

    for result in annotation.get("result", []):

        if result.get("type") != "keypointlabels":
            continue

        labels = result["value"].get("keypointlabels", [])

        if not labels:
            continue

        label = labels[0]

        if label not in {"front", "back"}:
            continue

        # Label Studio coordinates are already percentages
        x = float(result["value"]["x"]) / 100.0
        y = float(result["value"]["y"]) / 100.0

        keypoints[label] = (x, y)

    # Need both points
    if "front" not in keypoints or "back" not in keypoints:
        skipped += 1
        continue

    front_x, front_y = keypoints["front"]
    back_x, back_y = keypoints["back"]

    # --------------------------------------------------
    # Resolve local image path
    # --------------------------------------------------

    image_url = task["data"]["image"]

    # Example:
    # /data/local-files/?d=labelstudio_round2/frames/bright_000101.jpg
    query = parse_qs(urlparse(image_url).query)

    if "d" not in query:
        print("Cannot resolve:", image_url)
        skipped += 1
        continue

    relative_path = query["d"][0]
    source_image = LOCAL_FILES_ROOT / relative_path

    if not source_image.exists():
        print("Missing image:", source_image)
        skipped += 1
        continue

    # Prefix to guarantee unique name
    output_name = "round2_" + source_image.name

    output_image = train_images / output_name
    output_label = train_labels / (Path(output_name).stem + ".txt")

    # --------------------------------------------------
    # Generate YOLO bbox from front/back
    # --------------------------------------------------

    dx = front_x - back_x
    dy = front_y - back_y

    distance = (dx**2 + dy**2) ** 0.5

    xc = (front_x + back_x) / 2
    yc = (front_y + back_y) / 2

    box_size = distance * BOX_SCALE

    x1 = max(0.0, xc - box_size / 2)
    y1 = max(0.0, yc - box_size / 2)

    x2 = min(1.0, xc + box_size / 2)
    y2 = min(1.0, yc + box_size / 2)

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    # YOLO Pose:
    #
    # class xc yc w h front_x front_y back_x back_y

    label = (
        f"0 "
        f"{xc:.8f} {yc:.8f} "
        f"{w:.8f} {h:.8f} "
        f"{front_x:.8f} {front_y:.8f} "
        f"{back_x:.8f} {back_y:.8f}\n"
    )

    shutil.copy2(source_image, output_image)
    output_label.write_text(label)

    added += 1


# --------------------------------------------------
# Update data.yaml
# --------------------------------------------------

yaml_text = f"""path: {NEW_DATASET.resolve()}

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

(NEW_DATASET / "data.yaml").write_text(yaml_text)


print()
print(f"Created: {NEW_DATASET}")
print(f"Added round-2 samples: {added}")
print(f"Skipped incomplete/invalid samples: {skipped}")