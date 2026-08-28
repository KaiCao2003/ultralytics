import json
import random
import shutil
from pathlib import Path


# ============================================================
# Settings
# ============================================================

JSON_PATH = Path(
    "/Users/vxf1610/Downloads/project-9-at-2026-08-27-22-14-21a8512a.json"
)

# Label Studio 里图片实际所在的位置
IMAGE_DIR = Path(
    "data/train/images"
)

OUTPUT_DIR = Path(
    "data/datasets/headplate_pose_dataset"
)

VAL_FRACTION = 0.2
RANDOM_SEED = 42

# bbox 根据 front-back 距离生成
# box 长宽大约 = 两点距离 * BOX_SCALE
BOX_SCALE = 2.5


# ============================================================
# Load Label Studio export
# ============================================================

with open(JSON_PATH, "r") as f:
    tasks = json.load(f)


samples = []

for task in tasks:

    # Skip tasks without annotations
    if not task.get("annotations"):
        continue

    annotation = task["annotations"][0]

    # Skip cancelled annotation
    if annotation.get("was_cancelled", False):
        continue

    keypoints = {}

    for result in annotation["result"]:

        if result["type"] != "keypointlabels":
            continue

        label = result["value"]["keypointlabels"][0]

        # Label Studio x/y are percentages: 0-100
        x = result["value"]["x"] / 100.0
        y = result["value"]["y"] / 100.0

        keypoints[label] = (x, y)

    if "front" not in keypoints or "back" not in keypoints:
        print(
            "Skipping incomplete annotation:",
            task["data"]["image"]
        )
        continue

    # Example:
    # /data/local-files/?d=train/images/000003.jpg
    image_url = task["data"]["image"]

    filename = image_url.split("/")[-1]

    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        print("Image missing:", image_path)
        continue

    samples.append({
        "image_path": image_path,
        "filename": filename,
        "front": keypoints["front"],
        "back": keypoints["back"],
    })


print(f"Found {len(samples)} valid samples")


# ============================================================
# Train / val split
# ============================================================

random.seed(RANDOM_SEED)
random.shuffle(samples)

n_val = round(len(samples) * VAL_FRACTION)

val_samples = samples[:n_val]
train_samples = samples[n_val:]


# ============================================================
# Create directories
# ============================================================

for split in ["train", "val"]:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Create YOLO Pose labels
# ============================================================

def write_sample(sample, split):

    front_x, front_y = sample["front"]
    back_x, back_y = sample["back"]

    # --------------------------------------------------------
    # Estimate bbox from keypoint geometry
    # --------------------------------------------------------

    dx = front_x - back_x
    dy = front_y - back_y

    distance = (dx**2 + dy**2) ** 0.5

    xc = (front_x + back_x) / 2
    yc = (front_y + back_y) / 2

    # Use a square bbox centered on the headplate
    box_size = distance * BOX_SCALE

    w = box_size
    h = box_size

    # Clamp to image boundaries
    x1 = max(0.0, xc - w / 2)
    y1 = max(0.0, yc - h / 2)

    x2 = min(1.0, xc + w / 2)
    y2 = min(1.0, yc + h / 2)

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    w = x2 - x1
    h = y2 - y1

    # --------------------------------------------------------
    # YOLO Pose:
    #
    # class xc yc w h
    # front_x front_y
    # back_x back_y
    # --------------------------------------------------------

    values = [
        0,
        xc,
        yc,
        w,
        h,
        front_x,
        front_y,
        back_x,
        back_y,
    ]

    label_line = " ".join(
        f"{v:.8f}" if isinstance(v, float) else str(v)
        for v in values
    )

    label_path = (
        OUTPUT_DIR /
        "labels" /
        split /
        (Path(sample["filename"]).stem + ".txt")
    )

    label_path.write_text(label_line + "\n")

    # Copy image
    shutil.copy2(
        sample["image_path"],
        OUTPUT_DIR / "images" / split / sample["filename"]
    )


for sample in train_samples:
    write_sample(sample, "train")

for sample in val_samples:
    write_sample(sample, "val")


# ============================================================
# data.yaml
# ============================================================

yaml_text = f"""path: {OUTPUT_DIR.resolve()}
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

(OUTPUT_DIR / "data.yaml").write_text(yaml_text)


print()
print("Done")
print("Train:", len(train_samples))
print("Val:", len(val_samples))
print("Dataset:", OUTPUT_DIR)
print("YAML:", OUTPUT_DIR / "data.yaml")