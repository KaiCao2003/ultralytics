import cv2
import numpy as np


# ============================================================
# Settings
# ============================================================

INPUT_VIDEO = "data/videos/bright.avi"
OUTPUT_VIDEO = "data/videos/tracked.avi"

# Reflective beads should be very bright.
THRESHOLD = 220

# Blob size filtering
MIN_AREA = 3
MAX_AREA = 200

# Number of reflective beads expected
N_BEADS = 5


# ============================================================
# Detect reflective beads
# ============================================================

def detect_beads(frame):
    """
    Returns:
        beads: [(x, y, area), ...]
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find bright pixels
    _, binary = cv2.threshold(
        gray,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Optional: clean isolated noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(binary)
    )

    beads = []

    # label 0 = background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if MIN_AREA <= area <= MAX_AREA:
            x, y = centroids[i]

            beads.append(
                (float(x), float(y), int(area))
            )

    # Largest blobs first
    beads.sort(
        key=lambda bead: bead[2],
        reverse=True
    )

    return beads[:N_BEADS]


# ============================================================
# Estimate position + orientation
# ============================================================

def estimate_pose(beads):
    """
    Very simple preliminary estimator.

    Position:
        centroid of all detected beads

    Orientation:
        vector from bead-cluster centroid toward
        the bead farthest from the centroid.

    Returns:
        center: (x, y)
        angle_deg: orientation in degrees
    """

    if len(beads) == 0:
        return None, None

    points = np.array(
        [[b[0], b[1]] for b in beads],
        dtype=np.float32
    )

    center = points.mean(axis=0)

    if len(points) < 2:
        return center, None

    # Find bead farthest from center
    distances = np.linalg.norm(
        points - center,
        axis=1
    )

    front_idx = np.argmax(distances)
    front = points[front_idx]

    dx = front[0] - center[0]
    dy = front[1] - center[1]

    # Image coordinates:
    # x -> right
    # y -> down
    angle_deg = np.degrees(
        np.arctan2(dx, -dy)
    )

    angle_deg %= 360

    return center, angle_deg


# ============================================================
# Draw
# ============================================================

def draw_tracking(frame, beads, center, angle_deg):
    output = frame.copy()

    # Draw each bead
    for i, (x, y, area) in enumerate(beads):
        p = (int(round(x)), int(round(y)))

        cv2.circle(
            output,
            p,
            5,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            str(i + 1),
            (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    if center is not None:
        cx, cy = center
        center_int = (
            int(round(cx)),
            int(round(cy))
        )

        cv2.circle(
            output,
            center_int,
            6,
            (0, 0, 255),
            -1
        )

        if angle_deg is not None:
            length = 60

            theta = np.radians(angle_deg)

            end_x = cx + length * np.sin(theta)
            end_y = cy - length * np.cos(theta)

            end = (
                int(round(end_x)),
                int(round(end_y))
            )

            cv2.arrowedLine(
                output,
                center_int,
                end,
                (0, 0, 255),
                3,
                tipLength=0.25
            )

            cv2.putText(
                output,
                f"HD: {angle_deg:.1f} deg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    cv2.putText(
        output,
        f"beads: {len(beads)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


# ============================================================
# Main
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open video: {INPUT_VIDEO}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"MJPG")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    beads = detect_beads(frame)

    center, angle_deg = estimate_pose(beads)

    output = draw_tracking(
        frame,
        beads,
        center,
        angle_deg
    )

    writer.write(output)

    frame_idx += 1

    if frame_idx % 500 == 0:
        print(f"Processed {frame_idx} frames")


cap.release()
writer.release()

print("Done.")
print(f"Saved to: {OUTPUT_VIDEO}")