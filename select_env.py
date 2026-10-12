import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

import cv2

capture = cv2.VideoCapture(sys.argv[1])
indices = sorted(random.Random(42).sample(range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))), 100))
images = []
for index in indices:
    if index != capture.get(cv2.CAP_PROP_POS_FRAMES):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if ok:
        images.append(frame)
capture.release()
votes = {"old": 0, "new": 0}
with redirect_stdout(sys.stderr):
    from ultralytics import YOLO

    model = YOLO(Path(__file__).resolve().parent / "weights/environment_detect_120.pt")
    for result in model.predict(images, imgsz=1024, conf=0.25, verbose=False):
        if len(result.boxes):
            votes[model.names[int(result.boxes.cls[result.boxes.conf.argmax()])]] += 1
print(f"Environment: new={votes['new']}/100 old={votes['old']}/100", file=sys.stderr)
print("old" if votes["old"] >= 80 else "new")
