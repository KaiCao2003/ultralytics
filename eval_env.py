from ultralytics import YOLO

model = YOLO("runs/detect/environment_detect_120/environment_detect_120/weights/best.pt")
total_correct = total = 0

for env in ("old", "new"):
    correct = count = missed = 0
    for result in model.predict(
        source=f"runs/env_video_eval_20261012_1000/frames/{env}",
        imgsz=1024,
        conf=0.25,
        batch=32,
        device=0,
        stream=True,
        verbose=False,
    ):
        boxes = result.boxes
        predicted = model.names[int(boxes.cls[boxes.conf.argmax()])] if len(boxes) else "none"
        correct += predicted == env
        missed += predicted == "none"
        count += 1
    assert count == 1000, f"{env}: expected 1000 frames, found {count}"
    print(f"{env}: {correct}/{count} = {correct / count:.2%}, missed={missed}")
    total_correct += correct
    total += count

print(f"Overall: {total_correct}/{total} = {total_correct / total:.2%}")
