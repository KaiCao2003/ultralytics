from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data/datasets/environment_detect_120/data.yaml",
    epochs=2000,
    imgsz=1024,
    batch=64,
    name="environment_detect_120",
    project="environment_detect_120",
    patience=0,
    mosaic=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.0,
)
