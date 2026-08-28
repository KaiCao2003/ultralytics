from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

model.train(
    data="data/datasets/headplate_pose_dataset/data.yaml",
    epochs=200,
    imgsz=1024,
    batch=8,
)