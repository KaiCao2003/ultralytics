from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

model.train(
    data="data/datasets/headplate_pose_dataset_v2/data.yaml",
    epochs=2000,
    imgsz=1024,
    batch=-1,
    name="headplate_pose_v2",
    patience=0,
)