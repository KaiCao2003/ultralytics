from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n-pose.pt")

    model.train(
        data="data/datasets/headplate_260921/data.yaml",
        epochs=2000,
        imgsz=1024,
        batch=64,  # Global batch: 16 images per GPU across four GPUs.
        device=[0, 1, 2, 3],
        name="headplate_260921",
        project="headplate_260921",
        patience=0,
    )
