# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLOv5-style entry point for YOLO26 detection training."""

from local_cli import ROOT, run

if __name__ == "__main__":
    run(
        "train",
        "model=yolo26n.pt",
        f"data={ROOT / 'data/mouse_1909.yaml'}",
        "epochs=100",
        "imgsz=640",
        f"project={ROOT / 'runs/train'}",
        "name=mouse_yolo26n",
    )
