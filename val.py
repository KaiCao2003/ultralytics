# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLOv5-style entry point for YOLO26 validation."""

from local_cli import ROOT, run

if __name__ == "__main__":
    run(
        "val",
        "model=yolo26n.pt",
        f"data={ROOT / 'data/mouse_1909.yaml'}",
        "imgsz=640",
        f"project={ROOT / 'runs/val'}",
        "name=mouse_yolo26n",
    )
