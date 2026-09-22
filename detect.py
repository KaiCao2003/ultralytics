# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLOv5-style entry point for YOLO26 prediction."""

from local_cli import ROOT, run

if __name__ == "__main__":
    run("predict", "model=yolo26n.pt", f"project={ROOT / 'runs/detect'}", "name=predict")
