# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLOv5-style entry point for YOLO26 model export."""

from local_cli import run

if __name__ == "__main__":
    run("export", "model=yolo26n.pt", "format=onnx", "imgsz=640")
