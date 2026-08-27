# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLOv5-style entry point for focused YOLO26 export benchmarking."""

from local_cli import ROOT, run

if __name__ == "__main__":
    run(
        "benchmark",
        "model=yolo26n.pt",
        f"data={ROOT / 'data/mouse_1909.yaml'}",
        "format=onnx",
        "imgsz=640",
    )
