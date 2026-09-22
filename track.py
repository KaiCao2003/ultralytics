# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Entry point for YOLO26 object tracking."""

from local_cli import ROOT, run

if __name__ == "__main__":
    run("track", "model=yolo26n.pt", f"project={ROOT / 'runs/track'}", "name=track")
