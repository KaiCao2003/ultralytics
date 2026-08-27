# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Shared adapter for the YOLOv5-style local entry scripts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(mode: str, *defaults: str) -> None:
    """Run an Ultralytics CLI mode with overridable local defaults."""
    sys.argv[1:1] = [mode, *defaults]
    from ultralytics.cfg import entrypoint

    entrypoint()
