# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Extract evenly sampled JPEG frames from one video or a directory of videos."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_cli import ROOT

VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    """Parse frame extraction arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "data/videos", help="Video file or directory.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/frames", help="Output directory.")
    parser.add_argument("--fps", type=float, default=1.0, help="Target frames per second.")
    parser.add_argument("--quality", type=int, default=95, choices=range(1, 101), metavar="1-100")
    parser.add_argument("--recursive", action="store_true", help="Search source subdirectories.")
    parser.add_argument("--overwrite", action="store_true", help="Replace frames with matching filenames.")
    return parser.parse_args()


def find_videos(source: Path, recursive: bool) -> list[Path]:
    """Return supported videos under source."""
    source = source.expanduser().resolve()
    if source.is_file():
        return [source] if source.suffix.lower() in VIDEO_SUFFIXES else []
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in source.glob(pattern) if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def extract_frames(video: Path, output: Path, fps: float, quality: int, overwrite: bool) -> int:
    """Extract frames from one video and return the saved frame count."""
    import cv2

    output.mkdir(parents=True, exist_ok=True)
    if not overwrite and next(output.glob("*.jpg"), None):
        raise FileExistsError(f"{output} already contains JPEGs; pass --overwrite to replace matching files")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Cannot determine FPS: {video}")

    interval = max(1.0, source_fps / fps)
    frame_index = saved = 0
    next_frame = 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index + 1e-9 >= next_frame:
            saved += 1
            target = output / f"{saved:06d}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                capture.release()
                raise RuntimeError(f"Failed to write frame: {target}")
            next_frame += interval
        frame_index += 1
    capture.release()
    return saved


def main() -> None:
    """Extract frames from all selected videos."""
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero")
    videos = find_videos(args.source, args.recursive)
    if not videos:
        raise SystemExit(f"No supported videos found at {args.source}")

    source_root = args.source.resolve() if args.source.is_dir() else args.source.resolve().parent
    for video in videos:
        relative = video.resolve().relative_to(source_root).with_suffix("")
        target = args.output.expanduser().resolve() / relative
        count = extract_frames(video, target, args.fps, args.quality, args.overwrite)
        print(f"{video} -> {target} ({count} frames)")


if __name__ == "__main__":
    main()
