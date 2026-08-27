# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Create deterministic train/val folders from flat YOLO image and label directories."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse dataset split arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Dataset root containing flat images/ and labels/ directories.")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--val-count", type=int, help="Exact validation image count.")
    size.add_argument("--val-ratio", type=float, default=0.2, help="Validation fraction (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move originals instead of copying them.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    parser.add_argument("--dry-run", action="store_true", help="Show the split without writing files.")
    return parser.parse_args()


def indexed_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    """Index direct child files by stem and reject ambiguous duplicate stems."""
    files: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in files:
            raise ValueError(f"Duplicate stem '{path.stem}': {files[path.stem]} and {path}")
        files[path.stem] = path
    return files


def main() -> None:
    """Split paired flat images and labels into train and validation directories."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    images_dir, labels_dir = root / "images", root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir}")

    images = indexed_files(images_dir, IMAGE_SUFFIXES)
    labels = indexed_files(labels_dir, {".txt"})
    paired = sorted(images.keys() & labels.keys())
    if not paired:
        raise SystemExit("No paired image/label files found")

    if args.val_count is not None:
        val_count = args.val_count
    else:
        if not 0 < args.val_ratio < 1:
            raise SystemExit("--val-ratio must be between 0 and 1")
        val_count = round(len(paired) * args.val_ratio)
    if not 0 < val_count < len(paired):
        raise SystemExit(f"Validation count must be between 1 and {len(paired) - 1}, got {val_count}")

    validation = set(random.Random(args.seed).sample(paired, val_count))
    destinations: list[tuple[Path, Path]] = []
    for stem in paired:
        split = "val" if stem in validation else "train"
        destinations.extend(
            (
                (images[stem], images_dir / split / images[stem].name),
                (labels[stem], labels_dir / split / labels[stem].name),
            )
        )

    conflicts = [target for _, target in destinations if target.exists()]
    if conflicts and not args.overwrite:
        example = "\n".join(str(path) for path in conflicts[:5])
        raise SystemExit(f"{len(conflicts)} destination files already exist; pass --overwrite. Examples:\n{example}")

    missing_labels = len(images.keys() - labels.keys())
    missing_images = len(labels.keys() - images.keys())
    mode = "move" if args.move else "copy"
    print(
        f"paired={len(paired)} train={len(paired) - val_count} val={val_count} "
        f"missing_labels={missing_labels} missing_images={missing_images} mode={mode} seed={args.seed}"
    )
    if args.dry_run:
        return

    for source, target in destinations:
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(source, target)
        else:
            shutil.copy2(source, target)
    print(f"Split written under {root}")


if __name__ == "__main__":
    main()
