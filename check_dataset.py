# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Validate YOLO detection image/label pairing and normalized label values."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_cli import ROOT
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths


def parse_args() -> argparse.Namespace:
    """Parse dataset validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=ROOT / "data/mouse_1909.yaml")
    return parser.parse_args()


def image_files(location: str | list[str]) -> list[Path]:
    """Resolve image files from YOLO split directories or text manifests."""
    locations = [location] if isinstance(location, str) else location
    files: list[Path] = []
    for item in locations:
        path = Path(item)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS)
        elif path.is_file():
            parent = path.parent
            files.extend(
                Path(line) if Path(line).is_absolute() else parent / line for line in path.read_text().splitlines()
            )
        else:
            raise FileNotFoundError(path)
    return sorted(files)


def check_label(path: Path, class_count: int) -> tuple[int, list[str]]:
    """Return instance count and errors for one YOLO detection label."""
    errors: list[str] = []
    instances = 0
    if not path.exists():
        return 0, ["missing label"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        try:
            numbers = [float(value) for value in values]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric value")
            continue
        if len(numbers) != 5:
            errors.append(f"line {line_number}: expected 5 columns, got {len(numbers)}")
            continue
        class_id, x, y, width, height = numbers
        if not class_id.is_integer() or not 0 <= class_id < class_count:
            errors.append(f"line {line_number}: invalid class {class_id:g}")
        if not all(0 <= value <= 1 for value in (x, y, width, height)) or width <= 0 or height <= 0:
            errors.append(f"line {line_number}: invalid normalized box")
        instances += 1
    return instances, errors


def main() -> None:
    """Validate every configured training and validation label."""
    args = parse_args()
    data = check_det_dataset(str(args.data.expanduser().resolve()), autodownload=False)
    failures: list[str] = []
    for split in ("train", "val"):
        images = image_files(data[split])
        labels = [Path(path) for path in img2label_paths([str(path) for path in images])]
        instances = 0
        for image, label in zip(images, labels):
            count, errors = check_label(label, data["nc"])
            instances += count
            failures.extend(f"{split}: {image.name}: {error}" for error in errors)
        print(
            f"{split}: {len(images)} images, {len(labels) - sum(not p.exists() for p in labels)} labels, {instances} objects"
        )

    if failures:
        print("\n".join(failures[:50]))
        raise SystemExit(f"Dataset check failed with {len(failures)} issue(s)")
    print(f"Dataset OK: {data['path']} ({data['nc']} class(es): {data['names']})")


if __name__ == "__main__":
    main()
