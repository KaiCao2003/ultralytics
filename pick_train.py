# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import random
import shutil
from pathlib import Path

SOURCE = Path("/path/to/source/images")
DESTINATION = Path("/path/to/training/images")
AMOUNT = 100


def main() -> None:
    """Copy a random JPG subset into the destination folder."""
    for image in random.sample(list(SOURCE.glob("*.jpg")), AMOUNT):
        shutil.copy2(image, DESTINATION)


if __name__ == "__main__":
    main()
