# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import random
import shutil
from pathlib import Path

SOURCE = Path("/path/to/source/images")
DESTINATION = Path("/path/to/training/images")
AMOUNT = 100

for image in random.sample(list(SOURCE.glob("*.jpg")), AMOUNT):
    shutil.copy2(image, DESTINATION)
