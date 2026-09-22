import random
import shutil
from pathlib import Path

SOURCE = Path("data/frames/dark")
DESTINATION = Path("data/train/images/")
AMOUNT = 120

for image in random.sample(list(SOURCE.glob("*.jpg")), AMOUNT):
    shutil.copy2(image, DESTINATION)
