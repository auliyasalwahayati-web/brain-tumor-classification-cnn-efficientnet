
from pathlib import Path

from src.config import TRAIN_DIR, TEST_DIR, CLASS_NAMES

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

for split, folder in [
    ("TRAINING", TRAIN_DIR),
    ("TESTING", TEST_DIR)
]:
    print(f"\n===== {split} =====")

    if not folder.exists():
        print("Folder tidak ditemukan:", folder)
        continue

    total = 0

    for class_name in CLASS_NAMES:
        class_dir = folder / class_name

        if not class_dir.exists():
            print(class_name, ": FOLDER TIDAK DITEMUKAN")
            continue

        count = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        total += count
        print(f"{class_name}: {count} gambar")

    print("Total gambar:", total)
