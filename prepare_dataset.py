# prepare_dataset.py
import os
import shutil

DATA_ROOT = "data/food-101"
OUTPUT_ROOT = "data/food-101-split"

os.makedirs(f"{OUTPUT_ROOT}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/test", exist_ok=True)

# Копируем классы
with open(f"{DATA_ROOT}/meta/classes.txt") as f:
    classes = [line.strip() for line in f]

for cls in classes:
    os.makedirs(f"{OUTPUT_ROOT}/train/{cls}", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/test/{cls}", exist_ok=True)

# Функция копирования
def copy_images(split):
    with open(f"{DATA_ROOT}/meta/{split}.txt") as f:
        for line in f:
            cls, img = line.strip().split("/")
            src = f"{DATA_ROOT}/images/{cls}/{img}.jpg"
            dst = f"{OUTPUT_ROOT}/{split}/{cls}/{img}.jpg"
            shutil.copy(src, dst)

copy_images("train")
copy_images("test")
print("✅ Dataset split prepared in data/food-101-split/")