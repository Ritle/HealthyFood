# prepare_dataset.py
import os
import shutil
import sys
from os.path import join

# === Настройка: укажи путь к Food-101 ВНЕ репозитория ===
FOOD101_SOURCE = "/root/datasets/food-101/food-101"  # ← ИЗМЕНИ НА СВОЙ ПУТЬ!
OUTPUT_DIR = "data/food-101-split"  # внутри репозитория (но в .gitignore)

def main():
    if not os.path.exists(FOOD101_SOURCE):
        print(f"❌ Ошибка: Food-101 не найден по пути:\n{FOOD101_SOURCE}")
        print("Убедитесь, что датасет распакован и путь указан верно.")
        sys.exit(1)

    # Создаём выходную директорию
    os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/test", exist_ok=True)

    # Загружаем классы
    with open(os.path.join(FOOD101_SOURCE, "meta", "classes.txt")) as f:
        classes = [line.strip() for line in f if line.strip()]

    print(f"📁 Найдено {len(classes)} классов. Создаём структуру в {OUTPUT_DIR}...")

    # Создаём папки классов
    for cls in classes:
        os.makedirs(f"{OUTPUT_DIR}/train/{cls}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/test/{cls}", exist_ok=True)

    def copy_split(split_name):
        list_file = os.path(join(FOOD101_SOURCE, "meta", f"{split_name}.txt"))
        if not os.path.exists(list_file):
            print(f"⚠️  Пропущен: {split_name}.txt")
            return

        count = 0
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if "/" not in line:
                    continue
                cls, img = line.split("/", 1)
                src = os.path.join(FOOD101_SOURCE, "images", cls, f"{img}.jpg")
                dst = os.path.join(OUTPUT_DIR, split_name, cls, f"{img}.jpg")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    count += 1
        print(f"✅ {split_name}: {count} изображений")

    copy_split("train")
    copy_split("test")
    print(f"\n🎉 Готово! Датасет подготовлен в: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()