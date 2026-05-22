import os
import json
import random
import shutil

IMAGE_DIR = "dataset/generated"
LABEL_DIR = "dataset/labels"

OUTPUT_DIR = "dataset/split"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def make_dirs():
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)


def split_dataset():
    make_dirs()

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    split_map = {
        "train": image_files[:train_end],
        "valid": image_files[train_end:valid_end],
        "test": image_files[valid_end:]
    }

    summary = {}

    for split, files in split_map.items():
        for image_file in files:
            image_id = os.path.splitext(image_file)[0]
            label_file = image_id + ".json"

            src_image = os.path.join(IMAGE_DIR, image_file)
            src_label = os.path.join(LABEL_DIR, label_file)

            dst_image = os.path.join(OUTPUT_DIR, split, "images", image_file)
            dst_label = os.path.join(OUTPUT_DIR, split, "labels", label_file)

            if not os.path.exists(src_label):
                print(f"라벨 없음: {label_file}")
                continue

            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_label, dst_label)

        summary[split] = len(files)

    summary_path = os.path.join(OUTPUT_DIR, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("데이터셋 분리 완료")
    print(summary)


if __name__ == "__main__":
    split_dataset()