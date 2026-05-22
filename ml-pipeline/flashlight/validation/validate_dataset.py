import os
import json
from PIL import Image

IMAGE_DIR = "dataset/generated"
LABEL_DIR = "dataset/labels"

def validate_dataset():
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])
    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(".json")])

    print(f"이미지 개수: {len(image_files)}")
    print(f"라벨 개수: {len(label_files)}")

    if len(image_files) != len(label_files):
        print("이미지 개수와 라벨 개수가 다릅니다.")

    error_count = 0

    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0]
        label_file = image_id + ".json"

        image_path = os.path.join(IMAGE_DIR, image_file)
        label_path = os.path.join(LABEL_DIR, label_file)

        if not os.path.exists(label_path):
            print(f"라벨 없음: {label_file}")
            error_count += 1
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)

        bbox = label["bbox"]

        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]

        if x < 0 or y < 0:
            print(f"좌표 음수 오류: {label_file}")
            error_count += 1

        if w <= 0 or h <= 0:
            print(f"bbox 크기 오류: {label_file}")
            error_count += 1

        if x + w > img_w or y + h > img_h:
            print(f"bbox 이미지 범위 초과: {label_file}")
            error_count += 1

        if w < 20 or h < 20:
            print(f"객체가 너무 작음: {label_file} / {w}x{h}")

        if w > img_w * 0.5 or h > img_h * 0.5:
            print(f"객체가 너무 큼: {label_file} / {w}x{h}")

    print("-------------------------")
    if error_count == 0:
        print("검증 완료: 치명적인 오류 없음")
    else:
        print(f"검증 완료: 오류 {error_count}개 발견")

if __name__ == "__main__":
    validate_dataset()