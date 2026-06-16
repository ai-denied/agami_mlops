import os
import json
import random

BASE_DIR = "processed/dynamic"
OUTPUT_FILE = "merged_dynamic_features_sampled.json"

BOT_TYPES = [
    "grid_search",
    "known_target",
    "other_search",
    "random_search"
]

BOT_SAMPLE_SIZE = 2500
RANDOM_SEED = 42


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_json_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"폴더 없음: {folder_path}")
        return []

    return sorted([
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".json")
    ])


def merge_sampled_data():
    random.seed(RANDOM_SEED)

    merged_data = []
    user_count = 1

    # 1. human 데이터는 전부 가져오기
    human_dir = os.path.join(BASE_DIR, "human")
    human_files = get_json_files(human_dir)

    print("human 파일 개수:", len(human_files))

    for file_path in human_files:
        data = load_json(file_path)

        data["user_id"] = f"user{user_count}"
        data["source_type"] = "human"
        data["bot_type"] = None
        data["original_file"] = os.path.basename(file_path)

        merged_data.append(data)
        user_count += 1

    # 2. bot 데이터는 각 폴더별 2300개 랜덤 샘플링
    for bot_type in BOT_TYPES:
        bot_dir = os.path.join(BASE_DIR, "bot", bot_type)
        bot_files = get_json_files(bot_dir)

        print(f"{bot_type} 전체 파일 개수:", len(bot_files))

        sample_size = min(BOT_SAMPLE_SIZE, len(bot_files))
        sampled_files = random.sample(bot_files, sample_size)

        print(f"{bot_type} 샘플링 개수:", len(sampled_files))

        for file_path in sampled_files:
            data = load_json(file_path)

            data["user_id"] = f"user{user_count}"
            data["source_type"] = "bot"
            data["bot_type"] = bot_type
            data["original_file"] = os.path.basename(file_path)

            merged_data.append(data)
            user_count += 1

    # 3. 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print("병합 완료")
    print("최종 데이터 개수:", len(merged_data))
    print("저장 파일:", OUTPUT_FILE)


if __name__ == "__main__":
    merge_sampled_data()