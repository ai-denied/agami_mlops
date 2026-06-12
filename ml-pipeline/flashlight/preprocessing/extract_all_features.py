import os
import json
import math
import pandas as pd

HUMAN_DIR = "mouse_logs/human"
BOT_DIR = "mouse_logs/bot"

OUTPUT_HUMAN_DIR = "processed/dynamic/human"
OUTPUT_BOT_DIR = "processed/dynamic/bot"


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_dynamic_features(mouse_logs):
    dynamic_features = []

    prev_velocity = 0
    prev_angle = None

    for i in range(1, len(mouse_logs)):
        prev = mouse_logs[i - 1]
        curr = mouse_logs[i]

        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        dt = curr["t"] - prev["t"]

        if dt <= 0:
            continue

        distance = math.sqrt(dx ** 2 + dy ** 2)
        velocity = distance / dt
        acceleration = (velocity - prev_velocity) / dt

        angle = math.atan2(dy, dx)

        if prev_angle is not None:
            angle_change = abs(angle - prev_angle)
            angle_change = min(angle_change, 2 * math.pi - angle_change)
        else:
            angle_change = 0

        dynamic_features.append({
            "dx": dx,
            "dy": dy,
            "dt": dt,
            "distance": distance,
            "velocity": velocity,
            "acceleration": acceleration,
            "angle_change": angle_change
        })

        prev_velocity = velocity
        prev_angle = angle

    return dynamic_features


def calculate_static_features(data, label):
    logs = data.get("mouse_logs", [])
    duration = data.get("duration", 0)
    click = data.get("click", None)

    if len(logs) < 5:
        return None

    if duration < 100 or duration > 20000:
        return None

    if not click:
        return None

    start = (logs[0]["x"], logs[0]["y"])
    end = (click["x"], click["y"])

    total_distance = 0
    speeds = []
    direction_changes = 0
    pauses = 0

    prev_angle = None

    for i in range(1, len(logs)):
        x1, y1, t1 = logs[i - 1]["x"], logs[i - 1]["y"], logs[i - 1]["t"]
        x2, y2, t2 = logs[i]["x"], logs[i]["y"], logs[i]["t"]

        dist = euclidean((x1, y1), (x2, y2))
        dt = max(t2 - t1, 1)

        total_distance += dist

        speed = dist / dt
        speeds.append(speed)

        angle = math.atan2(y2 - y1, x2 - x1)

        if prev_angle is not None:
            angle_change = abs(angle - prev_angle)
            angle_change = min(angle_change, 2 * math.pi - angle_change)

            if angle_change > 0.5:
                direction_changes += 1

        prev_angle = angle

        if dist < 1:
            pauses += 1

    straight_distance = euclidean(start, end)

    return {
        "duration": duration,
        "log_count": len(logs),
        "total_distance": total_distance,
        "straight_distance": straight_distance,
        "distance_ratio": total_distance / (straight_distance + 1e-6),
        "avg_speed": sum(speeds) / len(speeds),
        "max_speed": max(speeds),
        "speed_std": float(pd.Series(speeds).std()),
        "direction_changes": direction_changes,
        "pauses": pauses,
        "label": label
    }


def save_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_one_file(input_path, output_path, label, source_type, bot_type=None):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"읽기 실패: {input_path} / {e}")
        return False

    logs = data.get("mouse_logs", [])

    static_features = calculate_static_features(data, label)

    if static_features is None:
        return False

    dynamic_features = calculate_dynamic_features(logs)

    result = {
        "source_file": os.path.basename(input_path),
        "source_type": source_type,
        "bot_type": bot_type,
        "label": label,

        "image_id": data.get("image_id"),
        "target_object": data.get("target_object"),
        "click": data.get("click"),

        "static_features": static_features,
        "dynamic_features": dynamic_features
    }

    save_json(result, output_path)
    return True


def process_human_logs():
    count = 0

    for file in os.listdir(HUMAN_DIR):
        input_path = os.path.join(HUMAN_DIR, file)

        if not file.endswith(".json") or not os.path.isfile(input_path):
            continue

        output_path = os.path.join(OUTPUT_HUMAN_DIR, file)

        success = process_one_file(
            input_path=input_path,
            output_path=output_path,
            label=0,
            source_type="human"
        )

        if success:
            count += 1

    return count


def process_bot_logs():
    count = 0

    for bot_type in os.listdir(BOT_DIR):
        bot_type_path = os.path.join(BOT_DIR, bot_type)

        if not os.path.isdir(bot_type_path):
            continue

        for file in os.listdir(bot_type_path):
            input_path = os.path.join(bot_type_path, file)

            if not file.endswith(".json") or not os.path.isfile(input_path):
                continue

            output_path = os.path.join(OUTPUT_BOT_DIR, bot_type, file)

            success = process_one_file(
                input_path=input_path,
                output_path=output_path,
                label=1,
                source_type="bot",
                bot_type=bot_type
            )

            if success:
                count += 1

    return count


if __name__ == "__main__":
    human_count = process_human_logs()
    bot_count = process_bot_logs()

    print("정적 + 동적 피처 JSON 저장 완료")
    print("human 처리 개수:", human_count)
    print("bot 처리 개수:", bot_count)
    print("저장 위치: processed/dynamic/")