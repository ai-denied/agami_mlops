import os
import json
import math
import pandas as pd

HUMAN_DIR = "mouse_logs/human"
BOT_DIR = "mouse_logs/bot"


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def extract_features(file_path, label):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return None

    logs = data.get("mouse_logs", [])

    if len(logs) < 5:
        return None

    duration = data.get("duration", 0)
    click = data.get("click", None)

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
        x1, y1, t1 = logs[i-1]["x"], logs[i-1]["y"], logs[i-1]["t"]
        x2, y2, t2 = logs[i]["x"], logs[i]["y"], logs[i]["t"]

        dist = euclidean((x1, y1), (x2, y2))
        dt = max(t2 - t1, 1)

        total_distance += dist

        speed = dist / dt
        speeds.append(speed)

        angle = math.atan2(y2 - y1, x2 - x1)
        if prev_angle is not None:
            if abs(angle - prev_angle) > 0.5:
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
        "speed_std": pd.Series(speeds).std(),
        "direction_changes": direction_changes,
        "pauses": pauses,
        "label": label
    }


def load_all_data():
    rows = []

    # human
    for file in os.listdir(HUMAN_DIR):
        path = os.path.join(HUMAN_DIR, file)

        if not file.endswith(".json") or not os.path.isfile(path):
            continue

        feat = extract_features(path, 0)
        if feat:
            rows.append(feat)

    # bot
    for bot_type in os.listdir(BOT_DIR):
        bot_path = os.path.join(BOT_DIR, bot_type)

        if not os.path.isdir(bot_path):
            continue

        for file in os.listdir(bot_path):
            path = os.path.join(bot_path, file)

            if not file.endswith(".json") or not os.path.isfile(path):
                continue

            feat = extract_features(path, 1)
            if feat:
                rows.append(feat)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_all_data()

    print("최종 데이터 개수:", len(df))

    df.to_csv("features.csv", index=False)
    print("features.csv 생성 완료")