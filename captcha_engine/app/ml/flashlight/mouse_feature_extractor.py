#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mouse_feature_extractor.py

Sentient-CAPTCHA 손전등 CAPTCHA 모델 입력 피처 추출 코드

역할:
- 브라우저/엔진에서 수집한 raw mouse event 로그를
  모델 입력 형식인 dynamic_features / static_features로 변환한다.
- 변환된 결과는 mouse_onnx_inference.py 또는 mouse_inference.py에 바로 넣을 수 있다.

중요:
- 현재 학습된 모델은 시간 단위를 ms 기준으로 학습했다.
- 따라서 dt, duration은 기본적으로 millisecond(ms) 단위로 산출한다.
- velocity는 px/ms, acceleration은 px/ms^2 기준이다.

입력 raw event 예시:
[
  {"x": 100, "y": 200, "t": 0},
  {"x": 105, "y": 203, "t": 16},
  {"x": 110, "y": 208, "t": 32}
]

또는:
{
  "events": [
    {"clientX": 100, "clientY": 200, "timeStamp": 0},
    {"clientX": 105, "clientY": 203, "timeStamp": 16}
  ]
}

출력 예시:
{
  "dynamic_features": [
    {
      "dx": 5.0,
      "dy": 3.0,
      "dt": 16.0,
      "distance": 5.8309,
      "velocity": 0.3644,
      "acceleration": 0.0,
      "angle_change": 0.0
    }
  ],
  "static_features": {
    "duration": 32.0,
    "log_count": 3,
    "total_distance": 12.9019,
    "straight_distance": 12.8062,
    "distance_ratio": 1.0075,
    "avg_speed": 0.4032,
    "max_speed": 0.4420,
    "speed_std": 0.0388,
    "direction_changes": 0,
    "pauses": 0
  }
}

사용 예시:
python mouse_feature_extractor.py \
  --raw "./raw_mouse_log.json" \
  --out "./sample_captcha_log.json"

timestamp가 초 단위라면:
python mouse_feature_extractor.py \
  --raw "./raw_mouse_log.json" \
  --out "./sample_captcha_log.json" \
  --timestamp-unit s
"""

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Tuple


SEQ_FEATURES = [
    "dx",
    "dy",
    "dt",
    "distance",
    "velocity",
    "acceleration",
    "angle_change",
]

STATIC_FEATURES = [
    "duration",
    "log_count",
    "total_distance",
    "straight_distance",
    "distance_ratio",
    "avg_speed",
    "max_speed",
    "speed_std",
    "direction_changes",
    "pauses",
]


X_KEYS = ["x", "clientX", "pageX", "screenX", "offsetX"]
Y_KEYS = ["y", "clientY", "pageY", "screenY", "offsetY"]
T_KEYS = ["t", "time", "timestamp", "timeStamp", "ts", "elapsed", "clientTime"]


def _first_existing(event: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in event and event[key] is not None:
            try:
                return float(event[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_event_list(raw: Any) -> List[Dict[str, Any]]:
    """
    raw 입력에서 실제 이벤트 리스트를 꺼낸다.
    지원 형식:
    1. [event, event, ...]
    2. {"events": [...]}
    3. {"mouse_events": [...]}
    4. {"logs": [...]}
    5. {"data": [...]}
    """

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ["events", "mouse_events", "mouseEvents", "logs", "data", "raw_events"]:
            if key in raw and isinstance(raw[key], list):
                return raw[key]

    raise ValueError(
        "raw mouse log 형식을 인식할 수 없습니다. "
        "리스트 또는 {'events': [...]} 형태로 전달해 주세요."
    )


def normalize_timestamp_to_ms(t: float, timestamp_unit: str = "ms") -> float:
    """
    timestamp를 millisecond 기준으로 변환한다.

    timestamp_unit:
    - "ms": 이미 millisecond 기준
    - "s": second 기준이므로 1000을 곱함
    """

    if timestamp_unit == "ms":
        return float(t)

    if timestamp_unit == "s":
        return float(t) * 1000.0

    raise ValueError("timestamp_unit은 'ms' 또는 's'만 지원합니다.")


def parse_raw_events(
    raw: Any,
    timestamp_unit: str = "ms",
    keep_event_types: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """
    다양한 raw event 형태에서 x, y, t_ms만 추출한다.

    keep_event_types:
    - None이면 x/y/t가 있는 이벤트를 모두 사용한다.
    - 예: ["mousemove", "pointermove"]를 넣으면 해당 type만 사용한다.
    """

    raw_events = _extract_event_list(raw)
    parsed = []

    for event in raw_events:
        if not isinstance(event, dict):
            continue

        if keep_event_types is not None:
            event_type = event.get("type")
            if event_type not in keep_event_types:
                continue

        x = _first_existing(event, X_KEYS)
        y = _first_existing(event, Y_KEYS)
        t = _first_existing(event, T_KEYS)

        if x is None or y is None or t is None:
            continue

        parsed.append({
            "x": float(x),
            "y": float(y),
            "t": normalize_timestamp_to_ms(float(t), timestamp_unit=timestamp_unit),
        })

    if len(parsed) < 2:
        raise ValueError(
            f"유효한 mouse event가 {len(parsed)}개입니다. "
            "최소 2개 이상의 x/y/t 이벤트가 필요합니다."
        )

    parsed.sort(key=lambda e: e["t"])
    return parsed


def _wrap_angle_rad(angle: float) -> float:
    """
    angle 차이를 -pi ~ pi 범위로 정규화한다.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def extract_mouse_features(
    raw: Any,
    timestamp_unit: str = "ms",
    min_dt_ms: float = 1.0,
    pause_threshold_ms: float = 500.0,
    direction_change_threshold_rad: float = 0.35,
    min_move_px: float = 1e-6,
    keep_event_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    raw mouse event 로그를 모델 입력 피처로 변환한다.

    반환 구조:
    {
      "dynamic_features": [...],
      "static_features": {...}
    }

    피처 단위:
    - dx, dy, distance: px
    - dt, duration: ms
    - velocity, avg_speed, max_speed, speed_std: px/ms
    - acceleration: px/ms^2
    - angle_change: radian
    """

    events = parse_raw_events(
        raw,
        timestamp_unit=timestamp_unit,
        keep_event_types=keep_event_types,
    )

    dynamic_features: List[Dict[str, float]] = []

    prev_velocity = 0.0
    prev_angle: Optional[float] = None

    total_distance = 0.0
    velocities: List[float] = []
    angle_changes: List[float] = []
    pauses = 0

    for i in range(1, len(events)):
        prev = events[i - 1]
        cur = events[i]

        dx = cur["x"] - prev["x"]
        dy = cur["y"] - prev["y"]

        raw_dt = cur["t"] - prev["t"]

        # 같은 timestamp가 찍히거나 역순이 섞인 경우를 방어한다.
        # 모델은 dt=0을 처리하기 어렵기 때문에 최소 dt를 둔다.
        dt = max(float(raw_dt), float(min_dt_ms))

        distance = math.hypot(dx, dy)
        velocity = distance / dt

        if i == 1:
            acceleration = 0.0
        else:
            acceleration = (velocity - prev_velocity) / dt

        if distance <= min_move_px:
            angle = prev_angle
            angle_change = 0.0
        else:
            angle = math.atan2(dy, dx)
            if prev_angle is None:
                angle_change = 0.0
            else:
                angle_change = abs(_wrap_angle_rad(angle - prev_angle))

        if angle is not None:
            prev_angle = angle

        if dt >= pause_threshold_ms:
            pauses += 1

        if angle_change >= direction_change_threshold_rad:
            angle_changes.append(angle_change)

        total_distance += distance
        velocities.append(velocity)
        prev_velocity = velocity

        dynamic_features.append({
            "dx": float(dx),
            "dy": float(dy),
            "dt": float(dt),
            "distance": float(distance),
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "angle_change": float(angle_change),
        })

    first = events[0]
    last = events[-1]

    duration = max(float(last["t"] - first["t"]), float(min_dt_ms))
    log_count = len(events)
    straight_distance = math.hypot(last["x"] - first["x"], last["y"] - first["y"])

    if straight_distance <= 1e-6:
        distance_ratio = total_distance / 1e-6
    else:
        distance_ratio = total_distance / straight_distance

    if velocities:
        avg_speed = sum(velocities) / len(velocities)
        max_speed = max(velocities)
        mean_speed = avg_speed
        speed_std = math.sqrt(sum((v - mean_speed) ** 2 for v in velocities) / len(velocities))
    else:
        avg_speed = 0.0
        max_speed = 0.0
        speed_std = 0.0

    static_features = {
        "duration": float(duration),
        "log_count": float(log_count),
        "total_distance": float(total_distance),
        "straight_distance": float(straight_distance),
        "distance_ratio": float(distance_ratio),
        "avg_speed": float(avg_speed),
        "max_speed": float(max_speed),
        "speed_std": float(speed_std),
        "direction_changes": int(len(angle_changes)),
        "pauses": int(pauses),
    }

    return {
        "dynamic_features": dynamic_features,
        "static_features": static_features,
    }


def validate_model_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    모델 입력에 필요한 필드가 모두 있는지 검증한다.
    """

    if "dynamic_features" not in sample or not isinstance(sample["dynamic_features"], list):
        raise ValueError("sample에는 dynamic_features 리스트가 필요합니다.")

    if "static_features" not in sample or not isinstance(sample["static_features"], dict):
        raise ValueError("sample에는 static_features 객체가 필요합니다.")

    missing_dynamic = []
    for idx, row in enumerate(sample["dynamic_features"]):
        for key in SEQ_FEATURES:
            if key not in row:
                missing_dynamic.append((idx, key))

    missing_static = [key for key in STATIC_FEATURES if key not in sample["static_features"]]

    return {
        "valid": len(missing_dynamic) == 0 and len(missing_static) == 0,
        "seq_len": len(sample["dynamic_features"]),
        "gru_input_shape": [1, len(sample["dynamic_features"]), len(SEQ_FEATURES)],
        "static_input_shape": [1, len(STATIC_FEATURES)],
        "missing_dynamic": missing_dynamic,
        "missing_static": missing_static,
    }


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract mouse features for Sentient-CAPTCHA model")

    parser.add_argument("--raw", required=True, help="raw mouse log JSON path")
    parser.add_argument("--out", required=True, help="output model sample JSON path")
    parser.add_argument(
        "--timestamp-unit",
        default="ms",
        choices=["ms", "s"],
        help="raw timestamp unit. 현재 학습 모델은 ms 기준을 권장합니다.",
    )
    parser.add_argument("--min-dt-ms", type=float, default=1.0)
    parser.add_argument("--pause-threshold-ms", type=float, default=500.0)
    parser.add_argument("--direction-change-threshold-rad", type=float, default=0.35)
    parser.add_argument(
        "--move-only",
        action="store_true",
        help="type이 mousemove/pointermove인 이벤트만 사용합니다.",
    )

    args = parser.parse_args()

    raw = load_json(args.raw)

    keep_event_types = None
    if args.move_only:
        keep_event_types = ["mousemove", "pointermove"]

    sample = extract_mouse_features(
        raw,
        timestamp_unit=args.timestamp_unit,
        min_dt_ms=args.min_dt_ms,
        pause_threshold_ms=args.pause_threshold_ms,
        direction_change_threshold_rad=args.direction_change_threshold_rad,
        keep_event_types=keep_event_types,
    )

    validation = validate_model_sample(sample)

    save_json(sample, args.out)

    print("Feature extraction complete")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
