#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agami / Sentient-CAPTCHA 마우스 trajectory → 모델 입력 feature 변환 어댑터

프론트 입력 예시
[
  {"x": 0.52, "y": 0.31, "t": 1777350550489},
  {"x": 0.51, "y": 0.32, "t": 1777350550539}
]

출력
{
  "static_features": {...},
  "dynamic_features": [{"dx": ..., "dy": ..., "dt": ...}, ...]
}

주의
- 학습 데이터의 최종 모델 입력은 raw x/y/t가 아니라 static_features + dynamic_features이다.
- 프론트가 x/y를 0~1 정규화로 보내면 coordinate_mode="normalized"로 두고 canvas_width/height를 넣어 pixel scale로 변환한다.
- 이미 pixel 좌표로 보내면 coordinate_mode="pixel"로 둔다.
"""

from __future__ import annotations

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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _angle(dx: float, dy: float) -> Optional[float]:
    if dx == 0 and dy == 0:
        return None
    return math.atan2(dy, dx)


def _angle_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    diff = abs(a - b)
    while diff > math.pi:
        diff = abs(diff - 2 * math.pi)
    return float(diff)


def normalize_or_pixel_points(
    trajectory: List[Dict[str, Any]],
    coordinate_mode: str = "normalized",
    canvas_width: Optional[float] = None,
    canvas_height: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    trajectory를 내부 계산용 pixel scale로 변환한다.

    coordinate_mode
    - "normalized": x/y가 0~1. canvas_width/height로 pixel 변환.
    - "pixel": x/y가 이미 pixel. 그대로 사용.
    """
    if coordinate_mode not in {"normalized", "pixel"}:
        raise ValueError("coordinate_mode must be 'normalized' or 'pixel'")

    if coordinate_mode == "normalized" and (canvas_width is None or canvas_height is None):
        raise ValueError("normalized 좌표를 사용할 때는 canvas_width와 canvas_height가 필요합니다.")

    points: List[Dict[str, float]] = []
    for p in trajectory:
        x = _to_float(p.get("x"))
        y = _to_float(p.get("y"))
        t = _to_float(p.get("t"))

        if coordinate_mode == "normalized":
            x *= float(canvas_width)
            y *= float(canvas_height)

        points.append({"x": x, "y": y, "t": t})

    # 시간 순서 보장
    points.sort(key=lambda p: p["t"])
    return points


def build_features_from_trajectory(
    trajectory: List[Dict[str, Any]],
    coordinate_mode: str = "normalized",
    canvas_width: Optional[float] = None,
    canvas_height: Optional[float] = None,
    direction_change_threshold_rad: float = 0.5,
    pause_velocity_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    raw trajectory를 모델 입력 sample 형태로 변환한다.
    """
    points = normalize_or_pixel_points(
        trajectory,
        coordinate_mode=coordinate_mode,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )

    if len(points) < 2:
        return {
            "static_features": {
                "duration": 0.0,
                "log_count": float(len(points)),
                "total_distance": 0.0,
                "straight_distance": 0.0,
                "distance_ratio": 0.0,
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "speed_std": 0.0,
                "direction_changes": 0.0,
                "pauses": 0.0,
            },
            "dynamic_features": [
                {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dt": 1.0,
                    "distance": 0.0,
                    "velocity": 0.0,
                    "acceleration": 0.0,
                    "angle_change": 0.0,
                }
            ],
        }

    dynamic_features: List[Dict[str, float]] = []
    distances: List[float] = []
    velocities: List[float] = []
    angle_changes: List[float] = []

    prev_velocity = 0.0
    prev_angle: Optional[float] = None

    for i in range(1, len(points)):
        prev = points[i - 1]
        cur = points[i]

        dx = cur["x"] - prev["x"]
        dy = cur["y"] - prev["y"]
        dt = cur["t"] - prev["t"]

        # dt가 0 이하이면 계산 안정성을 위해 1ms로 보정
        if dt <= 0:
            dt = 1.0

        distance = math.sqrt(dx * dx + dy * dy)
        velocity = distance / dt
        acceleration = (velocity - prev_velocity) / dt

        cur_angle = _angle(dx, dy)
        angle_change = _angle_diff(prev_angle, cur_angle)

        dynamic_features.append({
            "dx": float(dx),
            "dy": float(dy),
            "dt": float(dt),
            "distance": float(distance),
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "angle_change": float(angle_change),
        })

        distances.append(distance)
        velocities.append(velocity)
        angle_changes.append(angle_change)

        prev_velocity = velocity
        prev_angle = cur_angle

    first = points[0]
    last = points[-1]
    duration = max(last["t"] - first["t"], 1.0)
    total_distance = float(sum(distances))
    straight_distance = math.sqrt((last["x"] - first["x"]) ** 2 + (last["y"] - first["y"]) ** 2)
    distance_ratio = total_distance / max(straight_distance, 1e-6)

    avg_speed = float(sum(velocities) / max(len(velocities), 1))
    max_speed = float(max(velocities)) if velocities else 0.0
    speed_std = float(_std(velocities))
    direction_changes = int(sum(1 for a in angle_changes if a >= direction_change_threshold_rad))
    pauses = int(sum(1 for v in velocities if v <= pause_velocity_threshold))

    static_features = {
        "duration": float(duration),
        "log_count": float(len(points)),
        "total_distance": float(total_distance),
        "straight_distance": float(straight_distance),
        "distance_ratio": float(distance_ratio),
        "avg_speed": float(avg_speed),
        "max_speed": float(max_speed),
        "speed_std": float(speed_std),
        "direction_changes": float(direction_changes),
        "pauses": float(pauses),
    }

    return {
        "static_features": static_features,
        "dynamic_features": dynamic_features,
    }


def _std(values: List[float]) -> float:
    if len(values) == 0:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)
