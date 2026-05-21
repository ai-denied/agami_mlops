"""
손전등 캡챠 번들 정책 (1챌린지 = 3장)
=====================================
좌표 매칭 + 모델 위험도를 결합한 최종 판정.

규칙:
  - high_risk score (>= HIGH_RISK_THRESHOLD)가 3장 중 1번이라도 나오면 block
  - 좌표가 3장 중 MIN_COORD_HITS (2) 미만이면 block
  - 그 외 allow

기존 `three_attempt_policy`의 session-누적 정책을 폐기하고 단일 라운드 평가로 단순화.
HIGH_RISK_THRESHOLD는 모델 학습 시점의 정책 일관성을 위해
`app/ml/flashlight/onnx_export_info.json`에서 모듈 로드 시 1회 읽어온다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

_EXPORT_INFO_PATH = (
    Path(__file__).resolve().parent.parent
    / "ml"
    / "flashlight"
    / "onnx_export_info.json"
)

with open(_EXPORT_INFO_PATH, encoding="utf-8") as _f:
    _info = json.load(_f)

_threshold_policy = _info.get("threshold_policy", {})

HIGH_RISK_THRESHOLD: float = float(_threshold_policy.get("high_risk_threshold", 0.60))
MIN_COORD_HITS: int = 2


def evaluate_flashlight_decision(
    scores: list[float],
    coord_hits: list[bool],
) -> Literal["allow", "block"]:
    """3장 묶음 평가 결과로 allow/block을 반환.

    Args:
        scores: 그림 3장 각각의 bot_risk_score (FlashlightModel.predict 결과).
        coord_hits: 그림 3장 각각의 좌표 정답 여부.

    Returns:
        "allow" 또는 "block".
    """
    if any(s >= HIGH_RISK_THRESHOLD for s in scores):
        return "block"
    if sum(coord_hits) < MIN_COORD_HITS:
        return "block"
    return "allow"


__all__ = [
    "evaluate_flashlight_decision",
    "HIGH_RISK_THRESHOLD",
    "MIN_COORD_HITS",
]
