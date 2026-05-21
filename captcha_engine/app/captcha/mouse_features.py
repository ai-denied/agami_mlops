"""
Mouse Trajectory Feature Adapter
=================================
팀원이 제공한 공식 feature extractor(`app/ml/flashlight/mouse_feature_extractor.py`)를
모델 추론 파이프라인에 맞게 호출하는 얇은 어댑터.

학습-추론 일관성 보장이 목적이므로 팀원 코드는 절대 수정하지 않고
sys.path를 통해 import하여 그대로 호출한다. dict 결과를 ONNX 입력 numpy
배열로만 변환한다.

임계값은 팀원 코드의 기본값과 동일하지만 명시적으로 선언해
어댑터 사용자에게 노출한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_ML_DIR = Path(__file__).resolve().parent.parent / "ml" / "flashlight"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from mouse_feature_extractor import (  # noqa: E402  (sys.path 조정 직후 import)
    SEQ_FEATURES,
    STATIC_FEATURES,
    extract_mouse_features,
)

DIRECTION_CHANGE_THRESHOLD_RAD = 0.35
PAUSE_THRESHOLD_MS = 500.0
MIN_DT_MS = 1.0


def extract_features_for_model(
    trajectory: Sequence[dict] | dict,
    timestamp_unit: str = "ms",
) -> tuple[np.ndarray, np.ndarray]:
    """raw 마우스 궤적 → ONNX 모델 입력 numpy 배열.

    Args:
        trajectory: 팀원 extractor가 지원하는 모든 형식 — `[{x,y,t}, ...]` 또는
            `{"events": [...]}`. key는 `x/clientX/pageX...`, `y/clientY/...`,
            `t/timeStamp/...` 자동 매핑.
        timestamp_unit: "ms"(기본) 또는 "s".

    Returns:
        dynamic_arr: shape `(N-1, 7)` float32, 순서는 `SEQ_FEATURES`.
        static_arr:  shape `(10,)`   float32, 순서는 `STATIC_FEATURES`.

    Raises:
        ValueError: 유효 이벤트가 2개 미만일 때 (팀원 코드의 검증을 그대로 전파).
    """
    result = extract_mouse_features(
        trajectory,
        timestamp_unit=timestamp_unit,
        min_dt_ms=MIN_DT_MS,
        pause_threshold_ms=PAUSE_THRESHOLD_MS,
        direction_change_threshold_rad=DIRECTION_CHANGE_THRESHOLD_RAD,
    )

    dynamic_arr = np.array(
        [[row[k] for k in SEQ_FEATURES] for row in result["dynamic_features"]],
        dtype=np.float32,
    )
    static_arr = np.array(
        [result["static_features"][k] for k in STATIC_FEATURES],
        dtype=np.float32,
    )
    return dynamic_arr, static_arr


__all__ = [
    "extract_features_for_model",
    "DIRECTION_CHANGE_THRESHOLD_RAD",
    "PAUSE_THRESHOLD_MS",
    "MIN_DT_MS",
    "SEQ_FEATURES",
    "STATIC_FEATURES",
]
