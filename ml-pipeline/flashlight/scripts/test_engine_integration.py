#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
엔진 연동 예시

1회 attempt마다 predict_trajectory()로 bot_risk_score 산출
3회 점수가 쌓이면 decide_three_attempts()로 최종 allow/block 판단

모델 경로: model-store/flashlight/current/
  mouse_gru.onnx    — ONNX 추론 모델
  normalizer.json   — 정규화 파라미터 + threshold_policy
  metadata.json     — 버전 및 성능 정보
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from flashlight.inference.onnx_mouse_detector import OnnxMouseBotRiskDetectorJson  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CURRENT = os.path.join(_ML_PIPELINE_ROOT, "model-store", "flashlight", "current")

ONNX_PATH       = os.path.join(_CURRENT, "mouse_gru.onnx")
NORMALIZER_PATH = os.path.join(_CURRENT, "normalizer.json")
METADATA_PATH   = os.path.join(_CURRENT, "metadata.json")


def main():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[model-store] version={meta.get('version')}  promoted_at={meta.get('promoted_at')}")

    detector = OnnxMouseBotRiskDetectorJson(
        onnx_path=ONNX_PATH,
        normalizer_json_path=NORMALIZER_PATH,
    )

    # 프론트에서 들어오는 trajectory 예시: x/y는 0~1 정규화, t는 Date.now() ms
    attempt_trajectories = [
        [
            {"x": 0.80, "y": 0.20, "t": 1777350550000},
            {"x": 0.76, "y": 0.23, "t": 1777350550050},
            {"x": 0.70, "y": 0.30, "t": 1777350550100},
        ],
        [
            {"x": 0.81, "y": 0.21, "t": 1777350560000},
            {"x": 0.75, "y": 0.24, "t": 1777350560050},
            {"x": 0.69, "y": 0.33, "t": 1777350560100},
        ],
        [
            {"x": 0.79, "y": 0.20, "t": 1777350570000},
            {"x": 0.72, "y": 0.24, "t": 1777350570050},
            {"x": 0.66, "y": 0.35, "t": 1777350570100},
        ],
    ]

    scores = []
    attempts = []

    for idx, trajectory in enumerate(attempt_trajectories, start=1):
        pred = detector.predict_trajectory(
            trajectory,
            coordinate_mode="normalized",
            canvas_width=600,
            canvas_height=400,
        )
        scores.append(pred["bot_risk_score"])
        attempts.append({
            "attempt_index": idx,
            "bot_risk_score": pred["bot_risk_score"],
            "risk_band": pred["risk_band"],
        })

    policy_result = detector.decide_three_attempts(scores)

    response = {
        "attempts": attempts,
        "policy": policy_result,
        "decision": policy_result["final_decision"],
        "is_bot": policy_result["is_bot"],
    }

    print(response)


if __name__ == "__main__":
    main()
