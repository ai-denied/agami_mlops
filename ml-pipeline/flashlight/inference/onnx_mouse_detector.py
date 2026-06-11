#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agami / Sentient-CAPTCHA ONNX Runtime 추론 어댑터

필수 파일
- mouse_gru_server_final_v2.onnx
- mouse_normalizer_server_final_v2.joblib
- mouse_metadata_server_final_v2.json

역할
- static/dynamic features를 학습 당시 scaler로 정규화
- ONNX Runtime으로 bot_risk_score 추론
- low/high risk band 계산
- 3회 누적 정책 적용
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import onnxruntime as ort

from flashlight.inference.mouse_feature_adapter import build_features_from_trajectory

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


def classify_single_attempt_risk(bot_risk_score: float, low_risk_threshold: float, high_risk_threshold: float) -> str:
    if bot_risk_score < low_risk_threshold:
        return "low_risk"
    if bot_risk_score < high_risk_threshold:
        return "suspicious"
    return "high_risk"


def apply_three_attempt_policy(
    scores: List[float],
    low_risk_threshold: float,
    high_risk_threshold: float,
    block_suspicious_count: int = 2,
    block_high_risk_count: int = 1,
    block_total_score: Optional[float] = None,
) -> Dict[str, Any]:
    if len(scores) == 0:
        raise ValueError("scores must not be empty")

    scores = [float(s) for s in scores]
    total_score = float(np.sum(scores))
    avg_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))

    suspicious_count = int(sum(s >= low_risk_threshold for s in scores))
    high_risk_count = int(sum(s >= high_risk_threshold for s in scores))

    if block_total_score is None:
        block_total_score = low_risk_threshold * 2.0

    if (
        high_risk_count >= block_high_risk_count
        or suspicious_count >= block_suspicious_count
        or total_score >= block_total_score
    ):
        internal_decision = "block"
    elif suspicious_count >= 1:
        internal_decision = "challenge_again"
    else:
        internal_decision = "allow"

    # MVP 최종 엔진 로직: challenge_again은 조건부 통과로 처리
    final_decision = "block" if internal_decision == "block" else "allow"
    is_bot = final_decision == "block"

    return {
        "scores": scores,
        "total_score": round(total_score, 6),
        "avg_score": round(avg_score, 6),
        "max_score": round(max_score, 6),
        "min_score": round(min_score, 6),
        "suspicious_count": suspicious_count,
        "high_risk_count": high_risk_count,
        "low_risk_threshold": float(low_risk_threshold),
        "high_risk_threshold": float(high_risk_threshold),
        "block_total_score": float(block_total_score),
        "internal_decision": internal_decision,
        "final_decision": final_decision,
        "is_bot": is_bot,
    }


class OnnxMouseBotRiskDetector:
    def __init__(
        self,
        onnx_path: str,
        normalizer_path: str,
        metadata_path: str,
        providers: Optional[List[str]] = None,
    ):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.normalizer = joblib.load(normalizer_path)
        self.low_risk_threshold = float(self.metadata.get("low_risk_threshold", 0.05))
        self.high_risk_threshold = float(self.metadata.get("high_risk_threshold", 0.65))
        self.policy = self.metadata.get("three_attempt_policy", {})

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)

    def _transform_sample(self, sample: Dict[str, Any]):
        seq = []
        for feat in sample.get("dynamic_features", []):
            seq.append([float(feat.get(k, 0.0)) for k in SEQ_FEATURES])
        if len(seq) == 0:
            seq = [[0.0] * len(SEQ_FEATURES)]
        seq_arr = np.asarray(seq, dtype=np.float32)

        stat = sample.get("static_features", {})
        static_arr = np.asarray([
            float(stat.get("duration", 0.0)),
            float(stat.get("log_count", 0.0)),
            float(stat.get("total_distance", 0.0)),
            float(stat.get("straight_distance", 0.0)),
            float(stat.get("distance_ratio", 0.0)),
            float(stat.get("avg_speed", 0.0)),
            float(stat.get("max_speed", 0.0)),
            float(stat.get("speed_std", 0.0)),
            float(stat.get("direction_changes", 0.0)),
            float(stat.get("pauses", 0.0)),
        ], dtype=np.float32)

        seq_scaled = self.normalizer.seq_scaler.transform(seq_arr).astype(np.float32)
        static_scaled = self.normalizer.static_scaler.transform(static_arr.reshape(1, -1)).astype(np.float32).reshape(-1)

        x_seq = seq_scaled[None, :, :].astype(np.float32)
        lengths = np.asarray([seq_scaled.shape[0]], dtype=np.int64)
        x_static = static_scaled[None, :].astype(np.float32)
        return x_seq, lengths, x_static

    def predict_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        x_seq, lengths, x_static = self._transform_sample(sample)
        outputs = self.session.run(
            ["bot_risk_score"],
            {
                "x_seq": x_seq,
                "lengths": lengths,
                "x_static": x_static,
            },
        )
        bot_risk_score = float(outputs[0][0])
        risk_band = classify_single_attempt_risk(
            bot_risk_score,
            self.low_risk_threshold,
            self.high_risk_threshold,
        )
        return {
            "bot_risk_score": bot_risk_score,
            "risk_band": risk_band,
            "low_risk_threshold": self.low_risk_threshold,
            "high_risk_threshold": self.high_risk_threshold,
        }

    def predict_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        coordinate_mode: str = "normalized",
        canvas_width: Optional[float] = None,
        canvas_height: Optional[float] = None,
    ) -> Dict[str, Any]:
        sample = build_features_from_trajectory(
            trajectory,
            coordinate_mode=coordinate_mode,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        result = self.predict_features(sample)
        result["feature_sample"] = sample
        return result

    def decide_three_attempts(self, scores: List[float]) -> Dict[str, Any]:
        return apply_three_attempt_policy(
            scores,
            low_risk_threshold=self.low_risk_threshold,
            high_risk_threshold=self.high_risk_threshold,
            block_suspicious_count=int(self.policy.get("block_suspicious_count", 2)),
            block_high_risk_count=int(self.policy.get("block_high_risk_count", 1)),
            block_total_score=float(self.policy.get("block_total_score", 0.25)),
        )


class OnnxMouseBotRiskDetectorJson:
    """
    captcha_engine JSON normalizer 포맷 기반 추론기.

    joblib 없이 JSON 파라미터만으로 정규화를 수행한다.
    captcha_engine(Go/Node 등)의 로딩 동작을 Python에서 시뮬레이션할 때 사용한다.

    필수 파일:
    - mouse_gru_server_final_v3_policy_tuned.onnx
    - mouse_normalizer_params_v3_policy_tuned.json
    """

    def __init__(
        self,
        onnx_path: str,
        normalizer_json_path: str,
        providers: Optional[List[str]] = None,
    ):
        with open(normalizer_json_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self._seq_mean = np.array(params["seq_scaler"]["mean"], dtype=np.float32)
        self._seq_scale = np.array(params["seq_scaler"]["scale"], dtype=np.float32)
        self._static_mean = np.array(params["static_scaler"]["mean"], dtype=np.float32)
        self._static_scale = np.array(params["static_scaler"]["scale"], dtype=np.float32)

        policy = params["threshold_policy"]
        self.low_risk_threshold = float(policy["low_risk_threshold"])
        self.high_risk_threshold = float(policy["high_risk_threshold"])
        self._policy = policy

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    def _transform_sample(self, sample: Dict[str, Any]):
        seq = []
        for feat in sample.get("dynamic_features", []):
            seq.append([float(feat.get(k, 0.0)) for k in SEQ_FEATURES])
        if not seq:
            seq = [[0.0] * len(SEQ_FEATURES)]
        seq_arr = np.asarray(seq, dtype=np.float32)

        stat = sample.get("static_features", {})
        static_arr = np.asarray([
            float(stat.get("duration", 0.0)),
            float(stat.get("log_count", 0.0)),
            float(stat.get("total_distance", 0.0)),
            float(stat.get("straight_distance", 0.0)),
            float(stat.get("distance_ratio", 0.0)),
            float(stat.get("avg_speed", 0.0)),
            float(stat.get("max_speed", 0.0)),
            float(stat.get("speed_std", 0.0)),
            float(stat.get("direction_changes", 0.0)),
            float(stat.get("pauses", 0.0)),
        ], dtype=np.float32)

        seq_scaled = ((seq_arr - self._seq_mean) / self._seq_scale).astype(np.float32)
        static_scaled = ((static_arr - self._static_mean) / self._static_scale).astype(np.float32)

        x_seq = seq_scaled[None, :, :]
        lengths = np.asarray([seq_scaled.shape[0]], dtype=np.int64)
        x_static = static_scaled[None, :]
        return x_seq, lengths, x_static

    def predict_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        x_seq, lengths, x_static = self._transform_sample(sample)
        outputs = self.session.run(
            ["bot_risk_score"],
            {"x_seq": x_seq, "lengths": lengths, "x_static": x_static},
        )
        bot_risk_score = float(outputs[0][0])
        risk_band = classify_single_attempt_risk(
            bot_risk_score,
            self.low_risk_threshold,
            self.high_risk_threshold,
        )
        return {
            "bot_risk_score": bot_risk_score,
            "risk_band": risk_band,
            "low_risk_threshold": self.low_risk_threshold,
            "high_risk_threshold": self.high_risk_threshold,
        }

    def predict_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        coordinate_mode: str = "normalized",
        canvas_width: Optional[float] = None,
        canvas_height: Optional[float] = None,
    ) -> Dict[str, Any]:
        from flashlight.inference.mouse_feature_adapter import build_features_from_trajectory
        sample = build_features_from_trajectory(
            trajectory,
            coordinate_mode=coordinate_mode,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        result = self.predict_features(sample)
        result["feature_sample"] = sample
        return result

    def decide_three_attempts(self, scores: List[float]) -> Dict[str, Any]:
        return apply_three_attempt_policy(
            scores,
            low_risk_threshold=self.low_risk_threshold,
            high_risk_threshold=self.high_risk_threshold,
            block_suspicious_count=int(self._policy.get("block_suspicious_count", 2)),
            block_high_risk_count=int(self._policy.get("block_high_risk_count", 1)),
            block_total_score=float(self._policy.get("block_total_score", 0.25)),
        )
