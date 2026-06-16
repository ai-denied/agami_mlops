"""
GRU 얼굴 활성도 ONNX 추론 어댑터.

ONNX 입력: x_seq (batch, 16, 20) float32 — raw 피처, 스케일러 내장
ONNX 출력: spoof_score (batch,) float32  — 0=live, 1=spoof

사용 예:
  detector = OnnxFaceLivenessDetector(
      onnx_path="model-store/facial_recognition/current/face_liveness.onnx",
      meta_path="model-store/facial_recognition/current/metadata.json",
  )
  result = detector.predict(x_seq_np)   # x_seq_np: (16, 20) float32
  print(result["spoof_score"], result["is_spoof"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort

SEQ_LEN     = 16
N_FEATURES  = 20


# ── 위험 등급 분류 ─────────────────────────────────────────────────────────────

def classify_spoof_risk(score: float, low_thr: float, high_thr: float) -> str:
    if score < low_thr:
        return "real_safe"
    if score < high_thr:
        return "suspicious"
    return "spoof_detected"


# ── 3회 시도 누적 정책 ────────────────────────────────────────────────────────

def apply_three_attempt_policy(
    scores: List[float],
    low_thr: float,
    high_thr: float,
    block_suspicious_count: int = 2,
    block_high_risk_count:  int = 1,
    block_total_score: Optional[float] = None,
) -> Dict[str, Any]:
    if not scores:
        raise ValueError("scores must not be empty")

    scores      = [float(s) for s in scores]
    total_score = float(np.sum(scores))
    avg_score   = float(np.mean(scores))
    max_score   = float(np.max(scores))

    suspicious_count = int(sum(s >= low_thr  for s in scores))
    high_risk_count  = int(sum(s >= high_thr for s in scores))

    if block_total_score is None:
        block_total_score = low_thr * 2.0

    if (
        high_risk_count  >= block_high_risk_count
        or suspicious_count >= block_suspicious_count
        or total_score       >= block_total_score
    ):
        decision = "block"
    elif suspicious_count >= 1:
        decision = "challenge_again"
    else:
        decision = "allow"

    final_decision = "block" if decision == "block" else "allow"

    return {
        "scores":              scores,
        "total_score":         round(total_score, 6),
        "avg_score":           round(avg_score, 6),
        "max_score":           round(max_score, 6),
        "suspicious_count":    suspicious_count,
        "high_risk_count":     high_risk_count,
        "low_spoof_threshold": float(low_thr),
        "high_spoof_threshold": float(high_thr),
        "internal_decision":   decision,
        "final_decision":      final_decision,
        "is_spoof":            final_decision == "block",
    }


# ── 메인 디텍터 ───────────────────────────────────────────────────────────────

class OnnxFaceLivenessDetector:
    """
    GRU 얼굴 활성도 ONNX 추론기.

    metadata.json 필드:
      threshold            (float) — 모델 학습 시 선택된 임계값
      high_spoof_threshold (float, 선택) — 미제공 시 threshold * 1.3
      three_attempt_policy (dict, 선택)
    """

    def __init__(
        self,
        onnx_path: str,
        meta_path: str,
        providers: Optional[List[str]] = None,
    ) -> None:
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.low_thr  = float(self.metadata.get("threshold", 0.5))
        self.high_thr = float(
            self.metadata.get(
                "high_spoof_threshold",
                min(self.low_thr * 1.3, 0.9),
            )
        )
        self._policy = self.metadata.get("three_attempt_policy", {})

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    # ── 단건 추론 ─────────────────────────────────────────────────────────────

    def predict(
        self,
        x_seq: np.ndarray,
        seq_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        x_seq      : (16, 20) 또는 (1, 16, 20) float32 — raw 얼굴 피처 시퀀스
        seq_length : 실제 유효 프레임 수 (없으면 16 사용)

        Returns
        -------
        dict with spoof_score, risk_band, is_spoof, ...
        """
        if x_seq.ndim == 2:
            x_seq = x_seq[np.newaxis, :, :]          # (1, 16, 20)
        if x_seq.shape[1] != SEQ_LEN or x_seq.shape[2] != N_FEATURES:
            raise ValueError(
                f"x_seq shape must be (batch, {SEQ_LEN}, {N_FEATURES}), got {x_seq.shape}"
            )

        x_seq = x_seq.astype(np.float32)
        raw   = self.session.run(["spoof_score"], {"x_seq": x_seq})
        score = float(raw[0][0])

        risk_band = classify_spoof_risk(score, self.low_thr, self.high_thr)
        return {
            "spoof_score":          round(score, 6),
            "risk_band":            risk_band,
            "is_spoof":             score >= self.low_thr,
            "low_spoof_threshold":  self.low_thr,
            "high_spoof_threshold": self.high_thr,
            "seq_length":           seq_length,
        }

    # ── 3회 시도 누적 ─────────────────────────────────────────────────────────

    def decide_three_attempts(self, scores: List[float]) -> Dict[str, Any]:
        return apply_three_attempt_policy(
            scores,
            low_thr=self.low_thr,
            high_thr=self.high_thr,
            block_suspicious_count=int(self._policy.get("block_suspicious_count", 2)),
            block_high_risk_count=int(self._policy.get("block_high_risk_count",  1)),
            block_total_score=float(
                self._policy.get("block_total_score", self.low_thr * 2.0)
            ),
        )
