
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Anti-Spoofing ONNX Runtime 추론 어댑터

필수 파일:
  face_antispoofing.onnx        - ONNX 모델 (input: face_image, output: spoof_score)
  face_metadata.json            - 학습 메타데이터 (spoof_threshold, img_size 등)

사용 예:
  detector = OnnxFaceAntiSpoofingDetector(
      onnx_path="runs/v1/face_antispoofing.onnx",
      metadata_path="runs/v1/face_metadata.json",
  )

  # numpy RGB 배열 (H, W, 3) uint8
  result = detector.predict_image(face_crop_rgb)

  # 이미지 파일 경로
  result = detector.predict_path("/tmp/face.jpg")

  # 3회 시도 누적 → 최종 위변조 여부
  decision = detector.decide_three_attempts([0.12, 0.85, 0.91])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── 전처리 유틸 ──────────────────────────────────────────────────────────────

def preprocess_image(
    img_rgb: np.ndarray,
    img_size: int = 224,
) -> np.ndarray:
    """
    numpy RGB uint8 (H, W, 3) → ONNX 입력 float32 (1, 3, H, W)
    ImageNet 정규화 적용.
    """
    resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)           # HWC → CHW
    return arr[None, :, :, :].copy()       # (1, 3, H, W)


def _load_image_as_rgb(path: Union[str, Path]) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── 위험 등급 분류 ─────────────────────────────────────────────────────────────

def classify_spoof_risk(
    spoof_score: float,
    low_spoof_threshold: float,
    high_spoof_threshold: float,
) -> str:
    """
    spoof_score: sigmoid(logit), 1에 가까울수록 위변조 의심.
    low_spoof_threshold  미만 → real_safe
    low_spoof_threshold  이상 → suspicious
    high_spoof_threshold 이상 → spoof_detected
    """
    if spoof_score < low_spoof_threshold:
        return "real_safe"
    if spoof_score < high_spoof_threshold:
        return "suspicious"
    return "spoof_detected"


# ── 3회 시도 정책 (마우스 디텍터와 동일 인터페이스) ─────────────────────────────

def apply_face_three_attempt_policy(
    scores: List[float],
    low_spoof_threshold: float,
    high_spoof_threshold: float,
    block_suspicious_count: int = 2,
    block_high_risk_count: int = 1,
    block_total_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    scores: 최대 3회 시도의 spoof_score 리스트.

    결정 로직:
    - spoof_detected(high_risk)가 1회 이상 → block
    - suspicious가 2회 이상        → block
    - total_score ≥ block_total_score → block
    - suspicious 1회               → challenge_again (→ allow)
    - 그 외                        → allow
    """
    if not scores:
        raise ValueError("scores must not be empty")

    scores       = [float(s) for s in scores]
    total_score  = float(np.sum(scores))
    avg_score    = float(np.mean(scores))
    max_score    = float(np.max(scores))

    suspicious_count = int(sum(s >= low_spoof_threshold  for s in scores))
    high_risk_count  = int(sum(s >= high_spoof_threshold for s in scores))

    if block_total_score is None:
        block_total_score = low_spoof_threshold * 2.0

    if (
        high_risk_count  >= block_high_risk_count
        or suspicious_count >= block_suspicious_count
        or total_score       >= block_total_score
    ):
        internal_decision = "block"
    elif suspicious_count >= 1:
        internal_decision = "challenge_again"
    else:
        internal_decision = "allow"

    final_decision = "block" if internal_decision == "block" else "allow"

    return {
        "scores":                scores,
        "total_score":           round(total_score, 6),
        "avg_score":             round(avg_score, 6),
        "max_score":             round(max_score, 6),
        "suspicious_count":      suspicious_count,
        "high_risk_count":       high_risk_count,
        "low_spoof_threshold":   float(low_spoof_threshold),
        "high_spoof_threshold":  float(high_spoof_threshold),
        "block_total_score":     float(block_total_score),
        "internal_decision":     internal_decision,
        "final_decision":        final_decision,
        "is_spoof":              final_decision == "block",
    }


# ── 메인 디텍터 클래스 ─────────────────────────────────────────────────────────

class OnnxFaceAntiSpoofingDetector:
    """
    ONNX Runtime 기반 얼굴 위변조 탐지기.

    metadata.json 필드:
      img_size           (int)   — 입력 이미지 크기, 기본 224
      spoof_threshold    (float) — 학습 중 선택된 threshold (= low_spoof_threshold)
      high_spoof_threshold (float, optional) — 미제공 시 spoof_threshold * 1.3
    """

    def __init__(
        self,
        onnx_path: str,
        metadata_path: str,
        providers: Optional[List[str]] = None,
    ):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.img_size             = int(self.metadata.get("img_size", 224))
        self.low_spoof_threshold  = float(self.metadata.get("spoof_threshold", 0.5))
        self.high_spoof_threshold = float(
            self.metadata.get(
                "high_spoof_threshold",
                min(self.low_spoof_threshold * 1.3, 0.9),
            )
        )
        self._policy = self.metadata.get("three_attempt_policy", {})

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    # ── 핵심 추론 ─────────────────────────────────────────────────────────────

    def predict_image(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """
        numpy RGB uint8 (H, W, 3) 입력 → 위변조 점수와 판정 반환.

        Returns:
          spoof_score   : 0~1 float. 1에 가까울수록 spoof.
          risk_band     : "real_safe" | "suspicious" | "spoof_detected"
          is_spoof      : bool (score ≥ low_spoof_threshold)
        """
        inp = preprocess_image(img_rgb, self.img_size)
        raw = self.session.run(["spoof_score"], {"face_image": inp})
        logit = float(raw[0][0])
        spoof_score = float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid

        risk_band = classify_spoof_risk(
            spoof_score,
            self.low_spoof_threshold,
            self.high_spoof_threshold,
        )
        return {
            "spoof_score":          round(spoof_score, 6),
            "logit":                round(logit, 6),
            "risk_band":            risk_band,
            "is_spoof":             spoof_score >= self.low_spoof_threshold,
            "low_spoof_threshold":  self.low_spoof_threshold,
            "high_spoof_threshold": self.high_spoof_threshold,
        }

    def predict_path(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """이미지 파일 경로를 받아 추론한다."""
        img = _load_image_as_rgb(image_path)
        if img is None:
            return {
                "error":        "image_load_failed",
                "image_path":   str(image_path),
                "spoof_score":  None,
                "risk_band":    None,
                "is_spoof":     None,
            }
        result = self.predict_image(img)
        result["image_path"] = str(image_path)
        return result

    # ── 3회 시도 누적 정책 ──────────────────────────────────────────────────────

    def decide_three_attempts(self, scores: List[float]) -> Dict[str, Any]:
        """
        최대 3회 시도의 spoof_score 리스트를 받아 최종 차단 여부를 반환한다.

        반환값 is_spoof=True → 위변조 차단.
        """
        return apply_face_three_attempt_policy(
            scores,
            low_spoof_threshold=self.low_spoof_threshold,
            high_spoof_threshold=self.high_spoof_threshold,
            block_suspicious_count=int(self._policy.get("block_suspicious_count", 2)),
            block_high_risk_count=int(self._policy.get("block_high_risk_count", 1)),
            block_total_score=float(
                self._policy.get("block_total_score", self.low_spoof_threshold * 2.0)
            ),
        )
