"""
Flashlight ONNX Model Wrapper
==============================
Phase 1: ONNX 추론용 lazy 싱글톤.

- normalizer params: (x - mean) / scale, JSON 키는 seq_scaler/static_scaler
- ONNX 입력 시그니처(x_seq, x_static [, lengths])는 런타임 감지
- 출력은 logits → sigmoid 적용해 bot_risk_score 반환
- threshold는 normalizer JSON의 threshold_policy에서 동적 로드
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "flashlight"
MODEL_PATH = MODEL_DIR / "mouse_gru_server_final_v3_policy_tuned.onnx"
NORMALIZER_PATH = MODEL_DIR / "mouse_normalizer_params_v3_policy_tuned.json"


class FlashlightModel:
    _instance: Optional["FlashlightModel"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
        if not NORMALIZER_PATH.exists():
            raise FileNotFoundError(f"Normalizer params not found: {NORMALIZER_PATH}")

        self.session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )

        with open(NORMALIZER_PATH, encoding="utf-8") as f:
            norm = json.load(f)

        self.seq_mean = np.array(norm["seq_scaler"]["mean"], dtype=np.float32)
        self.seq_scale = np.array(norm["seq_scaler"]["scale"], dtype=np.float32)
        self.static_mean = np.array(norm["static_scaler"]["mean"], dtype=np.float32)
        self.static_scale = np.array(norm["static_scaler"]["scale"], dtype=np.float32)

        tp = norm.get("threshold_policy", {})
        self.low_threshold = float(tp.get("low_risk_threshold", 0.05))
        self.high_threshold = float(tp.get("high_risk_threshold", 0.60))

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

        logger.info(
            "FlashlightModel loaded: inputs=%s output=%s thresholds=(low=%s, high=%s)",
            self.input_names,
            self.output_name,
            self.low_threshold,
            self.high_threshold,
        )

    @classmethod
    def get_instance(cls) -> "FlashlightModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict(self, dynamic: np.ndarray, static: np.ndarray) -> float:
        x_seq = ((dynamic - self.seq_mean) / self.seq_scale).astype(np.float32)
        x_static = ((static - self.static_mean) / self.static_scale).astype(np.float32)
        x_seq = x_seq[np.newaxis, :, :]
        x_static = x_static[np.newaxis, :]

        feeds: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name == "x_seq":
                feeds[name] = x_seq
            elif name == "x_static":
                feeds[name] = x_static
            elif name == "lengths":
                feeds[name] = np.array([dynamic.shape[0]], dtype=np.int64)
            else:
                raise RuntimeError(f"Unexpected ONNX input name: {name}")

        outputs = self.session.run([self.output_name], feeds)
        raw = float(np.asarray(outputs[0]).squeeze())

        # 출력이 이미 sigmoid를 거친 확률([0,1])인지, raw logit인지 둘 다 대응.
        if 0.0 <= raw <= 1.0:
            score = raw
        else:
            score = 1.0 / (1.0 + math.exp(-raw))
        return score

    def classify(self, score: float) -> str:
        if score < self.low_threshold:
            return "low_risk"
        if score < self.high_threshold:
            return "suspicious"
        return "high_risk"
