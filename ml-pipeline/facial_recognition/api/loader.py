"""
model-store/facial_recognition/current/ 에서 OnnxFaceLivenessDetector 싱글턴을 로드한다.

환경변수 FACE_MODEL_DIR 로 모델 경로를 오버라이드할 수 있다.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from facial_recognition.inference.onnx_face_liveness_detector import OnnxFaceLivenessDetector

_ML_PIPELINE_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_MODEL_DIR = os.path.join(_ML_PIPELINE_ROOT, "model-store", "facial_recognition", "current")

_detector: Optional[OnnxFaceLivenessDetector] = None
_metadata: dict = {}


def get_model_dir() -> str:
    return os.environ.get("FACE_MODEL_DIR", _DEFAULT_MODEL_DIR)


def load_detector() -> OnnxFaceLivenessDetector:
    global _detector, _metadata

    model_dir = get_model_dir()
    onnx_path = os.path.join(model_dir, "face_liveness.onnx")
    meta_path = os.path.join(model_dir, "metadata.json")

    for path in (onnx_path, meta_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"모델 파일 없음: {path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    _detector = OnnxFaceLivenessDetector(
        onnx_path=onnx_path,
        meta_path=meta_path,
    )
    return _detector


def get_detector() -> OnnxFaceLivenessDetector:
    if _detector is None:
        raise RuntimeError("모델이 로드되지 않았습니다. load_detector()를 먼저 호출하세요.")
    return _detector


def get_metadata() -> dict:
    return _metadata


def is_loaded() -> bool:
    return _detector is not None
