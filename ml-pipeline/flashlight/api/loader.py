#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model-store/flashlight/current/ 에서 OnnxMouseBotRiskDetectorJson 싱글턴을 로드한다.

환경 변수 FLASHLIGHT_MODEL_DIR 로 모델 경로를 오버라이드할 수 있다.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from flashlight.inference.onnx_mouse_detector import OnnxMouseBotRiskDetectorJson

_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_MODEL_DIR = os.path.join(_ML_PIPELINE_ROOT, "model-store", "flashlight", "current")

_detector: Optional[OnnxMouseBotRiskDetectorJson] = None
_metadata: dict = {}


def get_model_dir() -> str:
    return os.environ.get("FLASHLIGHT_MODEL_DIR", _DEFAULT_MODEL_DIR)


def load_detector() -> OnnxMouseBotRiskDetectorJson:
    global _detector, _metadata

    model_dir = get_model_dir()
    onnx_path = os.path.join(model_dir, "mouse_gru.onnx")
    normalizer_path = os.path.join(model_dir, "normalizer.json")
    metadata_path = os.path.join(model_dir, "metadata.json")

    for path in (onnx_path, normalizer_path, metadata_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"모델 파일 없음: {path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    _detector = OnnxMouseBotRiskDetectorJson(
        onnx_path=onnx_path,
        normalizer_json_path=normalizer_path,
    )
    return _detector


def get_detector() -> OnnxMouseBotRiskDetectorJson:
    if _detector is None:
        raise RuntimeError("모델이 로드되지 않았습니다. load_detector()를 먼저 호출하세요.")
    return _detector


def get_metadata() -> dict:
    return _metadata


def is_loaded() -> bool:
    return _detector is not None
