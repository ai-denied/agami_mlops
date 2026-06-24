#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facial_recognition ONNX 입출력 contract 검증.

config/runtime_contract.yaml(고정 계약)과 metadata.json의 input_name/input_shape/
output_name/output_shape(export_face_liveness_onnx.py가 "이 모델은 이렇게
생겼다"고 적어두는 값) 양쪽을 실제 onnx 그래프(onnxruntime으로 읽은
input/output의 이름·shape·dtype)와 대조한다.

metadata.json 필드 존재 여부만 보는 게 아니라, 실제 onnx 파일을 열어서 직접
비교하는 게 핵심이다 - metadata.json은 export 스크립트가 잘못 적을 수 있고,
inference/onnx_face_liveness_detector.py는 그 metadata.json을 보지 않고
"x_seq" -> "spoof_score" 텐서 이름을 하드코딩해서 session.run()을 호출하므로,
실제 그래프가 다르면 서빙이 깨진다.

scripts/promote_model.py가 승격 전 게이트로 사용한다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FACIAL_RECOGNITION_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_RUNTIME_CONTRACT_PATH = os.path.join(_FACIAL_RECOGNITION_ROOT, "config", "runtime_contract.yaml")

_DTYPE_TO_ONNX = {
    "float32": "tensor(float)",
    "float64": "tensor(double)",
    "int64": "tensor(int64)",
    "int32": "tensor(int32)",
}


def load_runtime_contract(path: str = _RUNTIME_CONTRACT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _onnx_io_specs(onnx_path: str):
    """실제 onnx 그래프에서 (inputs, outputs) 두 개의 {name: {shape, dtype}} 딕셔너리를 읽는다."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def _to_spec(items) -> Dict[str, Dict[str, Any]]:
        return {it.name: {"shape": list(it.shape), "dtype": it.type} for it in items}

    return _to_spec(session.get_inputs()), _to_spec(session.get_outputs())


def _shape_matches(actual_shape: List, expected_shape: List) -> bool:
    """차원 개수가 같아야 하고, contract에 정수로 고정된 차원은 정확히 일치해야
    한다. contract에 문자열(batch 등)로 적힌 가변 차원은 onnx 쪽이 무엇이든
    (정수든 symbolic 문자열이든) 허용한다."""
    if len(actual_shape) != len(expected_shape):
        return False
    for actual_dim, expected_dim in zip(actual_shape, expected_shape):
        if isinstance(expected_dim, int):
            if not isinstance(actual_dim, int) or actual_dim != expected_dim:
                return False
    return True


def _check_io(label: str, actual: Dict[str, Dict[str, Any]], expected: List[dict]) -> List[str]:
    problems: List[str] = []

    expected_names = {spec["name"] for spec in expected if spec.get("name")}
    actual_names = set(actual.keys())

    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing:
        problems.append(f"{label}: 계약에 있는 이름이 실제 onnx 그래프에 없음: {sorted(missing)}")
    if extra:
        problems.append(f"{label}: 실제 onnx 그래프에 계약에 없는 이름이 있음: {sorted(extra)}")

    for spec in expected:
        name = spec.get("name")
        if not name or name not in actual:
            continue
        actual_spec = actual[name]

        expected_shape = spec.get("shape")
        if expected_shape and not _shape_matches(actual_spec["shape"], expected_shape):
            problems.append(
                f"{label}.{name}: shape 불일치 — 기대={expected_shape}, 실제 onnx={actual_spec['shape']}"
            )

        expected_dtype = spec.get("dtype")
        if expected_dtype:
            expected_onnx_dtype = _DTYPE_TO_ONNX.get(expected_dtype, expected_dtype)
            if actual_spec["dtype"] != expected_onnx_dtype:
                problems.append(
                    f"{label}.{name}: dtype 불일치 — 기대={expected_dtype} ({expected_onnx_dtype}), "
                    f"실제 onnx={actual_spec['dtype']}"
                )

    return problems


def validate_onnx_contract(onnx_path: str, contract: Optional[dict] = None) -> List[str]:
    """실제 onnx 그래프 vs config/runtime_contract.yaml. 빈 리스트 = 통과."""
    contract = contract or load_runtime_contract()
    onnx_contract = contract["onnx_contract"]

    actual_inputs, actual_outputs = _onnx_io_specs(onnx_path)

    problems: List[str] = []
    problems += _check_io("runtime_contract.input", actual_inputs, onnx_contract["inputs"])
    problems += _check_io("runtime_contract.output", actual_outputs, onnx_contract["outputs"])
    return problems


def _metadata_io_specs(metadata: dict) -> List[dict]:
    """metadata.json의 flat 필드(input_name/input_shape/input_dtype,
    output_name/output_shape/output_dtype)를 _check_io()가 받는
    {"name", "shape", "dtype"} 목록 형태로 변환한다. export_face_liveness_onnx.py가
    실제로 쓰는 키 이름과 정확히 맞춰야 한다."""
    specs = []
    if metadata.get("input_name"):
        specs.append({
            "name": metadata["input_name"],
            "shape": metadata.get("input_shape"),
            "dtype": metadata.get("input_dtype"),
        })
    if metadata.get("output_name"):
        specs.append({
            "name": metadata["output_name"],
            "shape": metadata.get("output_shape"),
            "dtype": metadata.get("output_dtype"),  # export 스크립트가 안 쓰면 None - 검증 생략됨
        })
    return specs


def validate_against_metadata(onnx_path: str, metadata: dict) -> List[str]:
    """실제 onnx 그래프 vs metadata.json의 input_name/input_shape/output_name/
    output_shape (export 시점에 '주장'된 스펙). 둘이 다르면 metadata.json이
    실제로 승격하려는 모델을 잘못 기술하고 있다는 뜻이다."""
    specs = _metadata_io_specs(metadata)
    if not specs:
        return ["metadata.json에 input_name/output_name이 없음 - 실제 onnx와 대조할 기준이 없음"]

    actual_inputs, actual_outputs = _onnx_io_specs(onnx_path)
    combined_actual = {**actual_inputs, **actual_outputs}

    return _check_io("metadata", combined_actual, specs)


def validate_candidate_onnx_contract(
    onnx_path: str,
    metadata: dict,
    contract: Optional[dict] = None,
) -> List[str]:
    """승격 게이트가 호출하는 단일 진입점. runtime_contract.yaml + metadata.json
    둘 다 실제 onnx 그래프와 대조한다. 빈 리스트 = 통과."""
    problems: List[str] = []
    problems += validate_onnx_contract(onnx_path, contract=contract)
    problems += validate_against_metadata(onnx_path, metadata)
    return problems
