#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flashlight ONNX 입출력 contract 검증.

config/runtime_contract.yaml(고정 계약)과 metadata.json의 onnx_spec(패키징
스크립트가 "이 모델은 이렇게 생겼다"고 주장하는 값) 양쪽을 실제 onnx 그래프
(onnxruntime으로 읽은 input/output의 이름·shape·dtype)와 대조한다.

metadata.json 필드 존재 여부만 보는 게 아니라, 실제 onnx 파일을 열어서 직접
비교하는 게 핵심이다 - metadata.json은 사람/스크립트가 잘못 적을 수 있고,
inference/onnx_mouse_detector.py는 그 metadata.json을 보지 않고 텐서 이름을
하드코딩해서 session.run()을 호출하므로, 실제 그래프가 다르면 서빙이 깨진다.

compare_candidate.py / promote_model.py가 승격 전 게이트로 사용한다.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FLASHLIGHT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_RUNTIME_CONTRACT_PATH = os.path.join(_FLASHLIGHT_ROOT, "config", "runtime_contract.yaml")

# onnxruntime이 보고하는 텐서 타입 문자열(예: "tensor(float)")과 contract.yaml에
# 적는 사람이 읽기 쉬운 dtype 이름을 매핑한다.
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
    한다. contract에 문자열(batch/seq_len 등)로 적힌 가변 차원은 onnx 쪽이
    무엇이든(정수든 symbolic 문자열이든) 허용한다."""
    if len(actual_shape) != len(expected_shape):
        return False
    for actual_dim, expected_dim in zip(actual_shape, expected_shape):
        if isinstance(expected_dim, int):
            if not isinstance(actual_dim, int) or actual_dim != expected_dim:
                return False
    return True


def _check_io(label: str, actual: Dict[str, Dict[str, Any]], expected: List[dict]) -> List[str]:
    """actual(실제 onnx에서 읽은 입력 또는 출력 스펙) vs expected(contract 또는
    metadata.json에서 가져온 기대 스펙 목록)를 대조한다."""
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


# metadata.json의 onnx_spec.inputs/output은 "x_seq [batch, seq_len, 7]" 같은
# 사람이 읽기 좋은 설명용 문자열이다 (package_for_captcha_engine.py가 생성).
_SPEC_STR_RE = re.compile(r"^(?P<name>\S+)\s*\[(?P<dims>[^\]]*)\]\s*$")


def _parse_spec_strings(spec_strings: List[str]) -> List[dict]:
    """'x_seq [batch, seq_len, 7]' -> {"name": "x_seq", "shape": ["batch", "seq_len", 7]}.
    숫자로 적힌 차원만 int로 변환하고 나머지(batch/seq_len 등)는 문자열로 남겨
    _shape_matches()가 가변 차원으로 인식하게 한다. 괄호가 없는 문자열은 이름만
    추출하고 shape 검증은 건너뛴다."""
    parsed = []
    for raw in spec_strings:
        s = raw.strip()
        m = _SPEC_STR_RE.match(s)
        if not m:
            parsed.append({"name": s, "shape": None})
            continue
        dims: List[Any] = []
        for d in m.group("dims").split(","):
            d = d.strip()
            if not d:
                continue
            dims.append(int(d) if re.fullmatch(r"-?\d+", d) else d)
        parsed.append({"name": m.group("name"), "shape": dims})
    return parsed


def validate_against_metadata(onnx_path: str, metadata: dict) -> List[str]:
    """실제 onnx 그래프 vs metadata.json의 onnx_spec (패키징 시점에 '주장'된 스펙).
    둘이 다르면 metadata.json이 실제로 승격하려는 모델을 잘못 기술하고 있다는
    뜻 - compare/promote 단계가 이 잘못된 설명을 신뢰하면 안 된다."""
    onnx_spec = metadata.get("onnx_spec")
    if not onnx_spec:
        return ["metadata.json에 onnx_spec이 없음 - 실제 onnx와 대조할 기준이 없음"]

    actual_inputs, actual_outputs = _onnx_io_specs(onnx_path)

    problems: List[str] = []

    raw_inputs = onnx_spec.get("inputs")
    if raw_inputs:
        problems += _check_io(
            "metadata.onnx_spec.input", actual_inputs, _parse_spec_strings(raw_inputs)
        )

    raw_output = onnx_spec.get("output")
    if raw_output:
        output_strings = raw_output if isinstance(raw_output, list) else [raw_output]
        problems += _check_io(
            "metadata.onnx_spec.output", actual_outputs, _parse_spec_strings(output_strings)
        )

    return problems


def validate_candidate_onnx_contract(
    onnx_path: str,
    metadata: dict,
    contract: Optional[dict] = None,
) -> List[str]:
    """승격 게이트가 호출하는 단일 진입점. runtime_contract.yaml +
    metadata.json.onnx_spec 둘 다 실제 onnx 그래프와 대조한다. 빈 리스트 = 통과."""
    problems: List[str] = []
    problems += validate_onnx_contract(onnx_path, contract=contract)
    problems += validate_against_metadata(onnx_path, metadata)
    return problems
