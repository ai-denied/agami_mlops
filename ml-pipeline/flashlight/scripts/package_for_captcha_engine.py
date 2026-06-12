#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 서버 학습 산출물 → model-store candidates 패키징 스크립트

입력 (학습 run 산출물):
  mouse_gru_server_final_v2.onnx          (.onnx)
  mouse_normalizer_server_final_v2.joblib (.joblib)
  mouse_metadata_server_final_v2.json     (.json)

출력 (model-store/flashlight/candidates/{version}/):
  mouse_gru.onnx       — ONNX 추론 모델
  normalizer.json      — 정규화 파라미터 + threshold_policy
  metadata.json        — 버전/성능/아키텍처 정보

current 폴더는 절대 수정하지 않는다.
승격(candidates → current)은 별도 promote_model.py로 수행한다.

사용법:
  # version 지정 (권장)
  python -m flashlight.scripts.package_for_captcha_engine \\
    --onnx       ./runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.onnx \\
    --normalizer ./runs/mouse_gru_final_v3_policy_tuned/mouse_normalizer_server_final_v2.joblib \\
    --metadata   ./runs/mouse_gru_final_v3_policy_tuned/mouse_metadata_server_final_v2.json \\
    --version    v4_20260610

  # output-dir 직접 지정 (version 무시)
  python -m flashlight.scripts.package_for_captcha_engine \\
    --onnx       ./runs/... \\
    --normalizer ./runs/... \\
    --metadata   ./runs/... \\
    --output-dir ./model-store/flashlight/candidates/v4_custom
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import joblib
import numpy as np

OUTPUT_ONNX_NAME       = "mouse_gru.onnx"
OUTPUT_NORMALIZER_NAME = "normalizer.json"
OUTPUT_METADATA_NAME   = "metadata.json"

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CANDIDATES_DIR  = os.path.join(_ML_PIPELINE_ROOT, "model-store", "flashlight", "candidates")
_CURRENT_DIR     = os.path.join(_ML_PIPELINE_ROOT, "model-store", "flashlight", "current")

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


# ---------------------------------------------------------------------------
# 변환
# ---------------------------------------------------------------------------

def _build_metadata_json(metadata_path: str, version: str) -> dict:
    """학습 metadata JSON → model-store metadata.json 포맷으로 변환."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    summary = src.get("summary", {})
    policy_raw = src.get("three_attempt_policy", {})

    return {
        "model_name":  src.get("model_name", "MouseGRUModelV2"),
        "version":     version,
        "packaged_at": datetime.now().strftime("%Y-%m-%d"),
        "source_run":  src.get("summary", {}).get("hyperparameters", {}).get("out_dir", ""),
        "model_arch": {
            "type":             "GRU",
            "seq_input_dim":    len(src.get("seq_features", [])),
            "static_input_dim": len(src.get("static_features", [])),
            "hidden":           int(src.get("hidden", 32)),
            "layers":           int(src.get("layers", 1)),
            "dropout":          float(src.get("dropout", 0.4)),
        },
        "onnx_spec": {
            "opset":    17,
            "inputs":   ["x_seq [batch, seq_len, 7]", "lengths [batch]", "x_static [batch, 10]"],
            "output":   "bot_risk_score [batch]",
            "score_name":  src.get("score_name", "bot_risk_score"),
            "label_rule":  src.get("label_rule", "0=human, 1=bot"),
        },
        "threshold_policy": {
            "low_risk_threshold":    float(src.get("low_risk_threshold", 0.2)),
            "high_risk_threshold":   float(src.get("high_risk_threshold", 0.65)),
            "block_suspicious_count": int(policy_raw.get("block_suspicious_count", 2)),
            "block_high_risk_count":  int(policy_raw.get("block_high_risk_count", 1)),
            "block_total_score":      float(policy_raw.get("block_total_score", 0.25)),
        },
        "performance": {
            "best_epoch":               summary.get("best_epoch"),
            "test_roc_auc":             summary.get("test_roc_auc"),
            "test_pr_auc":              summary.get("test_pr_auc"),
            "test_accuracy":            summary.get("test_accuracy"),
            "test_f1_bot":              summary.get("test_f1_bot"),
            "test_human_block_rate":    summary.get("test_human_block_rate"),
            "test_bot_miss_rate":       summary.get("test_bot_miss_rate"),
            "three_attempt_bot_block_rate":   summary.get("three_attempt_test_summary", {}).get("bot_block_rate"),
            "three_attempt_human_block_rate": summary.get("three_attempt_test_summary", {}).get("human_block_rate"),
        },
    }


def _build_normalizer_json(normalizer_path: str, metadata_path: str) -> dict:
    """joblib normalizer + metadata JSON → captcha_engine JSON dict 생성."""
    normalizer = joblib.load(normalizer_path)

    seq_scaler = normalizer.seq_scaler
    static_scaler = normalizer.static_scaler

    if not hasattr(seq_scaler, "mean_"):
        raise RuntimeError("joblib normalizer가 fit되지 않았습니다 (seq_scaler.mean_ 없음)")
    if not hasattr(static_scaler, "mean_"):
        raise RuntimeError("joblib normalizer가 fit되지 않았습니다 (static_scaler.mean_ 없음)")

    assert len(seq_scaler.mean_) == len(SEQ_FEATURES), (
        f"seq_scaler 차원 불일치: {len(seq_scaler.mean_)} != {len(SEQ_FEATURES)}"
    )
    assert len(static_scaler.mean_) == len(STATIC_FEATURES), (
        f"static_scaler 차원 불일치: {len(static_scaler.mean_)} != {len(STATIC_FEATURES)}"
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    policy_raw = metadata.get("three_attempt_policy", {})

    return {
        "seq_features": SEQ_FEATURES,
        "static_features": STATIC_FEATURES,
        "seq_scaler": {
            "mean": seq_scaler.mean_.tolist(),
            "scale": seq_scaler.scale_.tolist(),
            "var": seq_scaler.var_.tolist(),
        },
        "static_scaler": {
            "mean": static_scaler.mean_.tolist(),
            "scale": static_scaler.scale_.tolist(),
            "var": static_scaler.var_.tolist(),
        },
        "threshold_policy": {
            "low_risk_threshold": float(metadata.get("low_risk_threshold", 0.2)),
            "high_risk_threshold": float(metadata.get("high_risk_threshold", 0.65)),
            "block_suspicious_count": int(policy_raw.get("block_suspicious_count", 2)),
            "block_high_risk_count": int(policy_raw.get("block_high_risk_count", 1)),
            "block_total_score": float(policy_raw.get("block_total_score", 0.25)),
        },
    }


def package(
    onnx_src: str,
    normalizer_src: str,
    metadata_src: str,
    output_dir: str,
    version: str,
) -> tuple[str, str, str]:
    """산출물을 output_dir에 model-store candidates 포맷으로 패키징한다."""
    for path, label in [
        (onnx_src, "--onnx"),
        (normalizer_src, "--normalizer"),
        (metadata_src, "--metadata"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} 파일을 찾을 수 없습니다: {path}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. ONNX 복사
    onnx_dst = os.path.join(output_dir, OUTPUT_ONNX_NAME)
    shutil.copy2(onnx_src, onnx_dst)
    print(f"[OK] ONNX 복사:            {onnx_src}")
    print(f"                        → {onnx_dst}")

    # 2. normalizer.json 생성
    normalizer_dict = _build_normalizer_json(normalizer_src, metadata_src)
    normalizer_dst = os.path.join(output_dir, OUTPUT_NORMALIZER_NAME)
    with open(normalizer_dst, "w", encoding="utf-8") as f:
        json.dump(normalizer_dict, f, indent=2, ensure_ascii=False)
    print(f"[OK] normalizer.json 생성: {normalizer_dst}")

    # 3. metadata.json 생성
    metadata_dict = _build_metadata_json(metadata_src, version)
    metadata_dst = os.path.join(output_dir, OUTPUT_METADATA_NAME)
    with open(metadata_dst, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    print(f"[OK] metadata.json 생성:   {metadata_dst}")

    return onnx_dst, normalizer_dst, metadata_dst


# ---------------------------------------------------------------------------
# 검증 (captcha_engine 로딩 시뮬레이션)
# ---------------------------------------------------------------------------

def _load_and_normalize_json(normalizer_json_path: str):
    """JSON 파라미터를 로드해 numpy 기반 transform 함수를 반환한다."""
    with open(normalizer_json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # 필수 키 검사
    for key in ("seq_features", "static_features", "seq_scaler", "static_scaler", "threshold_policy"):
        if key not in params:
            raise ValueError(f"normalizer JSON에 '{key}' 키가 없습니다")
    for sub in ("mean", "scale", "var"):
        if sub not in params["seq_scaler"]:
            raise ValueError(f"seq_scaler에 '{sub}' 키가 없습니다")
        if sub not in params["static_scaler"]:
            raise ValueError(f"static_scaler에 '{sub}' 키가 없습니다")

    seq_mean = np.array(params["seq_scaler"]["mean"], dtype=np.float32)
    seq_scale = np.array(params["seq_scaler"]["scale"], dtype=np.float32)
    static_mean = np.array(params["static_scaler"]["mean"], dtype=np.float32)
    static_scale = np.array(params["static_scaler"]["scale"], dtype=np.float32)

    n_seq = len(params["seq_features"])
    n_static = len(params["static_features"])
    policy = params["threshold_policy"]

    def transform_seq(raw: np.ndarray) -> np.ndarray:
        return ((raw - seq_mean) / seq_scale).astype(np.float32)

    def transform_static(raw: np.ndarray) -> np.ndarray:
        return ((raw - static_mean) / static_scale).astype(np.float32)

    return transform_seq, transform_static, n_seq, n_static, policy


def validate(onnx_path: str, normalizer_json_path: str) -> None:
    """
    1. JSON normalizer 구조 검증
    2. ONNX 세션 로딩 검증
    3. 더미 inference로 score 범위 확인
    4. threshold_policy 키 존재 확인
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[SKIP] onnxruntime 미설치 — 추론 검증 생략")
        return

    print("\n[검증 시작]")

    # JSON 로드 및 normalizer 구조 검증
    transform_seq, transform_static, n_seq, n_static, policy = _load_and_normalize_json(normalizer_json_path)
    print(f"[OK] normalizer JSON 로드 — seq_dim={n_seq}, static_dim={n_static}")

    # threshold_policy 필수 키
    for key in ("low_risk_threshold", "high_risk_threshold", "block_suspicious_count",
                "block_high_risk_count", "block_total_score"):
        if key not in policy:
            raise ValueError(f"threshold_policy에 '{key}' 키가 없습니다")
    print(f"[OK] threshold_policy = {policy}")

    # ONNX 세션 로드
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}
    expected = {"x_seq", "lengths", "x_static"}
    if expected - input_names:
        raise ValueError(f"ONNX 입력 이름 불일치: expected {expected}, got {input_names}")
    print(f"[OK] ONNX 세션 로드 — 입력: {input_names}")

    # 더미 inference (seq_len=32, batch=1)
    seq_len = 32
    rng = np.random.default_rng(42)
    raw_seq = rng.standard_normal((seq_len, n_seq)).astype(np.float32)
    raw_static = rng.standard_normal(n_static).astype(np.float32)

    scaled_seq = transform_seq(raw_seq)
    scaled_static = transform_static(raw_static)

    x_seq = scaled_seq[None, :, :]
    lengths = np.array([seq_len], dtype=np.int64)
    x_static = scaled_static[None, :]

    outputs = session.run(
        ["bot_risk_score"],
        {"x_seq": x_seq, "lengths": lengths, "x_static": x_static},
    )
    score = float(outputs[0][0])

    if not (0.0 <= score <= 1.0):
        raise ValueError(f"bot_risk_score 범위 오류: {score}")
    print(f"[OK] 더미 추론 성공 — bot_risk_score = {score:.6f}")

    # 경계값 테스트: seq_len=1
    raw_seq_min = rng.standard_normal((1, n_seq)).astype(np.float32)
    x_seq_min = transform_seq(raw_seq_min)[None, :, :]
    lengths_min = np.array([1], dtype=np.int64)
    out_min = session.run(
        ["bot_risk_score"],
        {"x_seq": x_seq_min, "lengths": lengths_min, "x_static": x_static},
    )
    score_min = float(out_min[0][0])
    if not (0.0 <= score_min <= 1.0):
        raise ValueError(f"seq_len=1 경계값 테스트 실패: {score_min}")
    print(f"[OK] seq_len=1 경계값 테스트 — bot_risk_score = {score_min:.6f}")

    print("\n[검증 완료] captcha_engine 로딩 검증 통과")


# ---------------------------------------------------------------------------
# 요약 출력
# ---------------------------------------------------------------------------

def _print_summary(normalizer_json_path: str, metadata_json_path: str) -> None:
    with open(normalizer_json_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    policy = params["threshold_policy"]
    seq_mean = params["seq_scaler"]["mean"]
    static_mean = params["static_scaler"]["mean"]
    perf = meta.get("performance", {})

    print("\n--- 패키지 요약 ---")
    print(f"version:            {meta.get('version')}  (packaged_at: {meta.get('packaged_at')})")
    print(f"seq_features ({len(params['seq_features'])}):    {params['seq_features']}")
    print(f"static_features ({len(params['static_features'])}): {params['static_features']}")
    print(f"seq_scaler.mean:    {[round(v, 4) for v in seq_mean]}")
    print(f"static_scaler.mean: {[round(v, 4) for v in static_mean]}")
    print(f"threshold_policy:   {policy}")
    print(f"test_roc_auc:       {perf.get('test_roc_auc')}  f1_bot: {perf.get('test_f1_bot')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="학습 run 산출물 → model-store/flashlight/candidates/{version}/ 패키징"
    )
    parser.add_argument(
        "--onnx",
        default="./runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.onnx",
        help="입력 ONNX 경로 (.onnx)",
    )
    parser.add_argument(
        "--normalizer",
        default="./runs/mouse_gru_final_v3_policy_tuned/mouse_normalizer_server_final_v2.joblib",
        help="입력 normalizer 경로 (.joblib)",
    )
    parser.add_argument(
        "--metadata",
        default="./runs/mouse_gru_final_v3_policy_tuned/mouse_metadata_server_final_v2.json",
        help="입력 metadata 경로 (.json)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="후보 버전명 (예: v4_20260610). 미지정 시 날짜 자동 생성.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="출력 디렉토리 직접 지정. 지정 시 --version 기반 기본 경로보다 우선.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="검증 단계 건너뜀",
    )
    return parser.parse_args()


def _resolve_output_dir(args) -> tuple[str, str]:
    """(output_dir, version) 결정 로직."""
    version = args.version or f"v_{datetime.now().strftime('%Y%m%d')}"

    if args.output_dir:
        return args.output_dir, version

    return os.path.join(_CANDIDATES_DIR, version), version


def main():
    args = _parse_args()
    output_dir, version = _resolve_output_dir(args)

    # current/ 보호 — 절대 덮어쓰지 않음 (상대·절대 경로 모두 차단)
    if os.path.realpath(output_dir) == os.path.realpath(_CURRENT_DIR):
        print("[ERROR] output-dir이 model-store/flashlight/current/입니다.")
        print("        current 폴더는 package 명령으로 직접 수정할 수 없습니다.")
        print("        승격은 promote_model.py를 사용하세요.")
        sys.exit(1)

    print("=== model-store candidates 패키징 시작 ===")
    print(f"버전:            {version}")
    print(f"입력 ONNX:       {args.onnx}")
    print(f"입력 normalizer: {args.normalizer}")
    print(f"입력 metadata:   {args.metadata}")
    print(f"출력 디렉토리:    {output_dir}\n")

    onnx_dst, normalizer_dst, metadata_dst = package(
        onnx_src=args.onnx,
        normalizer_src=args.normalizer,
        metadata_src=args.metadata,
        output_dir=output_dir,
        version=version,
    )

    _print_summary(normalizer_dst, metadata_dst)

    if not args.skip_validate:
        validate(onnx_dst, normalizer_dst)

    print(f"\n출력 파일:")
    print(f"  {onnx_dst}")
    print(f"  {normalizer_dst}")
    print(f"  {metadata_dst}")
    print("\n=== 완료 ===")
    print(f"\n다음 단계 — 후보 검증 후 current로 승격:")
    print(f"  python -m flashlight.scripts.promote_model --version {version}")


if __name__ == "__main__":
    main()
