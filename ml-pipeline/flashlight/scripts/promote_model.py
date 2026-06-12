#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model-store/flashlight/candidates/{version}/ → current/ 승격 스크립트

승격 절차:
  1. candidates/{version}/ 파일 존재 확인
  2. (선택) ONNX 로딩 검증
  3. current/ → archive/{timestamp}_{old_version}/ 백업
  4. candidates/{version}/ → current_staging_{timestamp}/ 복사
  5. metadata.json에 promoted_at, promoted_from_candidate 기록
  6. current_staging/ → current/ 원자적 교체
  7. 실패 시 current/ 자동 복원

사용법:
  python -m flashlight.scripts.promote_model --version v4_20260610
  python -m flashlight.scripts.promote_model --version v4_20260610 --dry-run
  python -m flashlight.scripts.promote_model --version v4_20260610 --skip-validate
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_STORE            = os.path.join(_ML_PIPELINE_ROOT, "model-store", "flashlight")
_CANDIDATES_DIR   = os.path.join(_STORE, "candidates")
_CURRENT_DIR      = os.path.join(_STORE, "current")
_ARCHIVE_DIR      = os.path.join(_STORE, "archive")

REQUIRED_FILES = ["mouse_gru.onnx", "normalizer.json", "metadata.json"]


# ---------------------------------------------------------------------------
# 단계별 함수
# ---------------------------------------------------------------------------

def _log(msg: str, dry: bool = False) -> None:
    prefix = "[DRY-RUN] " if dry else ""
    print(f"{prefix}{msg}")


def _validate_candidate(candidate_dir: str) -> None:
    """후보 폴더에 필수 파일 3개가 모두 있는지 확인한다."""
    if not os.path.isdir(candidate_dir):
        raise FileNotFoundError(f"candidates 폴더가 없습니다: {candidate_dir}")

    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(candidate_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"필수 파일이 없습니다: {missing}\n"
            f"  경로: {candidate_dir}"
        )


def _validate_onnx(onnx_path: str, normalizer_json_path: str) -> None:
    """ONNX 세션 로딩 및 더미 추론으로 후보 모델을 검증한다."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"[SKIP] 검증 의존성 미설치 — {e}")
        return

    with open(normalizer_json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    n_seq    = len(params["seq_features"])
    n_static = len(params["static_features"])

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}
    expected = {"x_seq", "lengths", "x_static"}
    if expected - input_names:
        raise ValueError(f"ONNX 입력 이름 불일치: expected={expected}, got={input_names}")

    rng = np.random.default_rng(42)
    x_seq    = rng.standard_normal((1, 32, n_seq)).astype("float32")
    lengths  = np.array([32], dtype="int64")
    x_static = rng.standard_normal((1, n_static)).astype("float32")

    out = session.run(["bot_risk_score"], {"x_seq": x_seq, "lengths": lengths, "x_static": x_static})
    score = float(out[0][0])
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"bot_risk_score 범위 오류: {score}")

    print(f"[OK] ONNX 검증 통과 — dummy bot_risk_score = {score:.6f}")


def _read_current_version() -> str:
    """current/metadata.json에서 version을 읽는다. 없으면 'unknown'."""
    meta_path = os.path.join(_CURRENT_DIR, "metadata.json")
    if not os.path.isfile(meta_path):
        return "unknown"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f).get("version", "unknown")
    except Exception:
        return "unknown"


def _backup_current(timestamp: str, old_version: str, dry: bool) -> str:
    """current/ → archive/{timestamp}_{old_version}/ 이름으로 백업."""
    archive_name = f"{timestamp}_{old_version}"
    archive_dst  = os.path.join(_ARCHIVE_DIR, archive_name)

    _log(f"백업: current/ → archive/{archive_name}/", dry)

    if not dry:
        os.makedirs(_ARCHIVE_DIR, exist_ok=True)
        shutil.copytree(_CURRENT_DIR, archive_dst)

    return archive_dst


def _build_staging(candidate_dir: str, timestamp: str, version: str, dry: bool) -> str:
    """candidate 파일을 staging 폴더에 복사하고 metadata를 갱신한다."""
    staging_dir = os.path.join(_STORE, f"current_staging_{timestamp}")

    _log(f"스테이징: candidates/{version}/ → {os.path.basename(staging_dir)}/", dry)

    if not dry:
        shutil.copytree(candidate_dir, staging_dir)

        meta_path = os.path.join(staging_dir, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta["promoted_at"]              = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta["promoted_from_candidate"]  = f"candidates/{version}"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return staging_dir


def _atomic_swap(staging_dir: str, dry: bool) -> None:
    """
    current/ 를 staging/ 으로 원자적으로 교체한다.

    os.replace()는 같은 파일시스템 내에서 디렉토리를 원자적으로 교체하지 않으므로,
    현재 current를 먼저 _current_old로 rename 후 staging을 current로 rename한다.
    두 rename이 모두 빠르고, 실패 시 _current_old를 복원할 수 있다.
    """
    old_tmp = os.path.join(_STORE, "_current_old")

    _log("원자적 교체: current/ ← staging/", dry)

    if dry:
        return

    # 기존 current를 임시 이름으로 이동
    os.rename(_CURRENT_DIR, old_tmp)
    try:
        # staging을 current로 이동
        os.rename(staging_dir, _CURRENT_DIR)
    except Exception:
        # staging → current 실패 시 이전 current 복원
        os.rename(old_tmp, _CURRENT_DIR)
        raise

    # 성공 — 임시 이름의 old 폴더 삭제
    shutil.rmtree(old_tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 메인 승격 로직
# ---------------------------------------------------------------------------

def promote(version: str, dry: bool, skip_validate: bool) -> None:
    candidate_dir = os.path.join(_CANDIDATES_DIR, version)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir   = os.path.join(_STORE, f"current_staging_{timestamp}")

    print("=" * 60)
    print("  flashlight model-store 승격")
    print("=" * 60)
    print(f"  후보 버전:  {version}")
    print(f"  후보 경로:  {candidate_dir}")
    print(f"  current:   {_CURRENT_DIR}")
    if dry:
        print("  모드:       DRY-RUN (파일 변경 없음)")
    print()

    # ── 1. 후보 파일 존재 확인 ──────────────────────────────────────────────
    print("[1/5] 후보 파일 확인")
    _validate_candidate(candidate_dir)
    for f in REQUIRED_FILES:
        size = os.path.getsize(os.path.join(candidate_dir, f))
        print(f"  [OK] {f}  ({size:,} bytes)")

    # ── 2. ONNX 검증 (선택) ────────────────────────────────────────────────
    print("\n[2/5] ONNX 모델 검증")
    if skip_validate:
        print("  [SKIP] --skip-validate 지정")
    else:
        _validate_onnx(
            os.path.join(candidate_dir, "mouse_gru.onnx"),
            os.path.join(candidate_dir, "normalizer.json"),
        )

    # ── 3. current 백업 ────────────────────────────────────────────────────
    print("\n[3/5] current 백업")
    old_version = _read_current_version()
    archive_dst = _backup_current(timestamp, old_version, dry)
    if not dry:
        print(f"  [OK] archive/{os.path.basename(archive_dst)}/")

    # ── 4. 스테이징 준비 ───────────────────────────────────────────────────
    print("\n[4/5] 스테이징 준비")
    try:
        _build_staging(candidate_dir, timestamp, version, dry)
        if not dry:
            print(f"  [OK] {os.path.basename(staging_dir)}/  (metadata 갱신 포함)")
    except Exception as e:
        # staging 실패 — current 미변경, staging 정리
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"스테이징 실패 (current 유지): {e}") from e

    # ── 5. 원자적 교체 ────────────────────────────────────────────────────
    print("\n[5/5] current 교체")
    try:
        _atomic_swap(staging_dir, dry)
    except Exception as e:
        # 교체 실패 — staging 정리
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"교체 실패 (current 유지): {e}") from e

    # ── 완료 ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if dry:
        print("  [DRY-RUN 완료] 실제 파일 변경 없음")
    else:
        print("  승격 완료")
        print(f"  current 버전:  {version}")
        print(f"  백업 경로:     archive/{os.path.basename(archive_dst)}/")
        print()
        print("  current/ 최종 파일:")
        for f in sorted(os.listdir(_CURRENT_DIR)):
            size = os.path.getsize(os.path.join(_CURRENT_DIR, f))
            print(f"    {f}  ({size:,} bytes)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="candidates/{version}/ → current/ 모델 승격"
    )
    parser.add_argument(
        "--version",
        required=True,
        help="승격할 후보 버전명 (예: v4_20260610)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 파일 변경 없이 절차만 출력",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="ONNX 로딩 검증 건너뜀",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    try:
        promote(
            version=args.version,
            dry=args.dry_run,
            skip_validate=args.skip_validate,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[FAILED] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
