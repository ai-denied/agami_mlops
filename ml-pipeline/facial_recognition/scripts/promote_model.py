"""
model-store/facial_recognition/candidates/{version}/ → current/ 승격 스크립트.

승격 절차:
  1. candidates/{version}/ 파일 존재 확인
  2. ONNX 로딩 검증 (선택)
  3. current/ → archive/{timestamp}_{old_version}/ 백업
  4. candidates/{version}/ → current_staging_{timestamp}/ 복사
  5. metadata.json 에 promoted_at 기록
  6. current_staging/ → current/ 원자적 교체
  7. 실패 시 current/ 자동 복원

사용법:
  python -m facial_recognition.scripts.promote_model --version v1_20260616
  python -m facial_recognition.scripts.promote_model --version v1_20260616 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from facial_recognition.evaluation.onnx_contract_check import validate_candidate_onnx_contract

_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_STORE            = os.path.join(_ML_PIPELINE_ROOT, "model-store", "facial_recognition")
_CANDIDATES_DIR   = os.path.join(_STORE, "candidates")
_CURRENT_DIR      = os.path.join(_STORE, "current")
_ARCHIVE_DIR      = os.path.join(_STORE, "archive")

REQUIRED_FILES = ["face_liveness.onnx", "seq_scaler.joblib", "metadata.json"]


def _log(msg: str, dry: bool = False) -> None:
    prefix = "[DRY-RUN] " if dry else ""
    print(f"{prefix}{msg}")


def _validate_candidate(candidate_dir: str) -> None:
    if not os.path.isdir(candidate_dir):
        raise FileNotFoundError(f"candidates 폴더가 없습니다: {candidate_dir}")
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(candidate_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"필수 파일이 없습니다: {missing}\n경로: {candidate_dir}"
        )


def _validate_onnx(onnx_path: str, metadata_path: str) -> None:
    """ONNX runtime contract 대조 + 세션 로딩/더미 추론으로 후보 모델을 검증한다.

    contract 위반(텐서 이름/shape/dtype 변경)은 더미 추론이 우연히 성공하더라도
    잘못된 모델이 승격되는 사고를 막는 핵심 게이트다 - 먼저 검사해서 실패하면
    예외를 던지고, 더미 추론까지 가지 않는다."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"[SKIP] 검증 의존성 미설치 — {e}")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    contract_problems = validate_candidate_onnx_contract(onnx_path, metadata)
    if contract_problems:
        raise ValueError(
            "ONNX runtime contract 위반 — 승격 불가:\n" +
            "\n".join(f"  - {p}" for p in contract_problems)
        )
    print("[OK] ONNX runtime contract 검증 통과 (x_seq -> spoof_score)")

    sess   = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy  = np.zeros((1, 16, 20), dtype=np.float32)
    out    = sess.run(["spoof_score"], {"x_seq": dummy})
    score  = float(out[0][0])
    assert 0.0 <= score <= 1.0, f"spoof_score 범위 오류: {score}"
    print(f"[OK] ONNX 검증 통과  dummy spoof_score={score:.6f}")


def _read_current_version() -> str:
    meta_path = os.path.join(_CURRENT_DIR, "metadata.json")
    if not os.path.isfile(meta_path):
        return "unknown"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f).get("version", "unknown")
    except Exception:
        return "unknown"


def _backup_current(timestamp: str, old_version: str, dry: bool) -> str:
    archive_name = f"{timestamp}_{old_version}"
    archive_dst  = os.path.join(_ARCHIVE_DIR, archive_name)
    _log(f"백업: current/ → archive/{archive_name}/", dry)
    if not dry:
        os.makedirs(_ARCHIVE_DIR, exist_ok=True)
        shutil.copytree(_CURRENT_DIR, archive_dst)
    return archive_dst


def _build_staging(candidate_dir: str, timestamp: str, version: str, dry: bool) -> str:
    staging_dir = os.path.join(_STORE, f"current_staging_{timestamp}")
    _log(f"스테이징: candidates/{version}/ → {os.path.basename(staging_dir)}/", dry)
    if not dry:
        shutil.copytree(candidate_dir, staging_dir)
        meta_path = os.path.join(staging_dir, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["promoted_at"]             = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta["promoted_from_candidate"] = f"candidates/{version}"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    return staging_dir


def _atomic_swap(staging_dir: str, dry: bool) -> None:
    old_tmp = os.path.join(_STORE, "_current_old")
    _log("원자적 교체: current/ ← staging/", dry)
    if dry:
        return
    os.rename(_CURRENT_DIR, old_tmp)
    try:
        os.rename(staging_dir, _CURRENT_DIR)
    except Exception:
        os.rename(old_tmp, _CURRENT_DIR)
        raise
    shutil.rmtree(old_tmp, ignore_errors=True)


def promote(version: str, dry: bool, skip_validate: bool) -> None:
    candidate_dir = os.path.join(_CANDIDATES_DIR, version)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir   = os.path.join(_STORE, f"current_staging_{timestamp}")

    print("=" * 60)
    print("  facial_recognition model-store 승격")
    print("=" * 60)
    print(f"  후보 버전:  {version}")
    print(f"  후보 경로:  {candidate_dir}")
    print(f"  current:   {_CURRENT_DIR}")
    if dry:
        print("  모드:       DRY-RUN (파일 변경 없음)")
    print()

    print("[1/5] 후보 파일 확인")
    _validate_candidate(candidate_dir)
    for f in REQUIRED_FILES:
        size = os.path.getsize(os.path.join(candidate_dir, f))
        print(f"  [OK] {f}  ({size:,} bytes)")

    print("\n[2/5] ONNX 모델 검증")
    if skip_validate:
        print("  [SKIP] --skip-validate 지정")
    else:
        _validate_onnx(
            os.path.join(candidate_dir, "face_liveness.onnx"),
            os.path.join(candidate_dir, "metadata.json"),
        )

    print("\n[3/5] current 백업")
    old_version = _read_current_version()
    archive_dst = _backup_current(timestamp, old_version, dry)
    if not dry:
        print(f"  [OK] archive/{os.path.basename(archive_dst)}/")

    print("\n[4/5] 스테이징 준비")
    try:
        _build_staging(candidate_dir, timestamp, version, dry)
        if not dry:
            print(f"  [OK] {os.path.basename(staging_dir)}/")
    except Exception as e:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"스테이징 실패 (current 유지): {e}") from e

    print("\n[5/5] current 교체")
    try:
        _atomic_swap(staging_dir, dry)
    except Exception as e:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"교체 실패 (current 유지): {e}") from e

    print()
    print("=" * 60)
    if dry:
        print("  [DRY-RUN 완료] 실제 파일 변경 없음")
    else:
        print("  승격 완료")
        print(f"  current 버전: {version}")
        print(f"  백업 경로:    archive/{os.path.basename(archive_dst)}/")
        print("\n  current/ 최종 파일:")
        for f in sorted(os.listdir(_CURRENT_DIR)):
            size = os.path.getsize(os.path.join(_CURRENT_DIR, f))
            print(f"    {f}  ({size:,} bytes)")
    print("=" * 60)


def _parse_args():
    parser = argparse.ArgumentParser(description="candidates/{version}/ → current/ 승격")
    parser.add_argument("--version",       required=True, help="승격할 버전명 (예: v1_20260616)")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        promote(version=args.version, dry=args.dry_run, skip_validate=args.skip_validate)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[FAILED] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
