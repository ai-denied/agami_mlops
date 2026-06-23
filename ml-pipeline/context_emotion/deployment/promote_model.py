#!/usr/bin/env python3
"""model-store/context_emotion/candidates/{version}/ -> current/ 승격.

flashlight/scripts/promote_model.py와 동일한 절차 (백업 -> 스테이징 ->
원자적 교체 -> 실패 시 자동 복원)지만, "후보 파일 존재 확인"이 아니라
deployment/model_store.py의 전체 contract 검증(필수 파일 + metadata +
label_schema + preprocessing_config)을 통과해야 승격을 시작한다 - 단순
파일 존재만 보는 flashlight보다 한 단계 더 엄격하다.

promote_model.py는 promotion_decision.json을 읽지 않는다 (그건
run_model_update_pipeline.py가 게이트로 판단해서 호출 여부를 결정하는
역할). 이 스크립트 자체는 "이 candidate를 무조건 승격해라"라는 명령으로
취급한다 - CLI에서 직접 쓸 때는 호출 전에 promotion_decision.json의
final_decision이 'promote'인지 사람이/오케스트레이터가 먼저 확인할 것.

사용법:
    python -m context_emotion.deployment.promote_model --version v1_20260701
    python -m context_emotion.deployment.promote_model --version v1_20260701 --dry-run
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.deployment import model_store  # noqa: E402


def _log(msg: str, dry: bool = False) -> None:
    print(f"{'[DRY-RUN] ' if dry else ''}{msg}")


def _validate_onnx(onnx_path: str, metadata_path: str, contract: dict) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"[SKIP] ONNX 검증 의존성 미설치 - {e}")
        return

    metadata = model_store.load_json(metadata_path)
    output_spec = metadata.get("output_spec", {})
    num_classes = (output_spec.get("shape") or [None, None])[-1]

    onnx_contract = contract["onnx_contract"]
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}
    if onnx_contract["input_name"] not in input_names:
        raise ValueError(f"ONNX 입력 이름 불일치: expected '{onnx_contract['input_name']}', got {input_names}")

    rng = np.random.default_rng(42)
    shape = [1, 3, 224, 224]  # batch=1 dummy - real spatial dims come from metadata if it ever varies
    dummy = rng.standard_normal(shape).astype("float32")
    out = session.run([onnx_contract["output_name"]], {onnx_contract["input_name"]: dummy})[0]

    if num_classes is not None and out.shape[-1] != num_classes:
        raise ValueError(f"ONNX 출력 차원 불일치: expected {num_classes}, got {out.shape[-1]}")
    print(f"[OK] ONNX 검증 통과 - dummy output shape={out.shape}")


def _read_current_version(current_dir: str) -> str:
    meta_path = os.path.join(current_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return "unknown"
    try:
        return model_store.load_json(meta_path).get("version", "unknown")
    except Exception:
        return "unknown"


def _backup_current(current_dir: str, archive_dir: str, timestamp: str, old_version: str, dry: bool) -> str:
    archive_dst = os.path.join(archive_dir, f"{timestamp}_{old_version}")
    _log(f"백업: current/ -> archive/{os.path.basename(archive_dst)}/", dry)
    if not dry and os.path.isdir(current_dir):
        os.makedirs(archive_dir, exist_ok=True)
        shutil.copytree(current_dir, archive_dst)
    return archive_dst


def _build_staging(store_root: str, candidate_dir: str, timestamp: str, version: str, dry: bool) -> str:
    staging_dir = os.path.join(store_root, f"current_staging_{timestamp}")
    _log(f"스테이징: candidates/{version}/ -> {os.path.basename(staging_dir)}/", dry)
    if not dry:
        shutil.copytree(candidate_dir, staging_dir)
        meta_path = os.path.join(staging_dir, "metadata.json")
        meta = model_store.load_json(meta_path)
        meta["promoted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta["promoted_from_candidate"] = f"candidates/{version}"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    return staging_dir


def _atomic_swap(store_root: str, current_dir: str, staging_dir: str, dry: bool) -> None:
    """current/ -> staging/로 원자적 교체. os.rename 두 번 (current를 임시로
    빼고 staging을 current로 넣는) 방식 - flashlight promote_model.py와 동일."""
    if dry:
        _log("원자적 교체: current/ <- staging/", dry=True)
        return

    old_tmp = os.path.join(store_root, "_current_old")
    if os.path.isdir(current_dir):
        os.rename(current_dir, old_tmp)
    try:
        os.rename(staging_dir, current_dir)
    except Exception:
        if os.path.isdir(old_tmp):
            os.rename(old_tmp, current_dir)
        raise
    if os.path.isdir(old_tmp):
        shutil.rmtree(old_tmp, ignore_errors=True)


def promote(
    version: str,
    dry: bool = False,
    skip_validate: bool = False,
    store_root_override: str = None,
) -> dict:
    """전체 승격 절차. 성공 시 {"promoted": True, "version": ..., "archive_dir": ...} 반환.
    실패 시 RuntimeError/FileNotFoundError/ValueError - current는 절대 부분적으로 바뀐 채 남지 않는다."""
    contract = model_store.load_runtime_contract()
    store_root, candidates_dir, current_dir, archive_dir = model_store.resolve_store_paths(store_root_override, contract)
    cand_dir = model_store.candidate_dir(version, candidates_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  context_emotion model-store 승격")
    print("=" * 60)
    print(f"  후보 버전: {version}")
    print(f"  후보 경로: {cand_dir}")
    print(f"  current:  {current_dir}")
    if dry:
        print("  모드:      DRY-RUN (파일 변경 없음)")
    print()

    print("[1/5] 후보 contract 검증")
    problems = model_store.validate_artifact_dir(cand_dir)
    flat_problems = [p for plist in problems.values() for p in plist]
    if flat_problems:
        raise FileNotFoundError(
            f"candidate {version}이 contract을 통과하지 못함:\n  - " + "\n  - ".join(flat_problems)
        )
    print(f"  [OK] 필수 파일 {len(model_store.REQUIRED_FILES)}개 + metadata/label_schema/preprocessing_config 검증 통과")

    print("\n[2/5] ONNX 모델 검증")
    if skip_validate:
        print("  [SKIP] --skip-validate 지정")
    else:
        _validate_onnx(
            os.path.join(cand_dir, "model.onnx"),
            os.path.join(cand_dir, "metadata.json"),
            contract,
        )

    print("\n[3/5] current 백업")
    old_version = _read_current_version(current_dir)
    archive_dst = _backup_current(current_dir, archive_dir, timestamp, old_version, dry)
    if not dry:
        print(f"  [OK] archive/{os.path.basename(archive_dst)}/")

    print("\n[4/5] 스테이징 준비")
    staging_dir = os.path.join(store_root, f"current_staging_{timestamp}")
    try:
        staging_dir = _build_staging(store_root, cand_dir, timestamp, version, dry)
        if not dry:
            print(f"  [OK] {os.path.basename(staging_dir)}/ (metadata 갱신 포함)")
    except Exception as e:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"스테이징 실패 (current 유지): {e}") from e

    print("\n[5/5] current 교체")
    try:
        _atomic_swap(store_root, current_dir, staging_dir, dry)
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
    print("=" * 60)

    return {"promoted": not dry, "version": version, "archive_dir": archive_dst}


def _parse_args():
    ap = argparse.ArgumentParser(description="candidates/{version}/ -> current/ 모델 승격")
    ap.add_argument("--version", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-validate", action="store_true", help="ONNX 로딩 검증 건너뜀")
    return ap.parse_args()


def main():
    args = _parse_args()
    try:
        promote(version=args.version, dry=args.dry_run, skip_validate=args.skip_validate)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[FAILED] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
