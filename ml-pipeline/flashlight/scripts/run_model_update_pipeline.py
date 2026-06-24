#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flashlight model-store 업데이트 파이프라인

package → compare → promote 를 순서대로 실행한다.
compare가 FAIL이면 promote는 실행하지 않는다.

사용법:
  # 전체 파이프라인 실행
  python -m flashlight.scripts.run_model_update_pipeline \\
    --version v4_20260610 \\
    --onnx       ./runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.onnx \\
    --normalizer ./runs/mouse_gru_final_v3_policy_tuned/mouse_normalizer_server_final_v2.joblib \\
    --metadata   ./runs/mouse_gru_final_v3_policy_tuned/mouse_metadata_server_final_v2.json

  # dry-run: package/compare 는 실제 실행, promote는 dry-run
  python -m flashlight.scripts.run_model_update_pipeline \\
    --version v4_20260610 \\
    --onnx ... --normalizer ... --metadata ... \\
    --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# 각 스크립트의 핵심 함수만 직접 임포트한다.
# main()은 sys.exit()를 호출하므로 사용하지 않는다.
from flashlight.scripts.package_for_captcha_engine import (
    package,
    validate,
    _print_summary,
    _CANDIDATES_DIR as _PKG_CANDIDATES_DIR,
)
from flashlight.scripts.compare_candidate import (
    _load_metadata,
    _build_rows,
    _overall_pass,
    print_comparison,
    print_contract_check,
    run_contract_check,
    _CANDIDATES_DIR as _CMP_CANDIDATES_DIR,
    _CURRENT_DIR    as _CMP_CURRENT_DIR,
)
from flashlight.scripts.promote_model import (
    promote,
)

_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))


# ---------------------------------------------------------------------------
# 로깅 유틸
# ---------------------------------------------------------------------------

_W = 60

def _banner(title: str) -> None:
    print("═" * _W)
    print(f"  {title}")
    print("═" * _W)


def _step_header(n: int, total: int, name: str, tag: str = "") -> None:
    tag_str = f"  [{tag}]" if tag else ""
    print()
    print(f"┌─ STEP {n}/{total}  {name}{tag_str}")
    print(f"│{'─' * (_W - 2)}")


def _step_ok(n: int, total: int, detail: str = "", elapsed: float = 0.0) -> None:
    time_str = f"  ({elapsed:.2f}s)" if elapsed else ""
    suffix   = f" — {detail}" if detail else ""
    print(f"└─ STEP {n}/{total}  ✓ 완료{suffix}{time_str}")


def _step_fail(n: int, total: int, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"└─ STEP {n}/{total}  ✗ 실패{suffix}")


def _step_skip(n: int, total: int, reason: str) -> None:
    print(f"└─ STEP {n}/{total}  ─ 건너뜀  ({reason})")


# ---------------------------------------------------------------------------
# 각 단계
# ---------------------------------------------------------------------------

def run_package(version: str, onnx: str, normalizer: str, metadata: str) -> bool:
    """
    package_for_captcha_engine.package() 실행.
    성공 시 True, 실패 시 False.
    """
    output_dir = os.path.join(_PKG_CANDIDATES_DIR, version)
    try:
        onnx_dst, normalizer_dst, metadata_dst = package(
            onnx_src=onnx,
            normalizer_src=normalizer,
            metadata_src=metadata,
            output_dir=output_dir,
            version=version,
        )
        _print_summary(normalizer_dst, metadata_dst)
        validate(onnx_dst, normalizer_dst)
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return False


def run_compare(version: str) -> bool:
    """
    ONNX runtime contract 검증 + compare_candidate 실행.
    contract가 FAIL이면 성능 지표를 보지도 않고 즉시 False - 텐서 이름/shape가
    바뀐 모델은 성능이 아무리 좋아도 승격 후보가 될 수 없다.
    """
    contract_passed, contract_problems = run_contract_check(version)
    print_contract_check(version, contract_passed, contract_problems)
    if not contract_passed:
        return False

    current_meta_path   = os.path.join(_CMP_CURRENT_DIR,   "metadata.json")
    candidate_meta_path = os.path.join(_CMP_CANDIDATES_DIR, version, "metadata.json")
    try:
        current_meta   = _load_metadata(current_meta_path,   "current")
        candidate_meta = _load_metadata(candidate_meta_path, f"candidates/{version}")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return False

    current_perf   = current_meta.get("performance", {})
    candidate_perf = candidate_meta.get("performance", {})

    rows   = _build_rows(current_perf, candidate_perf)
    passed = _overall_pass(rows)

    print_comparison(current_meta, candidate_meta, rows, passed)
    return passed


def run_promote(version: str, dry_run: bool) -> bool:
    """
    promote_model.promote() 실행.
    성공 시 True, 실패 시 False.
    """
    try:
        # skip_validate=False — run_compare()가 이미 contract+성능을 검증했지만,
        # promote_model.py 자체의 ONNX 로딩 검증도 마지막 방어선으로 항상 실행한다.
        promote(version=version, dry=dry_run, skip_validate=False)
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 파이프라인 오케스트레이터
# ---------------------------------------------------------------------------

def run_pipeline(args) -> bool:
    """
    3단계 파이프라인을 순서대로 실행한다.
    전체 성공 시 True 반환.
    """
    dry_tag = "DRY-RUN" if args.dry_run else ""

    _banner(
        f"flashlight model-store 업데이트 파이프라인\n"
        f"  버전: {args.version}"
        + (f"  [{dry_tag}]" if dry_tag else "")
    )

    results: dict[str, bool | None] = {
        "package": None,
        "compare": None,
        "promote": None,
    }
    pipeline_start = time.monotonic()

    # ── STEP 1: package ────────────────────────────────────────────────────
    _step_header(1, 3, "패키징", "package_for_captcha_engine")
    t0 = time.monotonic()
    ok = run_package(
        version=args.version,
        onnx=args.onnx,
        normalizer=args.normalizer,
        metadata=args.metadata,
    )
    elapsed = time.monotonic() - t0
    results["package"] = ok

    if ok:
        _step_ok(1, 3, elapsed=elapsed)
    else:
        _step_fail(1, 3, "패키징 실패 — 이후 단계를 중단합니다")
        _print_final(results, time.monotonic() - pipeline_start)
        return False

    # ── STEP 2: compare ────────────────────────────────────────────────────
    _step_header(2, 3, "성능 비교", "compare_candidate")
    t0 = time.monotonic()
    ok = run_compare(version=args.version)
    elapsed = time.monotonic() - t0
    results["compare"] = ok

    if ok:
        _step_ok(2, 3, "PASS — 승격 조건 충족", elapsed)
    else:
        _step_fail(2, 3, "FAIL — 승격 조건 미충족")
        _step_skip(3, 3, "compare FAIL로 인한 중단")
        _print_final(results, time.monotonic() - pipeline_start)
        return False

    # ── STEP 3: promote ────────────────────────────────────────────────────
    promote_tag = "DRY-RUN" if args.dry_run else ""
    _step_header(3, 3, "승격", f"promote_model{('  [' + promote_tag + ']') if promote_tag else ''}")
    t0 = time.monotonic()
    ok = run_promote(version=args.version, dry_run=args.dry_run)
    elapsed = time.monotonic() - t0
    results["promote"] = ok

    if ok:
        detail = "DRY-RUN 완료" if args.dry_run else f"current 버전 → {args.version}"
        _step_ok(3, 3, detail, elapsed)
    else:
        _step_fail(3, 3, "승격 실패")
        _print_final(results, time.monotonic() - pipeline_start)
        return False

    _print_final(results, time.monotonic() - pipeline_start)
    return True


def _print_final(results: dict, total_elapsed: float) -> None:
    all_done = all(v is True for v in results.values() if v is not None)
    verdict  = "SUCCESS" if all_done else "FAILED"

    print()
    print("═" * _W)
    print(f"  단계별 결과:")

    icons = {True: "✓", False: "✗", None: "─"}
    names = {"package": "패키징  (package_for_captcha_engine)",
             "compare": "비교     (compare_candidate)",
             "promote": "승격     (promote_model)"}
    for key, name in names.items():
        val  = results.get(key)
        icon = icons[val]
        skipped = "  [건너뜀]" if val is None else ""
        print(f"    {icon}  {name}{skipped}")

    print()
    print(f"  총 소요시간: {total_elapsed:.2f}s")
    print()
    if verdict == "SUCCESS":
        print(f"  최종 결과:  {verdict}")
    else:
        print(f"  최종 결과:  {verdict}")
    print("═" * _W)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="package → compare → promote 파이프라인"
    )
    parser.add_argument(
        "--version",
        required=True,
        help="모델 버전명 (예: v4_20260610). candidates/{version}/ 에 저장됩니다.",
    )
    parser.add_argument(
        "--onnx",
        default="./runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.onnx",
        help="입력 ONNX 경로",
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
        "--dry-run",
        action="store_true",
        help="package/compare는 실제 실행, promote는 dry-run으로만 수행",
    )
    return parser.parse_args()


def main():
    args   = _parse_args()
    passed = run_pipeline(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
