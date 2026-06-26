#!/usr/bin/env python3
"""
captcha_bank 전체 MLOps 파이프라인 오케스트레이터 (8단계).

STEP 1/8  풀 검증            validate_captcha_pool
STEP 2/8  모델 학습          train_attack_model
STEP 3/8  보안 평가          run_attack_eval
STEP 4/8  선택지 정책 리포트 build_choice_policy_report
STEP 5/8  패키징             package_model
STEP 6/8  후보 비교          compare_candidate
STEP 7/8  승격               promote_model
STEP 8/8  스모크 테스트      smoke_test

사용법:
    # 전체 파이프라인
    python -m context_emotion.captcha_bank.scripts.run_pipeline \\
        --pool-csv /workspace/data/context_emotion/captcha_bank/captcha_pool.csv \\
        --version  v1_20260701

    # 비교 실패 시에도 강제 승격
    python -m context_emotion.captcha_bank.scripts.run_pipeline ... --force-promote

    # 드라이런 (승격 단계만 skip)
    python -m context_emotion.captcha_bank.scripts.run_pipeline ... --dry-run
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR       = Path(__file__).resolve().parent
_ML_PIPELINE_ROOT = _SCRIPT_DIR.parents[3]
_STORE            = _ML_PIPELINE_ROOT / "model-store" / "captcha_bank"
_CANDIDATES_DIR   = _STORE / "candidates"
_CURRENT_DIR      = _STORE / "current"
_DEFAULT_POLICY   = _SCRIPT_DIR.parent / "config" / "promotion_policy.yaml"

_W = 64


# ── 공통 출력 헬퍼 ──────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print()
    print("═" * _W)
    for line in title.splitlines():
        print(f"  {line}")
    print("═" * _W)
    print()


def _step(n: int, total: int, name: str, module: str) -> None:
    print()
    print(f"  ┌─ STEP {n}/{total}  {name}")
    print(f"  │  ({module})")
    print(f"  └{'─' * (_W - 4)}")
    print()


def _ok(label: str) -> None:
    print(f"\n  └─ ✓ {label}\n")


def _fail(label: str, reason: str = "") -> None:
    msg = f"  └─ ✗ {label}"
    if reason:
        msg += f": {reason}"
    print(msg, file=sys.stderr)


# ── 각 단계 실행 래퍼 ────────────────────────────────────────────────────────

def _step1_validate(pool_csv: Path, min_rows: int) -> bool:
    from context_emotion.captcha_bank.scripts.validate_captcha_pool import validate
    return validate(pool_csv, min_rows=min_rows)


def _step2_train(pool_csv: Path, model_out: Path, version: str) -> bool:
    from context_emotion.captcha_bank.training.train_attack_model import train
    try:
        train(pool_csv, model_out, version)
        return True
    except Exception as e:
        _fail("학습 실패", str(e))
        return False


def _step3_eval(pool_csv: Path, model_path: Path, eval_out: Path, version: str) -> bool:
    from context_emotion.captcha_bank.evaluation.run_attack_eval import evaluate
    try:
        evaluate(pool_csv, model_path, eval_out, version)
        return True
    except Exception as e:
        _fail("평가 실패", str(e))
        return False


def _step4_choice_report(pool_csv: Path, report_csv: Path, report_md: Path) -> bool:
    import subprocess
    cmd = [
        sys.executable, "-m", "context_emotion.captcha_bank.build_choice_policy_report",
        "--input-csv",  str(pool_csv),
        "--output-csv", str(report_csv),
        "--output-md",  str(report_md),
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def _step5_package(
    version: str,
    model_joblib: Path,
    eval_json: Path,
    pool_csv: Path,
    policy_md: Path,
) -> bool:
    from context_emotion.captcha_bank.scripts.package_model import package
    try:
        package(version, model_joblib, eval_json, pool_csv, policy_md)
        return True
    except Exception as e:
        _fail("패키징 실패", str(e))
        return False


def _step6_compare(version: str, policy_path: Path) -> bool:
    from context_emotion.captcha_bank.scripts.compare_candidate import compare
    return compare(version, policy_path)


def _step7_promote(version: str, dry: bool) -> bool:
    from context_emotion.captcha_bank.scripts.promote_model import promote
    try:
        return promote(version, dry=dry)
    except Exception as e:
        _fail("승격 실패", str(e))
        return False


def _step8_smoke(current_dir: Path) -> bool:
    from context_emotion.captcha_bank.scripts.smoke_test import smoke_test
    return smoke_test(current_dir)


# ── 메인 파이프라인 ──────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> bool:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = args.version

    _banner(
        f"captcha_bank MLOps 파이프라인\n"
        f"  버전: {version}   풀: {args.pool_csv}\n"
        f"  시작: {ts}"
    )

    workdir = Path(tempfile.mkdtemp(prefix="captcha_bank_pipeline_"))
    model_out    = workdir / "model.joblib"
    eval_out     = workdir / "evaluation_result.json"
    report_csv   = workdir / "captcha_pool_with_choices.csv"
    report_md    = workdir / "choice_policy_report.md"

    _step(1, 8, "풀 검증", "validate_captcha_pool")
    if not _step1_validate(args.pool_csv, min_rows=args.min_rows):
        _fail("STEP 1 실패 — 파이프라인 중단")
        return False
    _ok("STEP 1 완료")

    _step(2, 8, "모델 학습", "train_attack_model")
    if not _step2_train(args.pool_csv, model_out, version):
        return False
    _ok("STEP 2 완료")

    _step(3, 8, "보안 평가", "run_attack_eval")
    if not _step3_eval(args.pool_csv, model_out, eval_out, version):
        return False
    _ok("STEP 3 완료")

    _step(4, 8, "선택지 정책 리포트", "build_choice_policy_report")
    if not _step4_choice_report(args.pool_csv, report_csv, report_md):
        print("  ⚠ 선택지 정책 리포트 생성 실패 — 패키징 시 제외 후 계속 진행")
    else:
        _ok("STEP 4 완료")

    _step(5, 8, "패키징", "package_model")
    if not _step5_package(
        version, model_out, eval_out, args.pool_csv,
        report_md if report_md.exists() else None,
    ):
        return False
    _ok("STEP 5 완료")

    _step(6, 8, "후보 비교", "compare_candidate")
    compare_ok = _step6_compare(version, args.policy)
    if not compare_ok:
        if args.force_promote:
            print("  ⚠ --force-promote: 비교 실패를 무시하고 승격 진행")
        else:
            _fail("STEP 6 실패 — 승격 거부 (--force-promote 로 우회 가능)")
            return False
    else:
        _ok("STEP 6 완료")

    _step(7, 8, "승격", "promote_model")
    if not _step7_promote(version, dry=args.dry_run):
        return False
    _ok("STEP 7 완료" + (" [DRY-RUN]" if args.dry_run else ""))

    _step(8, 8, "스모크 테스트", "smoke_test")
    if not _step8_smoke(_CURRENT_DIR):
        _fail("STEP 8 실패 — current/ 모델이 올바르지 않습니다")
        return False
    _ok("STEP 8 완료")

    _banner(
        f"파이프라인 완료  ✓\n"
        f"  버전: {version}\n"
        f"  model-store: {_CURRENT_DIR}"
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="captcha_bank MLOps 파이프라인 (8단계)")
    ap.add_argument("--pool-csv",      type=Path, required=True,
                    help="export_captcha_pool.py 출력 CSV")
    ap.add_argument("--version",       required=True,
                    help="모델 버전 (예: v1_20260701)")
    ap.add_argument("--policy",        type=Path, default=_DEFAULT_POLICY,
                    help="승격 정책 YAML")
    ap.add_argument("--min-rows",      type=int, default=200,
                    help="풀 최소 행 수 (기본: 200)")
    ap.add_argument("--dry-run",       action="store_true",
                    help="승격 단계를 실행하지 않음")
    ap.add_argument("--force-promote", action="store_true",
                    help="비교 게이트 실패 시에도 승격 강제 진행")
    args = ap.parse_args()

    ok = run_pipeline(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
