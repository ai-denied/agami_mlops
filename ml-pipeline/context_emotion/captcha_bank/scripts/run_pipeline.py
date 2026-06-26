#!/usr/bin/env python3
"""
captcha_bank MLOps 파이프라인 오케스트레이터.

── train 모드 (기본) ──────────────────────────────────────────────────────────
STEP 1/8  풀 검증            validate_captcha_pool
STEP 2/8  모델 학습          train_attack_model
STEP 3/8  보안 평가          run_attack_eval
STEP 4/8  선택지 정책 리포트 build_choice_policy_report
STEP 5/8  패키징             package_model
STEP 6/8  후보 비교          compare_candidate
STEP 7/8  승격               promote_model
STEP 8/8  스모크 테스트      smoke_test

── feedback 모드 (--mode feedback) ────────────────────────────────────────────
STEP F1/4  트리거 조건 확인  trigger_policy
STEP F2/4  attempt 로그 집계 aggregate_attempt_logs
STEP F3/4  품질 스코어링     score_problem_quality
STEP F4/4  candidate pool 빌드 build_candidate_pool_from_feedback
→ 이후 train 모드 STEP 1~8 자동 실행 (pool-csv = F4 출력)

사용법:
    # train 모드 (기본)
    python -m context_emotion.captcha_bank.scripts.run_pipeline \\
        --pool-csv /data/captcha_pool.csv --version v1_20260701

    # feedback 모드 (attempt_logs → 자동 pool 갱신 → 평가/승격)
    python -m context_emotion.captcha_bank.scripts.run_pipeline \\
        --mode feedback \\
        --log-dir /data/context_emotion/attempt_logs

    # 공통 옵션
    --force-promote   비교 게이트 실패 시에도 강제 승격
    --dry-run         승격 단계만 skip
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR            = Path(__file__).resolve().parent
_ML_PIPELINE_ROOT      = _SCRIPT_DIR.parents[3]
_STORE                 = _ML_PIPELINE_ROOT / "model-store" / "captcha_bank"
_CANDIDATES_DIR        = _STORE / "candidates"
_CURRENT_DIR           = _STORE / "current"
_DEFAULT_POLICY        = _SCRIPT_DIR.parent / "config" / "promotion_policy.yaml"
_DEFAULT_QUALITY_POLICY = _SCRIPT_DIR.parent / "config" / "feedback_quality_policy.yaml"
_DEFAULT_TRIGGER_POLICY = _SCRIPT_DIR.parent / "config" / "feedback_trigger_policy.yaml"
_DEFAULT_LOG_DIR       = Path("/data/context_emotion/attempt_logs")

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


# ── feedback 단계 실행 래퍼 ─────────────────────────────────────────────────

def _stepf1_trigger(log_dir: Path, model_dir: Path, policy: Path, force: bool) -> bool:
    """트리거 조건 확인. 미충족 시 False 반환 (파이프라인 skip)."""
    from context_emotion.captcha_bank.feedback.trigger_policy import (
        evaluate, _load_policy,
    )
    p = _load_policy(policy)
    if force:
        print("  --force-promote/feedback: 트리거 게이트 건너뜀")
        return True
    should_run, gates = evaluate(
        log_dir=log_dir,
        model_dir=model_dir,
        quality_csv=None,
        policy=p,
    )
    for g in gates:
        print(repr(g))
    return should_run


def _stepf2_aggregate(log_dir: Path, output: Path, since_days: int) -> bool:
    from context_emotion.captcha_bank.feedback.aggregate_attempt_logs import main as agg_main
    rc = agg_main([
        "--log-dir",    str(log_dir),
        "--since-days", str(since_days),
        "--output",     str(output),
    ])
    return rc == 0


def _stepf3_score(agg_csv: Path, output: Path, policy: Path) -> bool:
    from context_emotion.captcha_bank.feedback.score_problem_quality import main as score_main
    rc = score_main([
        "--agg-csv", str(agg_csv),
        "--policy",  str(policy),
        "--output",  str(output),
    ])
    return rc == 0


def _stepf4_build_pool(
    current_pool: Path,
    quality_csv: Path,
    output: Path,
    new_problems: Path | None,
) -> bool:
    from context_emotion.captcha_bank.feedback.build_candidate_pool_from_feedback import main as build_main
    cmd = [
        "--current-pool", str(current_pool),
        "--quality-csv",  str(quality_csv),
        "--output",       str(output),
    ]
    if new_problems and new_problems.exists():
        cmd += ["--new-problems", str(new_problems)]
    rc = build_main(cmd)
    return rc == 0


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


# ── feedback 전처리 파이프라인 ────────────────────────────────────────────────

def run_feedback_preprocess(args: argparse.Namespace, workdir: Path) -> Path | None:
    """F1~F4 실행 후 생성된 feedback_pool.csv 경로를 반환. skip 또는 실패 시 None."""

    _banner(
        f"captcha_bank Feedback MLOps\n"
        f"  로그: {args.log_dir}\n"
        f"  시작: {datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    TOTAL_F = 4

    _step(1, TOTAL_F, "트리거 조건 확인", "trigger_policy")
    trigger_ok = _stepf1_trigger(
        log_dir=args.log_dir,
        model_dir=_CURRENT_DIR,
        policy=getattr(args, "trigger_policy", _DEFAULT_TRIGGER_POLICY),
        force=getattr(args, "force_promote", False) or getattr(args, "skip_trigger", False),
    )
    if not trigger_ok:
        print("\n  ─── 트리거 조건 미충족 — 파이프라인 skip ───\n")
        return None
    _ok("STEP F1 완료")

    agg_csv     = workdir / "aggregated_stats.csv"
    quality_csv = workdir / "quality_scores.csv"
    feedback_pool = workdir / "feedback_pool.csv"

    _step(2, TOTAL_F, "attempt 로그 집계", "aggregate_attempt_logs")
    since_days = getattr(args, "since_days", 30)
    if not _stepf2_aggregate(args.log_dir, agg_csv, since_days):
        _fail("STEP F2 실패")
        return None
    _ok("STEP F2 완료")

    _step(3, TOTAL_F, "품질 스코어링", "score_problem_quality")
    quality_policy = getattr(args, "quality_policy", _DEFAULT_QUALITY_POLICY)
    if not _stepf3_score(agg_csv, quality_csv, quality_policy):
        _fail("STEP F3 실패")
        return None
    _ok("STEP F3 완료")

    _step(4, TOTAL_F, "candidate pool 빌드", "build_candidate_pool_from_feedback")
    current_pool = _CURRENT_DIR / "captcha_pool.csv"
    if not current_pool.exists():
        _fail("current pool CSV 없음", str(current_pool))
        return None
    new_problems = getattr(args, "new_problems", None)
    if not _stepf4_build_pool(current_pool, quality_csv, feedback_pool, new_problems):
        _fail("STEP F4 실패")
        return None
    _ok("STEP F4 완료")

    return feedback_pool


# ── 메인 파이프라인 ──────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> bool:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = args.version or f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    # ── feedback 모드: F1~F4 먼저 실행 후 pool_csv 갱신 ─────────────────────
    mode = getattr(args, "mode", "train")
    workdir = Path(tempfile.mkdtemp(prefix="captcha_bank_pipeline_"))

    if mode == "feedback":
        feedback_pool = run_feedback_preprocess(args, workdir)
        if feedback_pool is None:
            return True  # skip은 성공 (트리거 미충족)
        args.pool_csv = feedback_pool
        print(f"\n  ─── feedback pool → train 파이프라인으로 이어집니다 ───\n")

    _banner(
        f"captcha_bank MLOps 파이프라인\n"
        f"  버전: {version}   풀: {args.pool_csv}\n"
        f"  시작: {ts}"
    )
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
    ap = argparse.ArgumentParser(
        description="captcha_bank MLOps 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 공통 ──────────────────────────────────────────────────────────────────
    ap.add_argument("--mode",          choices=["train", "feedback"], default="train",
                    help="train: 풀 CSV 직접 지정 | feedback: attempt_logs → 자동 pool 갱신")
    ap.add_argument("--version",       default="",
                    help="모델 버전 (예: v1_20260701). 생략 시 날짜 기반 자동 생성")
    ap.add_argument("--policy",        type=Path, default=_DEFAULT_POLICY,
                    help="승격 정책 YAML")
    ap.add_argument("--min-rows",      type=int, default=200,
                    help="풀 최소 행 수 (기본: 200)")
    ap.add_argument("--dry-run",       action="store_true",
                    help="승격 단계를 실행하지 않음")
    ap.add_argument("--force-promote", action="store_true",
                    help="비교 게이트 실패 시에도 승격 강제 진행")

    # ── train 모드 전용 ────────────────────────────────────────────────────────
    train_grp = ap.add_argument_group("train 모드")
    train_grp.add_argument("--pool-csv", type=Path, default=None,
                           help="export_captcha_pool.py 출력 CSV (train 모드 필수)")

    # ── feedback 모드 전용 ────────────────────────────────────────────────────
    fb_grp = ap.add_argument_group("feedback 모드")
    fb_grp.add_argument("--log-dir",        type=Path, default=_DEFAULT_LOG_DIR,
                        help="attempt log 디렉토리 (기본: /data/context_emotion/attempt_logs)")
    fb_grp.add_argument("--since-days",     type=int,  default=30,
                        help="집계 기간 일수 (기본: 30)")
    fb_grp.add_argument("--quality-policy", type=Path, default=_DEFAULT_QUALITY_POLICY,
                        help="feedback_quality_policy.yaml 경로")
    fb_grp.add_argument("--trigger-policy", type=Path, default=_DEFAULT_TRIGGER_POLICY,
                        help="feedback_trigger_policy.yaml 경로")
    fb_grp.add_argument("--new-problems",   type=Path, default=None,
                        help="신규 검수 완료 문제 CSV (선택)")
    fb_grp.add_argument("--skip-trigger",   action="store_true",
                        help="트리거 조건 무시하고 강제 실행 (--force-promote 와 같은 효과)")

    args = ap.parse_args()

    # ── 검증 ─────────────────────────────────────────────────────────────────
    if args.mode == "train" and not args.pool_csv:
        ap.error("--mode train 에는 --pool-csv 가 필요합니다.")

    ok = run_pipeline(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
