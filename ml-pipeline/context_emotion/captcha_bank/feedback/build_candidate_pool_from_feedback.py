#!/usr/bin/env python3
"""
current captcha_pool.csv + quality_scores.csv → 피드백 기반 candidate pool CSV.

처리 방식:
  RETIRED     → 풀에서 제거 (attack_exposed)
  REVIEW      → 유지하되 pool_status=REVIEW (인간 검수 대기)
  ACTIVE      → 현재 상태 유지
  신규 문제   → --new-problems CSV 에서 pool_status=CANDIDATE 로 추가

출력 컬럼은 입력 captcha_pool.csv 의 모든 컬럼을 유지하며
다음 품질 메타데이터 컬럼을 추가/갱신한다:
  pool_status       ACTIVE / REVIEW / RETIRED / CANDIDATE
  quality_label     robust / normal / ambiguous / ux_poor / attack_exposed / confusing / insufficient_data / unknown
  quality_confidence 0.0~1.0
  human_pass_rate_obs (관측값 — 모델 학습 입력과 구분)
  last_quality_update ISO 8601 UTC

사용법:
    python -m context_emotion.captcha_bank.feedback.build_candidate_pool_from_feedback \\
        --current-pool  model-store/captcha_bank/current/captcha_pool.csv \\
        --quality-csv   /workdir/quality_scores.csv \\
        --output        /workdir/feedback_pool.csv \\
        [--new-problems /path/to/new_problems.csv]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_STORE = Path(__file__).resolve().parents[4] / "model-store" / "captcha_bank"
_DEFAULT_CURRENT_POOL = _STORE / "current" / "captcha_pool.csv"

_W = 64
_QUALITY_META_COLS = [
    "pool_status",
    "quality_label",
    "quality_confidence",
    "human_pass_rate_obs",
    "last_quality_update",
]


# ── 빌드 ─────────────────────────────────────────────────────────────────────

def build(
    current_pool_path: Path,
    quality_path: Path,
    output_path: Path,
    new_problems_path: Path | None = None,
) -> pd.DataFrame:
    """피드백 기반 candidate pool DataFrame을 반환하고 output_path에 저장한다."""

    # ── current pool 로드 ────────────────────────────────────────────────────
    pool_df = pd.read_csv(current_pool_path)
    print(f"  current pool: {len(pool_df):,}개")

    # ── 품질 점수 로드 ───────────────────────────────────────────────────────
    quality_df = pd.read_csv(quality_path)
    quality_df = quality_df.rename(columns={
        "human_pass_rate": "human_pass_rate_obs",
    })
    quality_df = quality_df[
        ["sample_id", "quality_label", "status", "confidence", "human_pass_rate_obs"]
    ].rename(columns={"status": "_feedback_status", "confidence": "quality_confidence"})
    print(f"  quality scores: {len(quality_df):,}개 문제")

    # ── 조인 ─────────────────────────────────────────────────────────────────
    merged = pool_df.merge(quality_df, on="sample_id", how="left")

    # 품질 데이터 없는 문제 (새 풀에만 있는 문제) → insufficient_data
    no_data_mask = merged["quality_label"].isna()
    merged.loc[no_data_mask, "quality_label"]       = "insufficient_data"
    merged.loc[no_data_mask, "_feedback_status"]    = "ACTIVE"
    merged.loc[no_data_mask, "quality_confidence"]  = 0.0
    merged.loc[no_data_mask, "human_pass_rate_obs"] = float("nan")

    # pool_status: 기존 값 유지하되 피드백 결과로 덮어씀
    if "pool_status" not in merged.columns:
        merged["pool_status"] = "ACTIVE"
    merged["pool_status"] = merged["_feedback_status"].fillna(merged["pool_status"])
    merged = merged.drop(columns=["_feedback_status"])

    # last_quality_update 갱신
    now_iso = datetime.now(timezone.utc).isoformat()
    merged["last_quality_update"] = now_iso

    # ── RETIRED 제거 ──────────────────────────────────────────────────────────
    retired_mask = merged["pool_status"] == "RETIRED"
    retired_count = retired_mask.sum()
    if retired_count:
        retired_ids = merged.loc[retired_mask, "sample_id"].tolist()
        logger.info("RETIRED 제거 (%d개): %s...", retired_count, retired_ids[:5])
    merged = merged[~retired_mask].reset_index(drop=True)
    print(f"  RETIRED 제거:   {retired_count:,}개")

    # ── 신규 문제 추가 ────────────────────────────────────────────────────────
    new_count = 0
    if new_problems_path and new_problems_path.exists():
        new_df = pd.read_csv(new_problems_path)
        # 이미 있는 sample_id 제외 (중복 방지)
        existing_ids = set(merged["sample_id"].tolist())
        new_df = new_df[~new_df["sample_id"].isin(existing_ids)].copy()

        new_df["pool_status"]           = "CANDIDATE"
        new_df["quality_label"]         = "insufficient_data"
        new_df["quality_confidence"]    = 0.0
        new_df["human_pass_rate_obs"]   = float("nan")
        new_df["last_quality_update"]   = now_iso
        new_count = len(new_df)

        merged = pd.concat([merged, new_df], ignore_index=True)
        print(f"  신규 CANDIDATE: {new_count:,}개 추가")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    return merged


# ── 리포트 출력 ───────────────────────────────────────────────────────────────

def _print_report(df: pd.DataFrame, original_count: int, retired_count: int) -> None:
    status_counts = df["pool_status"].value_counts()
    label_counts  = df["quality_label"].value_counts()

    print(f"\n  pool_status 분포:")
    for st, cnt in status_counts.items():
        pct = 100 * cnt / len(df)
        print(f"    {st:<12} {cnt:>4}개  ({pct:.1f}%)")

    print(f"\n  quality_label 분포:")
    for lbl, cnt in label_counts.items():
        pct = 100 * cnt / len(df)
        print(f"    {lbl:<22} {cnt:>4}개  ({pct:.1f}%)")

    print(f"\n  요약:")
    print(f"    original:  {original_count:,}개")
    print(f"    retired:   {retired_count:,}개 제거")
    print(f"    candidate: {len(df):,}개 (최종)")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="피드백 기반 candidate pool 생성")
    ap.add_argument("--current-pool",  type=Path, default=_DEFAULT_CURRENT_POOL,
                    help="현재 배포된 captcha_pool.csv")
    ap.add_argument("--quality-csv",   type=Path, required=True,
                    help="score_problem_quality 출력 CSV")
    ap.add_argument("--output",        type=Path, required=True,
                    help="피드백 candidate pool CSV 출력 경로")
    ap.add_argument("--new-problems",  type=Path, default=None,
                    help="신규 검수 완료 문제 CSV (선택)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print()
    print("═" * _W)
    print("  BUILD CANDIDATE POOL FROM FEEDBACK")
    print("═" * _W)

    for p, label in [(args.current_pool, "current pool"), (args.quality_csv, "quality CSV")]:
        if not p.exists():
            print(f"  [오류] 파일 없음: {p} ({label})", file=sys.stderr)
            return 1

    original_df = pd.read_csv(args.current_pool)
    original_count = len(original_df)

    quality_df_raw = pd.read_csv(args.quality_csv)
    retired_count = int((quality_df_raw["status"] == "RETIRED").sum())

    result_df = build(
        current_pool_path=args.current_pool,
        quality_path=args.quality_csv,
        output_path=args.output,
        new_problems_path=args.new_problems,
    )

    _print_report(result_df, original_count, retired_count)
    print(f"\n  출력 파일: {args.output}")
    print("═" * _W)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
