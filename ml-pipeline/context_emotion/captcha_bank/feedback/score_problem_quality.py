#!/usr/bin/env python3
"""
집계 통계(aggregated_stats.csv) → 문제별 품질 레이블·상태 CSV.

품질 레이블 (우선순위 순):
  1. insufficient_data  — attempt_count < min_attempts_per_problem
  2. attack_exposed     — suspicious_rate > max_suspicious_rate           → RETIRED
  3. ambiguous          — human_pass_rate 범위 초과 (너무 쉽거나 너무 어려움)
  4. ux_poor            — avg_solve_time_ms > max_avg_solve_time_ms
  5. confusing          — retry_rate > max_retry_rate
  6. robust             — human_pass_rate ≥ min_robust_pass_rate AND suspicious_rate ≤ max_robust_suspicious_rate
  7. normal             — 나머지

상태 (status):
  ACTIVE  ← robust, normal, insufficient_data
  REVIEW  ← ambiguous, ux_poor, confusing
  RETIRED ← attack_exposed

사용법:
    python -m context_emotion.captcha_bank.feedback.score_problem_quality \\
        --agg-csv  /workdir/aggregated_stats.csv \\
        --policy   captcha_bank/config/feedback_quality_policy.yaml \\
        --output   /workdir/quality_scores.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .log_schema import QUALITY_COLUMNS

logger = logging.getLogger(__name__)

_DEFAULT_POLICY = (
    Path(__file__).resolve().parents[1] / "config" / "feedback_quality_policy.yaml"
)
_W = 64

# ── 기본 임계값 (policy YAML에서 덮어씀) ─────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "min_attempts_per_problem":  30,
    "min_human_pass_rate":       0.55,
    "max_human_pass_rate":       0.92,
    "max_suspicious_rate":       0.15,
    "max_avg_solve_time_ms":     45_000,
    "max_retry_rate":            0.40,
    "min_robust_pass_rate":      0.65,
    "max_robust_suspicious_rate": 0.08,
}


def _load_policy(path: Path) -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if path.exists():
        raw = yaml.safe_load(path.read_text())
        cfg.update((raw or {}).get("thresholds", {}))
    else:
        logger.warning("policy 파일 없음, 기본값 사용: %s", path)
    return cfg


# ── 단일 행 품질 판정 ─────────────────────────────────────────────────────────

def _score_row(row: dict, p: dict) -> tuple[str, str, float, str]:
    """(quality_label, status, confidence, score_note) 반환."""
    n    = int(row.get("attempt_count", 0))
    hpr  = float(row.get("human_pass_rate", 0.0))
    sr   = float(row.get("suspicious_rate", 0.0))
    avg_t = float(row.get("avg_solve_time_ms", 0.0))
    rr   = float(row.get("retry_rate", 0.0))

    # 신뢰도: attempt_count / min_attempts 를 [0, 1] 클램프
    confidence = min(1.0, n / max(1, p["min_attempts_per_problem"]))

    # ── 우선순위 순 판정 ──────────────────────────────────────────────────────

    if n < p["min_attempts_per_problem"]:
        return (
            "insufficient_data", "ACTIVE",
            round(confidence, 3),
            f"attempt {n} < min {p['min_attempts_per_problem']}",
        )

    if sr > p["max_suspicious_rate"]:
        return (
            "attack_exposed", "RETIRED",
            round(confidence, 3),
            f"suspicious_rate {sr:.3f} > max {p['max_suspicious_rate']}",
        )

    if hpr < p["min_human_pass_rate"]:
        return (
            "ambiguous", "REVIEW",
            round(confidence, 3),
            f"human_pass_rate {hpr:.3f} < min {p['min_human_pass_rate']} (너무 어려움)",
        )

    if hpr > p["max_human_pass_rate"]:
        return (
            "ambiguous", "REVIEW",
            round(confidence, 3),
            f"human_pass_rate {hpr:.3f} > max {p['max_human_pass_rate']} (너무 쉬움)",
        )

    if avg_t > p["max_avg_solve_time_ms"]:
        return (
            "ux_poor", "REVIEW",
            round(confidence, 3),
            f"avg_solve_time {avg_t:.0f}ms > max {p['max_avg_solve_time_ms']}ms",
        )

    if rr > p["max_retry_rate"]:
        return (
            "confusing", "REVIEW",
            round(confidence, 3),
            f"retry_rate {rr:.3f} > max {p['max_retry_rate']}",
        )

    if (
        hpr >= p["min_robust_pass_rate"]
        and sr <= p["max_robust_suspicious_rate"]
    ):
        return (
            "robust", "ACTIVE",
            round(confidence, 3),
            f"pass_rate={hpr:.3f} suspicious={sr:.3f}",
        )

    return (
        "normal", "ACTIVE",
        round(confidence, 3),
        f"pass_rate={hpr:.3f} suspicious={sr:.3f} retry={rr:.3f}",
    )


# ── 전체 스코어링 ─────────────────────────────────────────────────────────────

def score(agg_df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """집계 DataFrame → 품질 점수 DataFrame."""
    rows = []
    for _, row in agg_df.iterrows():
        label, status, conf, note = _score_row(row.to_dict(), policy)
        rows.append({
            "sample_id":        row["sample_id"],
            "quality_label":    label,
            "status":           status,
            "confidence":       conf,
            "human_pass_rate":  row.get("human_pass_rate", 0.0),
            "suspicious_rate":  row.get("suspicious_rate", 0.0),
            "avg_solve_time_ms": row.get("avg_solve_time_ms", 0.0),
            "retry_rate":       row.get("retry_rate", 0.0),
            "attempt_count":    row.get("attempt_count", 0),
            "score_note":       note,
        })

    if not rows:
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    return pd.DataFrame(rows)[QUALITY_COLUMNS]


# ── 리포트 출력 ───────────────────────────────────────────────────────────────

def _print_report(df: pd.DataFrame) -> None:
    label_counts = df["quality_label"].value_counts()
    status_counts = df["status"].value_counts()

    print(f"\n  {'품질 레이블 분포':─<50}")
    for label, cnt in label_counts.items():
        pct = 100 * cnt / len(df)
        bar = "█" * int(pct / 2)
        print(f"    {label:<20} {cnt:>4}개  {pct:5.1f}%  {bar}")

    print(f"\n  {'상태 분포':─<50}")
    for st, cnt in status_counts.items():
        print(f"    {st:<10} {cnt:>4}개")

    retired = df[df["status"] == "RETIRED"]
    if not retired.empty:
        print(f"\n  [주의] RETIRED 문제 {len(retired)}개 — 풀에서 제거 예정:")
        for _, r in retired.head(5).iterrows():
            print(f"    {r['sample_id']}  ({r['score_note']})")
        if len(retired) > 5:
            print(f"    ... 외 {len(retired)-5}개")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="문제별 품질 스코어링")
    ap.add_argument("--agg-csv", type=Path, required=True,
                    help="aggregate_attempt_logs 출력 CSV")
    ap.add_argument("--policy",  type=Path, default=_DEFAULT_POLICY,
                    help="feedback_quality_policy.yaml 경로")
    ap.add_argument("--output",  type=Path, required=True,
                    help="품질 점수 CSV 출력 경로")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print()
    print("═" * _W)
    print("  SCORE PROBLEM QUALITY")
    print("═" * _W)

    if not args.agg_csv.exists():
        print(f"  [오류] 집계 CSV 없음: {args.agg_csv}", file=sys.stderr)
        return 1

    agg_df = pd.read_csv(args.agg_csv)
    policy = _load_policy(args.policy)
    print(f"  집계된 문제 수: {len(agg_df):,}개")
    print(f"  정책 파일:      {args.policy}")

    scored_df = score(agg_df, policy)
    _print_report(scored_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(args.output, index=False)
    print(f"\n  출력 파일: {args.output}")
    print("═" * _W)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
