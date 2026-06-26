#!/usr/bin/env python3
"""
feedback MLOps 트리거 조건 평가.

트리거 결정 (모든 활성 조건이 True일 때 실행):
  1. new_attempts    ≥ min_new_attempts             (기본: 1000건)
  2. avg_per_problem ≥ min_avg_attempts_per_problem  (기본: 30건)
  3. days_since_last ≥ min_days_since_last_promote   (기본: 7일)
  4. (선택) bad_problem_ratio ≥ min_bad_problem_ratio (기본: 0.10)
     — quality_scores.csv 가 있을 때만 평가

종료 코드:
  0 — 트리거 (파이프라인 실행 권장)
  1 — 조건 미충족 (skip)
  2 — 오류

/tmp/should_run.txt 에 "true" / "false" 기록 (Argo output parameter 용).

사용법:
    python -m context_emotion.captcha_bank.feedback.trigger_policy \\
        --log-dir    /data/context_emotion/attempt_logs \\
        --model-dir  model-store/captcha_bank/current \\
        [--quality-csv /workdir/quality_scores.csv]   \\
        [--policy     captcha_bank/config/feedback_trigger_policy.yaml] \\
        [--output     /tmp/should_run.txt]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[4] / "model-store" / "captcha_bank" / "current"
)
_DEFAULT_LOG_DIR = Path("/data/context_emotion/attempt_logs")
_DEFAULT_POLICY = (
    Path(__file__).resolve().parents[1] / "config" / "feedback_trigger_policy.yaml"
)
_W = 64

_DEFAULTS: dict[str, Any] = {
    "min_new_attempts":             1000,
    "min_avg_attempts_per_problem": 30,
    "min_bad_problem_ratio":        0.10,
    "min_days_since_last_promote":  7,
    "require_quality_check":        False,
}


def _load_policy(path: Path) -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if path.exists():
        raw = yaml.safe_load(path.read_text())
        cfg.update((raw or {}).get("trigger", {}))
    return cfg


def _promoted_at(model_dir: Path) -> datetime | None:
    """current/metadata.json 에서 promoted_at 읽기. 없으면 None."""
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        ts = meta.get("promoted_at") or meta.get("trained_at")
        if ts:
            dt = datetime.fromisoformat(ts)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _count_new_attempts(log_dir: Path, since: datetime) -> int:
    """since 이후의 attempt 건수를 빠르게 집계한다 (줄 수 카운트)."""
    total = 0
    for log_file in sorted(log_dir.glob("attempts_*.jsonl")):
        try:
            date_str = log_file.stem.replace("attempts_", "")
            file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            if file_date < since - timedelta(days=1):
                continue
        except ValueError:
            pass

        with log_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json as _json
                    rec = _json.loads(line)
                    ts_raw = rec.get("timestamp", "2000-01-01T00:00:00+00:00")
                    ts = datetime.fromisoformat(ts_raw)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= since:
                        total += 1
                except (ValueError, KeyError, __import__("json").JSONDecodeError):
                    continue
    return total


def _pool_size(model_dir: Path) -> int:
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        return 0
    try:
        meta = json.loads(meta_path.read_text())
        return int(meta.get("pool_size", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _bad_problem_ratio(quality_csv_path: Path) -> float | None:
    """quality_scores.csv 에서 bad problem (REVIEW + RETIRED) 비율 반환."""
    if not quality_csv_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(quality_csv_path)
        if df.empty:
            return None
        bad = df["status"].isin(["REVIEW", "RETIRED"]).sum()
        return bad / len(df)
    except Exception:
        return None


# ── 평가 ─────────────────────────────────────────────────────────────────────

class Gate:
    def __init__(self, name: str, value: Any, threshold: Any, passed: bool, note: str = ""):
        self.name = name
        self.value = value
        self.threshold = threshold
        self.passed = passed
        self.note = note

    def __repr__(self) -> str:
        icon = "✓" if self.passed else "✗"
        return f"  [{icon}] {self.name:<35} {self.value!s:>10}  (기준: {self.threshold!s})"


def evaluate(
    log_dir: Path,
    model_dir: Path,
    quality_csv: Path | None,
    policy: dict,
) -> tuple[bool, list[Gate]]:
    """트리거 여부와 게이트 목록을 반환한다."""
    gates: list[Gate] = []
    now = datetime.now(timezone.utc)

    # ── Gate 1: 신규 attempt 건수 ────────────────────────────────────────────
    promoted = _promoted_at(model_dir) or (now - timedelta(days=9999))
    new_attempts = _count_new_attempts(log_dir, since=promoted)
    min_att = policy["min_new_attempts"]
    gates.append(Gate(
        "신규 attempt 건수",
        new_attempts,
        f"≥ {min_att}",
        new_attempts >= min_att,
    ))

    # ── Gate 2: 문제별 평균 attempt ──────────────────────────────────────────
    pool_sz = _pool_size(model_dir)
    avg_per = (new_attempts / pool_sz) if pool_sz > 0 else 0.0
    min_avg = policy["min_avg_attempts_per_problem"]
    gates.append(Gate(
        "문제별 평균 attempt",
        f"{avg_per:.1f}",
        f"≥ {min_avg}",
        avg_per >= min_avg,
        f"(pool_size={pool_sz})",
    ))

    # ── Gate 3: 마지막 승격 이후 경과일 ─────────────────────────────────────
    days_elapsed = (now - promoted).days if promoted else 9999
    min_days = policy["min_days_since_last_promote"]
    gates.append(Gate(
        "마지막 승격 이후 경과일",
        f"{days_elapsed}일",
        f"≥ {min_days}일",
        days_elapsed >= min_days,
        f"(promoted_at={promoted.date() if promoted.year > 1000 else 'never'})",
    ))

    # ── Gate 4 (선택): bad problem 비율 ──────────────────────────────────────
    qp = quality_csv if quality_csv else Path("/nonexistent")
    ratio = _bad_problem_ratio(qp)
    require_quality = policy.get("require_quality_check", False)
    if ratio is not None:
        min_ratio = policy["min_bad_problem_ratio"]
        gates.append(Gate(
            "bad problem 비율",
            f"{ratio:.3f}",
            f"≥ {min_ratio}",
            ratio >= min_ratio,
            "(REVIEW+RETIRED/전체)",
        ))
    elif require_quality:
        gates.append(Gate(
            "bad problem 비율",
            "N/A",
            "require_quality_check=true",
            False,
            "(quality_scores.csv 없음)",
        ))

    should_run = all(g.passed for g in gates)
    return should_run, gates


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="feedback MLOps 트리거 평가")
    ap.add_argument("--log-dir",     type=Path, default=_DEFAULT_LOG_DIR)
    ap.add_argument("--model-dir",   type=Path, default=_DEFAULT_MODEL_DIR,
                    help="current/ 디렉토리 (metadata.json 읽기)")
    ap.add_argument("--quality-csv", type=Path, default=None,
                    help="score_problem_quality 출력 CSV (선택)")
    ap.add_argument("--policy",      type=Path, default=_DEFAULT_POLICY)
    ap.add_argument("--output",      type=Path, default=Path("/tmp/should_run.txt"),
                    help="'true'/'false' 기록 (Argo output parameter 용)")
    ap.add_argument("--force", action="store_true",
                    help="모든 게이트를 무시하고 강제 트리거")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print()
    print("═" * _W)
    print("  FEEDBACK TRIGGER POLICY CHECK")
    print("═" * _W)

    policy = _load_policy(args.policy)

    if args.force:
        print("  --force: 모든 게이트 건너뜀 → 트리거")
        args.output.write_text("true")
        return 0

    should_run, gates = evaluate(
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        quality_csv=args.quality_csv,
        policy=policy,
    )

    print()
    for g in gates:
        print(repr(g))
        if g.note:
            print(f"        {g.note}")

    decision = "true" if should_run else "false"
    print()
    print(f"  {'─' * 56}")
    print(f"  TRIGGER DECISION: {decision.upper()}")
    print("═" * _W)
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(decision)

    return 0 if should_run else 1


if __name__ == "__main__":
    sys.exit(main())
