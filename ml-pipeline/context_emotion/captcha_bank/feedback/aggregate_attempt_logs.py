#!/usr/bin/env python3
"""
attempt log JSONL → sample_id별 집계 통계 CSV.

사용법:
    python -m context_emotion.captcha_bank.feedback.aggregate_attempt_logs \\
        --log-dir /data/context_emotion/attempt_logs \\
        --since-days 30 \\
        --output /workdir/aggregated_stats.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .log_schema import AGG_COLUMNS, SUSPICIOUS_SOLVE_TIME_MS, ProblemStats

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path("/data/context_emotion/attempt_logs")
_W = 64


# ── 로그 로드 ────────────────────────────────────────────────────────────────

def load_records(log_dir: Path, since: datetime) -> list[dict]:
    """since 이후의 JSONL attempt 레코드를 모두 읽어 반환한다."""
    records: list[dict] = []

    log_files = sorted(log_dir.glob("attempts_*.jsonl"))
    if not log_files:
        logger.warning("로그 파일 없음: %s", log_dir)
        return []

    for log_file in log_files:
        # 파일명 날짜 (attempts_YYYYMMDD.jsonl) 로 조기 스킵
        try:
            date_str = log_file.stem.replace("attempts_", "")
            file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            # 하루 여유를 주어 날짜 경계 attempt를 놓치지 않는다
            if file_date < since - timedelta(days=1):
                continue
        except ValueError:
            pass  # 파일명 형식 비표준 — 포함

        loaded = 0
        with log_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts_raw = rec.get("timestamp", "2000-01-01T00:00:00+00:00")
                    ts = datetime.fromisoformat(ts_raw)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < since:
                        continue
                    records.append(rec)
                    loaded += 1
                except (json.JSONDecodeError, ValueError):
                    continue
        logger.info("  %s → %d건 로드", log_file.name, loaded)

    return records


# ── 집계 ─────────────────────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> pd.DataFrame:
    """레코드 목록을 sample_id별로 집계해 DataFrame으로 반환한다."""
    stats: dict[str, ProblemStats] = {}

    for rec in records:
        sid = rec.get("sample_id")
        if not sid:
            continue

        s = stats.setdefault(sid, ProblemStats(sample_id=sid))
        s.attempt_count += 1

        if rec.get("is_correct"):
            s.correct_count += 1

        if float(rec.get("points", 0.0)) == 0.5:
            s.aux_count += 1

        t = int(rec.get("solve_time_ms", 0))
        s.solve_times.append(t)
        if t < SUSPICIOUS_SOLVE_TIME_MS:
            s.suspicious_count += 1

        s.retry_counts.append(int(rec.get("retry_count", 0)))

        pfx = rec.get("session_id_pfx", "")
        if pfx:
            s.session_pfxs.add(pfx)

        pv = rec.get("pool_version", "")
        if pv:
            s.pool_versions.add(pv)

    rows = []
    for sid, s in stats.items():
        n = s.attempt_count
        times  = s.solve_times  or [0]
        retries = s.retry_counts or [0]

        rows.append({
            "sample_id":            sid,
            "attempt_count":        n,
            "correct_count":        s.correct_count,
            "aux_count":            s.aux_count,
            "human_pass_rate":      round(s.correct_count / n, 4) if n else 0.0,
            "avg_solve_time_ms":    round(float(np.mean(times)), 1),
            "median_solve_time_ms": round(float(np.median(times)), 1),
            "retry_rate":           round(sum(1 for r in retries if r > 0) / n, 4) if n else 0.0,
            "suspicious_rate":      round(s.suspicious_count / n, 4) if n else 0.0,
            "unique_sessions":      len(s.session_pfxs),
            "pool_versions_seen":   len(s.pool_versions),
        })

    if not rows:
        return pd.DataFrame(columns=AGG_COLUMNS)

    return pd.DataFrame(rows)[AGG_COLUMNS]


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="attempt log 집계")
    ap.add_argument("--log-dir",    type=Path, default=_DEFAULT_LOG_DIR)
    ap.add_argument("--since-days", type=int,  default=30,
                    help="집계 기간 (일). 기본 30일")
    ap.add_argument("--since",      type=str,  default=None,
                    help="ISO 8601 날짜 (YYYY-MM-DD). --since-days 보다 우선")
    ap.add_argument("--output",     type=Path, required=True,
                    help="집계 결과 CSV 출력 경로")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print()
    print("═" * _W)
    print("  AGGREGATE ATTEMPT LOGS")
    print("═" * _W)

    if args.since:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    else:
        since_dt = datetime.now(timezone.utc) - timedelta(days=args.since_days)

    print(f"  로그 디렉토리: {args.log_dir}")
    print(f"  집계 기준:     {since_dt.date()} 이후")

    records = load_records(args.log_dir, since_dt)
    print(f"  로드된 attempt: {len(records):,}건")

    if not records:
        print("  [경고] attempt 없음 — 빈 집계 결과 저장")

    df = aggregate(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"  집계된 문제 수: {len(df):,}개")
    print(f"  출력 파일:      {args.output}")
    print("═" * _W)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
