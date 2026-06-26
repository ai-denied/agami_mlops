#!/usr/bin/env python3
"""
CAPTCHA 풀 CSV 구조 및 내용 검증 스크립트.

사용법:
    python -m context_emotion.captcha_bank.scripts.validate_captcha_pool \\
        --pool-csv captcha_pool.csv \\
        [--min-rows 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from context_emotion.captcha_bank.choice_generation import EMOTIONS, load_rows

REQUIRED_COLUMNS = [
    "sample_id",
    "image_path",
    "final_emotion",
]

_W = 60


def validate(pool_csv: Path, min_rows: int = 200) -> bool:
    print("═" * _W)
    print("  CAPTCHA 풀 검증")
    print("═" * _W)

    if not pool_csv.exists():
        _fail(f"파일이 없습니다: {pool_csv}")
        return False

    rows = load_rows(pool_csv)
    print(f"  전체 행 수:  {len(rows)}")

    ok = True

    # ── 필수 열 존재 확인 ────────────────────────────────────────────────────
    if rows:
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
        if missing_cols:
            _fail(f"필수 열 없음: {missing_cols}")
            ok = False
        else:
            print(f"  ✓ 필수 열 존재: {REQUIRED_COLUMNS}")

    # ── 최소 행 수 확인 ─────────────────────────────────────────────────────
    if len(rows) < min_rows:
        _fail(f"행 수 부족: {len(rows)} < {min_rows} (최소값)")
        ok = False
    else:
        print(f"  ✓ 행 수 충분: {len(rows)} >= {min_rows}")

    # ── final_emotion 유효성 ─────────────────────────────────────────────────
    invalid_emotions = [
        r.get("final_emotion", "") for r in rows
        if r.get("final_emotion", "") not in EMOTIONS
    ]
    if invalid_emotions:
        _fail(f"유효하지 않은 final_emotion: {len(invalid_emotions)}개 (예: {invalid_emotions[:3]})")
        ok = False
    else:
        print(f"  ✓ 모든 final_emotion 유효 ({len(EMOTIONS)}종 감정 중)")

    # ── 중복 sample_id 확인 ─────────────────────────────────────────────────
    ids = [r.get("sample_id", "") for r in rows]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        _fail(f"중복 sample_id: {duplicates}개")
        ok = False
    else:
        print(f"  ✓ sample_id 고유성 확인")

    # ── 클래스 분포 ─────────────────────────────────────────────────────────
    from collections import Counter
    dist = Counter(r.get("final_emotion", "") for r in rows)
    zero_classes = [e for e in EMOTIONS if dist.get(e, 0) == 0]
    if zero_classes:
        print(f"  ⚠ 0개 클래스: {zero_classes}")
    else:
        print(f"  ✓ 모든 {len(EMOTIONS)}개 감정 클래스 존재")

    print()
    if ok:
        print("  └─ ✓ 검증 통과")
    else:
        print("  └─ ✗ 검증 실패")

    return ok


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPTCHA 풀 CSV 검증")
    ap.add_argument("--pool-csv", type=Path, required=True)
    ap.add_argument("--min-rows", type=int, default=200)
    args = ap.parse_args()

    ok = validate(args.pool_csv, min_rows=args.min_rows)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
