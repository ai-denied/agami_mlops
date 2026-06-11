#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
current vs candidate 성능 비교 스크립트

current/metadata.json과 candidates/{version}/metadata.json의
performance 섹션을 읽어 지표를 비교하고 승격 조건 PASS/FAIL을 판단한다.

파일은 수정하지 않는다.

사용법:
  python -m flashlight.scripts.compare_candidate --version v4_20260610
  python -m flashlight.scripts.compare_candidate --version v4_20260610 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_STORE            = os.path.join(_ML_PIPELINE_ROOT, "model-store", "flashlight")
_CANDIDATES_DIR   = os.path.join(_STORE, "candidates")
_CURRENT_DIR      = os.path.join(_STORE, "current")


# ---------------------------------------------------------------------------
# 지표 정의
# ---------------------------------------------------------------------------

@dataclass
class MetricDef:
    key:            str
    label:          str
    higher_better:  bool   # True → 높을수록 좋음 / False → 낮을수록 좋음
    condition:      Optional[str] = None   # None | "gte_current" | "lte_current" | "lte_{value}"


METRICS: list[MetricDef] = [
    MetricDef(
        key="test_roc_auc",
        label="ROC-AUC (test)",
        higher_better=True,
        condition="gte_current",         # candidate >= current
    ),
    MetricDef(
        key="test_f1_bot",
        label="F1-bot (test)",
        higher_better=True,
        condition="gte_current",         # candidate >= current
    ),
    MetricDef(
        key="test_human_block_rate",
        label="Human block rate (test)",
        higher_better=False,
        condition="lte_0.20",            # candidate <= 0.20 (절대 상한)
    ),
    MetricDef(
        key="test_bot_miss_rate",
        label="Bot miss rate (test)",
        higher_better=False,
        condition=None,                  # 참고 지표 (승격 조건 아님)
    ),
    MetricDef(
        key="three_attempt_bot_block_rate",
        label="3-attempt bot block rate",
        higher_better=True,
        condition=None,                  # 참고 지표
    ),
    MetricDef(
        key="three_attempt_human_block_rate",
        label="3-attempt human block rate",
        higher_better=False,
        condition="lte_current",         # candidate <= current (기존보다 나쁘지 않음)
    ),
]


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def _load_metadata(path: str, label: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} metadata.json을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 비교 로직
# ---------------------------------------------------------------------------

@dataclass
class MetricRow:
    key:        str
    label:      str
    current:    Optional[float]
    candidate:  Optional[float]
    delta:      Optional[float]
    direction:  str    # "▲" | "▼" | "─"
    has_cond:   bool
    cond_pass:  Optional[bool]
    cond_desc:  str


def _evaluate_condition(cond: str, candidate: float, current: Optional[float]) -> tuple[bool, str]:
    """(pass, description) 반환."""
    if cond == "gte_current":
        ok = (current is None) or (candidate >= current)
        return ok, f"candidate ≥ current  ({candidate:.4f} ≥ {current:.4f})"
    if cond == "lte_current":
        ok = (current is None) or (candidate <= current)
        return ok, f"candidate ≤ current  ({candidate:.4f} ≤ {current:.4f})"
    if cond.startswith("lte_"):
        threshold = float(cond[4:])
        ok = candidate <= threshold
        return ok, f"candidate ≤ {threshold}  ({candidate:.4f} ≤ {threshold:.4f})"
    return True, cond


def _build_rows(
    current_perf: dict,
    candidate_perf: dict,
) -> list[MetricRow]:
    rows = []
    for m in METRICS:
        cur = current_perf.get(m.key)
        cand = candidate_perf.get(m.key)

        cur  = float(cur)  if cur  is not None else None
        cand = float(cand) if cand is not None else None

        if cur is not None and cand is not None:
            delta = cand - cur
            if abs(delta) < 1e-9:
                direction = "─"
            elif (delta > 0 and m.higher_better) or (delta < 0 and not m.higher_better):
                direction = "▲"   # 개선
            else:
                direction = "▼"   # 악화
        else:
            delta = None
            direction = "─"

        has_cond = m.condition is not None
        cond_pass = None
        cond_desc = ""
        if has_cond and cand is not None:
            cond_pass, cond_desc = _evaluate_condition(m.condition, cand, cur)

        rows.append(MetricRow(
            key=m.key,
            label=m.label,
            current=cur,
            candidate=cand,
            delta=delta,
            direction=direction,
            has_cond=has_cond,
            cond_pass=cond_pass,
            cond_desc=cond_desc,
        ))
    return rows


def _overall_pass(rows: list[MetricRow]) -> bool:
    return all(r.cond_pass for r in rows if r.has_cond and r.cond_pass is not None)


# ---------------------------------------------------------------------------
# 출력 — 텍스트
# ---------------------------------------------------------------------------

def _fmt(val: Optional[float]) -> str:
    return f"{val:.4f}" if val is not None else "N/A"


def _fmt_delta(delta: Optional[float]) -> str:
    if delta is None:
        return "  N/A  "
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def print_comparison(
    current_meta: dict,
    candidate_meta: dict,
    rows: list[MetricRow],
    passed: bool,
) -> None:
    cur_ver  = current_meta.get("version", "unknown")
    cand_ver = candidate_meta.get("version", "unknown")

    W_LABEL  = 30
    W_VAL    = 10
    W_DELTA  = 9
    W_DIR    = 3
    W_COND   = 8
    SEP = "─" * (W_LABEL + W_VAL * 2 + W_DELTA + W_DIR + W_COND + 13)

    print()
    print("┌" + "─" * (len(SEP) - 2) + "┐")
    title = f"  성능 비교:  current ({cur_ver})  →  candidate ({cand_ver})"
    print(f"│{title:<{len(SEP)-2}}│")
    print("├" + SEP[1:-1] + "┤")

    header = (
        f"│ {'지표':<{W_LABEL}} │ {'current':>{W_VAL}} │ {'candidate':>{W_VAL}} │"
        f" {'delta':>{W_DELTA}} │ {'↑↓':^{W_DIR}} │ {'조건':^{W_COND}} │"
    )
    print(header)
    print("├" + SEP[1:-1] + "┤")

    cond_rows = []
    ref_rows  = []
    for r in rows:
        (cond_rows if r.has_cond else ref_rows).append(r)

    def _print_row(r: MetricRow) -> None:
        if r.has_cond:
            if r.cond_pass is None:
                cond_str = " N/A    "
            else:
                cond_str = " PASS   " if r.cond_pass else " FAIL   "
        else:
            cond_str = " (ref)  "

        print(
            f"│ {r.label:<{W_LABEL}} │ {_fmt(r.current):>{W_VAL}} │"
            f" {_fmt(r.candidate):>{W_VAL}} │ {_fmt_delta(r.delta):>{W_DELTA}} │"
            f" {r.direction:^{W_DIR}} │{cond_str}│"
        )

    # 승격 조건 지표
    for r in cond_rows:
        _print_row(r)

    # 구분선 + 참고 지표
    if ref_rows:
        print("├" + "┄" * (len(SEP) - 2) + "┤")
        for r in ref_rows:
            _print_row(r)

    print("└" + "─" * (len(SEP) - 2) + "┘")

    # 조건 상세
    print()
    print("  승격 조건 상세:")
    for r in cond_rows:
        if r.cond_pass is None:
            status = "  ?  "
        elif r.cond_pass:
            status = " PASS"
        else:
            status = " FAIL"
        print(f"  [{status}]  {r.label}")
        if r.cond_desc:
            print(f"           조건: {r.cond_desc}")

    # 최종 판정
    verdict     = "PASS" if passed else "FAIL"
    verdict_bar = "=" * 44
    print()
    print(verdict_bar)
    if passed:
        print(f"  최종 판정:  {verdict}  —  승격 가능")
    else:
        print(f"  최종 판정:  {verdict}  —  승격 불가")
        print("  FAIL 조건을 해결한 후 재시도하세요.")
    print(verdict_bar)
    print()


# ---------------------------------------------------------------------------
# 출력 — JSON
# ---------------------------------------------------------------------------

def build_json_result(
    current_meta: dict,
    candidate_meta: dict,
    rows: list[MetricRow],
    passed: bool,
) -> dict:
    metrics_out = []
    for r in rows:
        metrics_out.append({
            "key":       r.key,
            "current":   r.current,
            "candidate": r.candidate,
            "delta":     r.delta,
            "direction": r.direction,
            "has_condition": r.has_cond,
            "condition_pass": r.cond_pass,
            "condition_desc": r.cond_desc if r.cond_desc else None,
        })

    conditions = [
        {
            "key":   r.key,
            "pass":  r.cond_pass,
            "desc":  r.cond_desc,
        }
        for r in rows if r.has_cond
    ]

    return {
        "current_version":   current_meta.get("version"),
        "candidate_version": candidate_meta.get("version"),
        "overall_pass":      passed,
        "verdict":           "PASS" if passed else "FAIL",
        "metrics":           metrics_out,
        "conditions":        conditions,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="current vs candidate 성능 비교 및 승격 조건 판단"
    )
    parser.add_argument(
        "--version",
        required=True,
        help="비교할 후보 버전명 (예: v4_20260610)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="비교 결과를 JSON으로 출력 (텍스트 출력 대신)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    version = args.version

    current_meta_path   = os.path.join(_CURRENT_DIR, "metadata.json")
    candidate_meta_path = os.path.join(_CANDIDATES_DIR, version, "metadata.json")

    try:
        current_meta   = _load_metadata(current_meta_path,   "current")
        candidate_meta = _load_metadata(candidate_meta_path, f"candidates/{version}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    current_perf   = current_meta.get("performance", {})
    candidate_perf = candidate_meta.get("performance", {})

    rows   = _build_rows(current_perf, candidate_perf)
    passed = _overall_pass(rows)

    if args.as_json:
        result = build_json_result(current_meta, candidate_meta, rows, passed)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_comparison(current_meta, candidate_meta, rows, passed)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
