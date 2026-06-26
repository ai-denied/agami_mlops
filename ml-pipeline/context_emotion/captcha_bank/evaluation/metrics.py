"""Security evaluation metrics for CAPTCHA pool + attacker proxy model.

compute_eval_metrics()가 evaluation_result.json의 전체 내용을 반환한다.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Optional

import numpy as np

from context_emotion.captcha_bank.choice_generation import (
    EMOTIONS,
    choice_credit,
    generate_choices,
    parse_aux,
)

_PASS_SCORE   = 2.5   # 3문제 기준 통과 최소 점수 (policy 일치)
_CHALLENGE_N  = 3     # 1회 챌린지 문항 수


def compute_eval_metrics(
    rows: list[dict],
    model_preds: list[str],
    version: str,
    pool_csv: str = "",
    pass_score: float = _PASS_SCORE,
) -> dict:
    """전체 평가 지표 딕셔너리를 생성한다.

    Args:
        rows:        captcha_pool.csv 전체 행 (list of dict)
        model_preds: emotion_attacker 모델이 각 행에 대해 예측한 레이블 리스트
        version:     모델 버전 문자열
        pool_csv:    소스 CSV 경로 (metadata용)
        pass_score:  챌린지 통과 최소 점수

    Returns:
        dict — evaluation_result.json 에 직접 저장 가능한 포맷.
    """
    assert len(rows) == len(model_preds), "rows / preds 길이 불일치"
    n = len(rows)
    if n == 0:
        return _empty(version)

    y_true = [r.get("final_emotion", "") for r in rows]

    # ── 1. 어태커 정답률 ────────────────────────────────────────────────────
    correct_mask = [pred == true for pred, true in zip(model_preds, y_true)]
    attacker_pass_rate = float(sum(correct_mask)) / n

    # ── 2. 3문제 챌린지 통과율 (이진 점수 합산) ────────────────────────────
    binary_scores = [1.0 if c else 0.0 for c in correct_mask]
    choice_policy_pass_rate = _three_question_pass_rate(binary_scores, pass_score)

    # ── 3. 모호 비율 (human_confidence = low 또는 ambiguous_models > 0) ────
    ambiguous_count = sum(
        r.get("human_confidence", "high") == "low"
        or int(r.get("ambiguous_models", 0) or 0) > 0
        for r in rows
    )
    ambiguous_rate = ambiguous_count / n

    # ── 4. 레이블 분포 ──────────────────────────────────────────────────────
    label_distribution = dict(Counter(y_true))

    # ── 5. security_grade 분포 (있을 경우) ─────────────────────────────────
    security_grade_dist = dict(Counter(r.get("security_grade", "") for r in rows))
    security_grade_dist.pop("", None)

    # ── 6. sklearn 모델 macro F1 / micro accuracy ───────────────────────────
    try:
        from sklearn.metrics import f1_score
        unique_labels = sorted(set(y_true + list(model_preds)))
        macro_f1 = float(f1_score(y_true, model_preds, labels=unique_labels,
                                  average="macro", zero_division=0))
        micro_acc = float(sum(correct_mask)) / n
    except ImportError:
        macro_f1 = 0.0
        micro_acc = float(sum(correct_mask)) / n

    # ── 7. VLM 공격자 통계 (열이 있을 경우) ────────────────────────────────
    vlm_stats = _vlm_attacker_stats(rows, pass_score)

    return {
        "version":                 version,
        "pool_size":               n,
        "pool_csv":                pool_csv,
        "pass_score":              pass_score,

        # 핵심 보안 지표
        "attacker_pass_rate":       round(attacker_pass_rate, 4),
        "robust_rate":              round(1.0 - attacker_pass_rate, 4),
        "ambiguous_rate":           round(ambiguous_rate, 4),

        # 챌린지 통과율 (sklearn attacker 기준)
        "choice_policy_pass_rate":  round(choice_policy_pass_rate, 4),

        # sklearn 모델 자체 성능
        "macro_f1_attacker":        round(macro_f1, 4),
        "micro_accuracy_attacker":  round(micro_acc, 4),

        # 분포
        "label_distribution":       label_distribution,
        "security_grade_distribution": security_grade_dist,

        # VLM 어태커 (선택)
        "vlm_attacker_stats":       vlm_stats,

        # 프로모션 적합성 (compare_candidate.py가 재계산하지만 빠른 참고용)
        "promotion_eligible":       attacker_pass_rate < 0.35 and n >= 200,
    }


def _three_question_pass_rate(per_question_scores: list[float], threshold: float) -> float:
    """이진 점수(0/1)의 경험적 분포로 3문제 합산 통과율을 계산한다."""
    counts: Counter = Counter(per_question_scores)
    total = len(per_question_scores)
    if total == 0:
        return 0.0

    dist = {0.0: 1.0}
    for _ in range(_CHALLENGE_N):
        nxt: Counter = Counter()
        for acc_score, prob in dist.items():
            for pts, cnt in counts.items():
                nxt[acc_score + pts] += prob * (cnt / total)
        dist = dict(nxt)

    return float(sum(p for s, p in dist.items() if s >= threshold))


def _vlm_attacker_stats(rows: list[dict], threshold: float) -> dict:
    """build_choice_policy_report 와 같은 방식으로 VLM 열 통계를 계산한다.

    qwen_emotion, smolvlm_emotion, self_attack_emotion 열이 있을 때만 계산.
    """
    ATTACKER_COLS = {
        "qwen":        "qwen_emotion",
        "smolvlm":     "smolvlm_emotion",
        "self_attack": "self_attack_emotion",
    }
    stats = {}

    for name, col in ATTACKER_COLS.items():
        if not rows or col not in rows[0]:
            continue

        rows_with_choices = []
        for r in rows:
            choices = generate_choices(r, seed=abs(hash(str(r.get("sample_id", "")))) % (2**31))
            rows_with_choices.append({**r, "choices": json.dumps(choices)})

        counts = Counter(
            _attacker_points_from_choices(r, col) for r in rows_with_choices
        )
        total = max(1, len(rows_with_choices))
        primary_rate = counts.get(1.0, 0) / total
        partial_rate = counts.get(0.5, 0) / total

        dist = {0.0: 1.0}
        for _ in range(_CHALLENGE_N):
            nxt: Counter = Counter()
            for s, p in dist.items():
                for pts, cnt in counts.items():
                    nxt[s + pts] += p * (cnt / total)
            dist = dict(nxt)
        pass_rate_3q = float(sum(p for s, p in dist.items() if s >= threshold))

        stats[name] = {
            "single_q_primary_rate": round(primary_rate, 4),
            "single_q_partial_rate": round(partial_rate, 4),
            "three_q_pass_rate":     round(pass_rate_3q, 4),
        }

    return stats


def _attacker_points_from_choices(row: dict, attacker_col: str) -> float:
    pred = row.get(attacker_col, "")
    try:
        choices = json.loads(row.get("choices", "[]"))
    except Exception:
        return 0.0
    if pred not in choices:
        return 0.0
    return choice_credit(pred, row)


def _empty(version: str) -> dict:
    return {
        "version": version, "pool_size": 0,
        "attacker_pass_rate": None, "robust_rate": None,
        "ambiguous_rate": None, "choice_policy_pass_rate": None,
        "macro_f1_attacker": None, "micro_accuracy_attacker": None,
        "label_distribution": {}, "security_grade_distribution": {},
        "vlm_attacker_stats": {}, "promotion_eligible": False,
    }
