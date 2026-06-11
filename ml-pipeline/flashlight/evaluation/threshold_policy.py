from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def classify_single_attempt_risk(
    bot_risk_score: float,
    low_risk_threshold: float,
    high_risk_threshold: float,
) -> str:
    if bot_risk_score < low_risk_threshold:
        return "low_risk"
    if bot_risk_score < high_risk_threshold:
        return "suspicious"
    return "high_risk"


def apply_three_attempt_policy(
    scores: List[float],
    low_risk_threshold: float,
    high_risk_threshold: float,
    block_suspicious_count: int = 2,
    block_high_risk_count: int = 1,
    block_total_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    3회 연속 손전등 CAPTCHA 점수 누적 정책.

    기본 정책:
    - high_risk가 1회 이상 나오면 block
    - suspicious가 2회 이상 나오면 block
    - total_score가 block_total_score 이상이면 block
    - suspicious가 1회만 있으면 challenge_again
    - 전부 low_risk면 allow
    """

    if len(scores) == 0:
        raise ValueError("scores must not be empty")

    scores = [float(s) for s in scores]

    total_score = float(np.sum(scores))
    avg_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))

    suspicious_count = int(sum(s >= low_risk_threshold for s in scores))
    high_risk_count = int(sum(s >= high_risk_threshold for s in scores))

    if block_total_score is None:
        block_total_score = low_risk_threshold * 2.0

    if (
        high_risk_count >= block_high_risk_count
        or suspicious_count >= block_suspicious_count
        or total_score >= block_total_score
    ):
        decision = "block"
    elif suspicious_count >= 1:
        decision = "challenge_again"
    else:
        decision = "allow"

    return {
        "scores": scores,
        "total_score": round(total_score, 6),
        "avg_score": round(avg_score, 6),
        "max_score": round(max_score, 6),
        "min_score": round(min_score, 6),
        "suspicious_count": suspicious_count,
        "high_risk_count": high_risk_count,
        "low_risk_threshold": float(low_risk_threshold),
        "high_risk_threshold": float(high_risk_threshold),
        "block_total_score": float(block_total_score),
        "decision": decision,
    }


def make_attempt_groups(
    y_true: np.ndarray,
    scores: np.ndarray,
    attempts: int,
    seed: int,
) -> List[Tuple[int, List[float]]]:
    """
    현재 데이터에 실제 3회 연속 세션 ID가 없을 수 있으므로,
    평가용으로 같은 label끼리 랜덤하게 3개씩 묶어 3회 수행을 시뮬레이션한다.
    """

    rng = np.random.default_rng(seed)
    y_true = np.array(y_true).astype(int)
    scores = np.array(scores).astype(float)

    groups = []

    for label in [0, 1]:
        idx = np.where(y_true == label)[0]
        rng.shuffle(idx)

        usable = (len(idx) // attempts) * attempts
        idx = idx[:usable]

        for i in range(0, usable, attempts):
            chunk = idx[i:i + attempts]
            groups.append((label, scores[chunk].tolist()))

    rng.shuffle(groups)
    return groups


def evaluate_three_attempt_policy(
    y_true: np.ndarray,
    scores: np.ndarray,
    low_risk_threshold: float,
    high_risk_threshold: float,
    attempts: int = 3,
    seed: int = 42,
    block_suspicious_count: int = 2,
    block_high_risk_count: int = 1,
    block_total_score: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    groups = make_attempt_groups(y_true, scores, attempts=attempts, seed=seed)
    rows = []

    for group_id, (label, score_list) in enumerate(groups):
        policy_result = apply_three_attempt_policy(
            score_list,
            low_risk_threshold=low_risk_threshold,
            high_risk_threshold=high_risk_threshold,
            block_suspicious_count=block_suspicious_count,
            block_high_risk_count=block_high_risk_count,
            block_total_score=block_total_score,
        )

        rows.append({
            "group_id": group_id,
            "label": int(label),
            "label_name": "bot" if label == 1 else "human",
            **policy_result,
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df, {}

    human_df = df[df["label"] == 0]
    bot_df = df[df["label"] == 1]

    def rate(frame, decision):
        if len(frame) == 0:
            return None
        return float((frame["decision"] == decision).mean())

    summary = {
        "attempts": attempts,
        "total_groups": int(len(df)),
        "human_groups": int(len(human_df)),
        "bot_groups": int(len(bot_df)),
        "low_risk_threshold": float(low_risk_threshold),
        "high_risk_threshold": float(high_risk_threshold),
        "block_suspicious_count": int(block_suspicious_count),
        "block_high_risk_count": int(block_high_risk_count),
        "block_total_score": float(block_total_score if block_total_score is not None else low_risk_threshold * 2.0),

        "human_allow_rate": rate(human_df, "allow"),
        "human_challenge_rate": rate(human_df, "challenge_again"),
        "human_block_rate": rate(human_df, "block"),

        "bot_allow_rate": rate(bot_df, "allow"),
        "bot_challenge_rate": rate(bot_df, "challenge_again"),
        "bot_block_rate": rate(bot_df, "block"),
    }

    summary["human_friction_rate"] = (
        None if summary["human_challenge_rate"] is None
        else summary["human_challenge_rate"] + summary["human_block_rate"]
    )

    summary["bot_not_allowed_rate"] = (
        None if summary["bot_challenge_rate"] is None
        else summary["bot_challenge_rate"] + summary["bot_block_rate"]
    )

    return df, summary