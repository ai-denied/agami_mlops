from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def safe_roc_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, scores)


def safe_pr_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return None
    return average_precision_score(y_true, scores)


def evaluate_thresholds(y_true, scores, thresholds):
    rows = []

    for th in thresholds:
        preds = (scores >= th).astype(int)

        acc = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

        human_block_rate = fp / max(fp + tn, 1)
        bot_miss_rate = fn / max(fn + tp, 1)
        bot_recall = tp / max(tp + fn, 1)

        rows.append({
            "threshold": float(th),
            "accuracy": float(acc),
            "precision_bot": float(precision),
            "recall_bot": float(recall),
            "f1_bot": float(f1),
            "human_block_rate": float(human_block_rate),
            "bot_miss_rate": float(bot_miss_rate),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "bot_recall": float(bot_recall),
        })

    return pd.DataFrame(rows)


def choose_threshold_by_human_block_rate(
    threshold_df: pd.DataFrame,
    max_human_block_rate: float,
    mode: str = "best_bot_recall",
    min_threshold: Optional[float] = None,
):
    """
    human_block_rate 제한을 만족하는 threshold 중 하나를 고른다.

    mode="best_bot_recall":
      - 제한 안에서 bot_recall / f1 / accuracy가 가장 좋은 threshold 선택
      - low_risk_threshold 선택에 사용

    mode="high_risk":
      - min_threshold 이상이면서 human_block_rate 제한을 만족하는 threshold 중
        bot_recall이 가장 높은 threshold 선택
      - high_risk_threshold 선택에 사용
      - 예: min_threshold=0.60이면 0.60 근처 이상의 강한 위험 기준을 우선 탐색
    """

    valid = threshold_df[threshold_df["human_block_rate"] <= max_human_block_rate].copy()

    if min_threshold is not None:
        valid = valid[valid["threshold"] >= min_threshold].copy()

    if len(valid) > 0:
        valid = valid.sort_values(
            by=["bot_recall", "f1_bot", "accuracy"],
            ascending=False,
        )

        if mode == "high_risk":
            return float(valid.iloc[0]["threshold"]), "high_risk_threshold_hbr_limited"

        return float(valid.iloc[0]["threshold"]), "policy_human_block_rate_limited"

    # min_threshold 조건 때문에 후보가 없으면 min_threshold 조건만 빼고 다시 탐색
    fallback_valid = threshold_df[threshold_df["human_block_rate"] <= max_human_block_rate].copy()

    if len(fallback_valid) > 0:
        fallback_valid = fallback_valid.sort_values(
            by=["bot_recall", "f1_bot", "accuracy"],
            ascending=False,
        )

        if mode == "high_risk":
            return float(fallback_valid.iloc[0]["threshold"]), "high_risk_threshold_fallback_without_min_threshold"

        return float(fallback_valid.iloc[0]["threshold"]), "policy_human_block_rate_limited_without_min_threshold"

    fallback = threshold_df.sort_values(by=["f1_bot", "accuracy"], ascending=False)
    return float(fallback.iloc[0]["threshold"]), "fallback_best_f1"


def get_tpr_at_fpr(y_true, scores, target_fpr_list=(0.01, 0.05, 0.1, 0.2)):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    result = {}

    for target_fpr in target_fpr_list:
        valid_idx = np.where(fpr <= target_fpr)[0]

        if len(valid_idx) == 0:
            result[f"tpr_at_fpr_{target_fpr}"] = {
                "tpr": 0.0,
                "fpr": None,
                "threshold": None,
            }
            continue

        best_idx = valid_idx[np.argmax(tpr[valid_idx])]

        result[f"tpr_at_fpr_{target_fpr}"] = {
            "tpr": float(tpr[best_idx]),
            "fpr": float(fpr[best_idx]),
            "threshold": float(thresholds[best_idx]),
        }

    return result


def print_eval_report(name: str, y_true, scores, threshold: float):
    preds = (scores >= threshold).astype(int)

    print(f"\n========== {name} 평가 ==========")
    print(f"Threshold: {threshold}")

    roc_auc = safe_roc_auc(y_true, scores)
    pr_auc = safe_pr_auc(y_true, scores)

    print(f"ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC: N/A")
    print(f"PR-AUC: {pr_auc:.4f}" if pr_auc is not None else "PR-AUC: N/A")

    print("\nConfusion Matrix [labels: 0=human, 1=bot]")
    print(confusion_matrix(y_true, preds, labels=[0, 1]))

    print("\nClassification Report")
    print(classification_report(
        y_true,
        preds,
        target_names=["human", "bot"],
        zero_division=0,
    ))