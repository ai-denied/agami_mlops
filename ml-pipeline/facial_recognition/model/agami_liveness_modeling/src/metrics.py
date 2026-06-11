import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def safe_pr_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, prob))


def binary_metrics(
    y_true: np.ndarray,
    prob_spoof: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    label: 0=live/genuine, 1=spoof/attack
    pred_spoof=1 means reject/block.
    FAR = attack인데 live로 통과된 비율 = FN / (TP + FN)
    FRR = live인데 spoof로 거절된 비율 = FP / (TN + FP)
    """
    y_true = np.asarray(y_true).astype(int)
    prob_spoof = np.asarray(prob_spoof).astype(float)
    pred = (prob_spoof >= threshold).astype(int)

    acc = accuracy_score(y_true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, pos_label=1, average="binary", zero_division=0
    )

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    live_count = tn + fp
    attack_count = tp + fn

    frr = fp / live_count if live_count else float("nan")
    far = fn / attack_count if attack_count else float("nan")
    genuine_pass_rate = tn / live_count if live_count else float("nan")
    attack_pass_rate = far
    attack_block_rate = tp / attack_count if attack_count else float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision_spoof": float(precision),
        "recall_spoof": float(recall),
        "f1_spoof": float(f1),
        "roc_auc": safe_auc(y_true, prob_spoof),
        "pr_auc": safe_pr_auc(y_true, prob_spoof),
        "far_attack_pass_rate": float(far),
        "frr_genuine_reject_rate": float(frr),
        "attack_pass_rate": float(attack_pass_rate),
        "attack_block_rate": float(attack_block_rate),
        "genuine_pass_rate": float(genuine_pass_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "positive_spoof_count": int((y_true == 1).sum()),
        "negative_live_count": int((y_true == 0).sum()),
    }


def threshold_sweep(
    y_true: np.ndarray,
    prob_spoof: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for th in thresholds:
        rows.append(binary_metrics(y_true, prob_spoof, float(th)))
    return pd.DataFrame(rows)


def choose_threshold(
    y_true: np.ndarray,
    prob_spoof: np.ndarray,
    strategy: str = "best_f1",
    max_frr: Optional[float] = None,
) -> float:
    strategy = strategy.lower()
    if strategy == "default":
        return 0.5

    df = threshold_sweep(y_true, prob_spoof)

    if strategy == "best_f1":
        idx = df["f1_spoof"].idxmax()
        return float(df.loc[idx, "threshold"])

    if strategy == "eer_like":
        diff = (df["far_attack_pass_rate"] - df["frr_genuine_reject_rate"]).abs()
        idx = diff.idxmin()
        return float(df.loc[idx, "threshold"])

    if strategy == "low_far":
        cand = df.copy()
        if max_frr is not None:
            cand = cand[cand["frr_genuine_reject_rate"] <= max_frr]
        if len(cand) == 0:
            cand = df
        cand = cand.sort_values(["far_attack_pass_rate", "f1_spoof"], ascending=[True, False])
        return float(cand.iloc[0]["threshold"])

    raise ValueError(f"Unknown threshold strategy: {strategy}")


def grouped_metrics(
    y_true: np.ndarray,
    prob_spoof: np.ndarray,
    groups: Iterable,
    threshold: float,
    group_name: str,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    rows = []
    y_true = np.asarray(y_true)
    prob_spoof = np.asarray(prob_spoof)
    groups = np.asarray(groups).astype(str)

    for g in sorted(pd.unique(groups)):
        idx = groups == g
        if idx.sum() == 0:
            continue
        m = binary_metrics(y_true[idx], prob_spoof[idx], threshold)
        m.update({
            "model": model_name,
            "split": split_name,
            "group_name": group_name,
            "group_value": str(g),
        })
        rows.append(m)
    return pd.DataFrame(rows)
