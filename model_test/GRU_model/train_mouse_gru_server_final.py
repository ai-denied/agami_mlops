#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentient-CAPTCHA Mouse Behavior GRU Trainer - Server GPU Final

목적:
- 손전등 CAPTCHA 1회 수행 로그를 입력으로 받아 human(0) / bot(1)을 분류하는 GRU 모델 학습
- 과적합 방지: Early Stopping, Dropout, Weight Decay, Gradient Clipping, LayerNorm, Noise Augmentation
- 결과 출력: ROC-AUC, PR-AUC, Precision, Recall, F1-score, Confusion Matrix,
             Threshold별 human_block_rate / bot_miss_rate / bot_recall 등
- 서버 GPU 자동 사용

실행 예시:
python train_mouse_gru_server_final.py \
  --data "/home/ubuntu/model_test/GRU_model/merged_dynamic_features_sampled.json" \
  --out-dir "./runs/mouse_gru_final" \
  --epochs 30 \
  --batch-size 128 \
  --hidden 32 \
  --layers 1 \
  --dropout 0.4 \
  --lr 0.0003 \
  --weight-decay 0.001 \
  --patience 5 \
  --max-human-block-rate 0.20 \
  --device auto
"""

import os
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

SEQ_FEATURES = [
    "dx", "dy", "dt", "distance", "velocity", "acceleration", "angle_change",
]

STATIC_FEATURES = [
    "duration", "log_count", "total_distance", "straight_distance", "distance_ratio",
    "avg_speed", "max_speed", "speed_std", "direction_changes", "pauses",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg != "auto":
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_info(device: torch.device):
    print(f"\n사용 디바이스: {device}")
    if device.type == "cuda":
        current_idx = torch.cuda.current_device()
        print(f"GPU 이름: {torch.cuda.get_device_name(current_idx)}")
        print(f"CUDA 버전(torch): {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class MouseFeatureNormalizer:
    """train 데이터에 대해서만 fit하고 val/test/inference에는 transform만 적용한다."""
    def __init__(self):
        self.seq_scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        self.is_fitted = False

    def _get_raw_seq(self, sample: Dict) -> np.ndarray:
        seq = []
        for feat in sample.get("dynamic_features", []):
            seq.append([float(feat.get(k, 0.0)) for k in SEQ_FEATURES])
        if len(seq) == 0:
            seq = [[0.0] * len(SEQ_FEATURES)]
        return np.array(seq, dtype=np.float32)

    def _get_raw_static(self, sample: Dict) -> np.ndarray:
        stat = sample.get("static_features", {})
        row = [
            float(stat.get("duration", 0.0)),
            float(stat.get("log_count", 0.0)),
            float(stat.get("total_distance", 0.0)),
            float(stat.get("straight_distance", 0.0)),
            float(stat.get("distance_ratio", 0.0)),
            float(stat.get("avg_speed", 0.0)),
            float(stat.get("max_speed", 0.0)),
            float(stat.get("speed_std", 0.0)),
            float(stat.get("direction_changes", 0.0)),
            float(stat.get("pauses", 0.0)),
        ]
        return np.array(row, dtype=np.float32)

    def fit(self, samples: List[Dict]):
        all_seq_rows = []
        all_static_rows = []
        for sample in samples:
            all_seq_rows.append(self._get_raw_seq(sample))
            all_static_rows.append(self._get_raw_static(sample))
        self.seq_scaler.fit(np.vstack(all_seq_rows))
        self.static_scaler.fit(np.vstack(all_static_rows))
        self.is_fitted = True

    def transform_seq(self, sample: Dict) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Normalizer is not fitted yet.")
        return self.seq_scaler.transform(self._get_raw_seq(sample)).astype(np.float32)

    def transform_static(self, sample: Dict) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Normalizer is not fitted yet.")
        static = self._get_raw_static(sample).reshape(1, -1)
        return self.static_scaler.transform(static).reshape(-1).astype(np.float32)


class MouseDataset(data.Dataset):
    def __init__(
        self,
        samples: List[Dict],
        normalizer: MouseFeatureNormalizer,
        is_train: bool = False,
        seq_noise_std: float = 0.0,
        static_noise_std: float = 0.0,
    ):
        self.samples = samples
        self.normalizer = normalizer
        self.is_train = is_train
        self.seq_noise_std = seq_noise_std
        self.static_noise_std = static_noise_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x_seq = self.normalizer.transform_seq(sample)
        x_static = self.normalizer.transform_static(sample)

        # 과적합 방지용 noise augmentation: train에서만 적용
        if self.is_train and self.seq_noise_std > 0:
            x_seq = x_seq + np.random.normal(0.0, self.seq_noise_std, x_seq.shape).astype(np.float32)
        if self.is_train and self.static_noise_std > 0:
            x_static = x_static + np.random.normal(0.0, self.static_noise_std, x_static.shape).astype(np.float32)

        label = float(sample.get("label", 0))
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(x_static, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_fn(batch):
    x_seqs, x_statics, ys = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in x_seqs], dtype=torch.long)
    padded_seqs = pad_sequence(x_seqs, batch_first=True, padding_value=0.0)
    return padded_seqs, lengths, torch.stack(x_statics), torch.stack(ys)


class MouseGRUModel(nn.Module):
    def __init__(self, seq_size=7, static_size=10, hidden=32, layers=1, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(static_size, 32),
            nn.LayerNorm(32),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.fc_final = nn.Sequential(
            nn.Linear(hidden + 32, 64),
            nn.LayerNorm(64),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, lengths, x_static):
        packed = pack_padded_sequence(
            x_seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hn = self.gru(packed)
        gru_out = hn[-1]
        static_out = self.static_mlp(x_static)
        combined = torch.cat([gru_out, static_out], dim=1)
        return self.fc_final(combined).view(-1)


def resolve_group(sample: Dict, idx: int, group_key: str) -> str:
    label = int(sample.get("label", 0))
    if group_key == "auto":
        if label == 0:
            return str(sample.get("user_id") or sample.get("original_file") or sample.get("source_file") or f"sample_{idx}")
        return str(sample.get("bot_type") or sample.get("original_file") or sample.get("source_file") or f"sample_{idx}")
    return str(sample.get(group_key) or sample.get("original_file") or sample.get("source_file") or f"sample_{idx}")


def has_both_classes(samples: List[Dict]) -> bool:
    labels = [int(s.get("label", 0)) for s in samples]
    return len(set(labels)) == 2


def group_split_once(samples: List[Dict], test_size: float, group_key: str, seed: int) -> Tuple[List[Dict], List[Dict], str]:
    labels = np.array([int(s.get("label", 0)) for s in samples])
    groups = np.array([resolve_group(s, i, group_key) for i, s in enumerate(samples)])
    unique_groups = np.unique(groups)

    if len(unique_groups) < 4:
        train_part, test_part = train_test_split(samples, test_size=test_size, random_state=seed, stratify=labels)
        return train_part, test_part, "stratified_random_fallback"

    splitter = GroupShuffleSplit(n_splits=30, test_size=test_size, random_state=seed)
    for train_idx, test_idx in splitter.split(samples, labels, groups):
        train_part = [samples[i] for i in train_idx]
        test_part = [samples[i] for i in test_idx]
        if has_both_classes(train_part) and has_both_classes(test_part):
            return train_part, test_part, f"group_split_{group_key}"

    train_part, test_part = train_test_split(samples, test_size=test_size, random_state=seed, stratify=labels)
    return train_part, test_part, "stratified_random_fallback"


def make_train_val_test_split(samples: List[Dict], group_key: str = "auto", seed: int = 42):
    train_val, test, mode1 = group_split_once(samples, 0.2, group_key, seed)
    train, val, mode2 = group_split_once(train_val, 0.2, group_key, seed + 1)

    print("\n[Split 정보]")
    print(f"1차 split 방식: {mode1}")
    print(f"2차 split 방식: {mode2}")
    print(f"Train: {len(train)}")
    print(f"Val  : {len(val)}")
    print(f"Test : {len(test)}")
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        labels = [int(s.get("label", 0)) for s in part]
        print(f"{name} label count:", pd.Series(labels).value_counts().to_dict())
    return train, val, test


def make_loader(samples, normalizer, batch_size, shuffle, is_train, seq_noise_std, static_noise_std, num_workers, device):
    return data.DataLoader(
        MouseDataset(samples, normalizer, is_train, seq_noise_std if is_train else 0.0, static_noise_std if is_train else 0.0),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def get_pos_weight(samples: List[Dict], device):
    labels = np.array([int(s.get("label", 0)) for s in samples])
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0:
        return torch.tensor([1.0], device=device)
    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float):
    model.train()
    total_loss, total_count = 0.0, 0
    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_seq, lengths, x_static)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x_seq, lengths, x_static)
        loss = criterion(logits, y)
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        logits = model(x_seq, lengths, x_static)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def safe_roc_auc(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, probs)


def safe_pr_auc(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return None
    return average_precision_score(y_true, probs)


def evaluate_thresholds(y_true, probs, thresholds):
    rows = []
    for th in thresholds:
        preds = (probs >= th).astype(int)
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
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "bot_recall": float(bot_recall),
        })
    return pd.DataFrame(rows)


def choose_best_threshold(threshold_df: pd.DataFrame, max_human_block_rate: float = 0.20):
    valid = threshold_df[threshold_df["human_block_rate"] <= max_human_block_rate].copy()
    if len(valid) > 0:
        valid = valid.sort_values(by=["bot_recall", "f1_bot", "accuracy"], ascending=False)
        return float(valid.iloc[0]["threshold"]), "policy_human_block_rate_limited"
    fallback = threshold_df.sort_values(by=["f1_bot", "accuracy"], ascending=False)
    return float(fallback.iloc[0]["threshold"]), "fallback_best_f1"


def calculate_tpr_at_fpr(y_true, probs, target_fprs=(0.01, 0.05, 0.10, 0.20)):
    if len(np.unique(y_true)) < 2:
        return {}
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    result = {}
    for target in target_fprs:
        valid_idx = np.where(fpr <= target)[0]
        if len(valid_idx) == 0:
            result[f"tpr_at_fpr_{target}"] = {"tpr": 0.0, "threshold": None, "fpr": None}
        else:
            best_idx = valid_idx[np.argmax(tpr[valid_idx])]
            result[f"tpr_at_fpr_{target}"] = {
                "tpr": float(tpr[best_idx]),
                "threshold": float(thresholds[best_idx]),
                "fpr": float(fpr[best_idx]),
            }
    return result


def print_eval_report(name: str, y_true, probs, threshold: float):
    preds = (probs >= threshold).astype(int)
    print(f"\n========== {name} 평가 ==========")
    print(f"Threshold: {threshold}")
    auc = safe_roc_auc(y_true, probs)
    pr_auc = safe_pr_auc(y_true, probs)
    print(f"ROC-AUC: {auc:.4f}" if auc is not None else "ROC-AUC: 계산 불가")
    print(f"PR-AUC: {pr_auc:.4f}" if pr_auc is not None else "PR-AUC: 계산 불가")
    print("\nConfusion Matrix [labels: 0=human, 1=bot]")
    print(confusion_matrix(y_true, preds, labels=[0, 1]))
    print("\nClassification Report")
    print(classification_report(y_true, preds, target_names=["human", "bot"], zero_division=0))
    print("\nTPR@FPR")
    for k, v in calculate_tpr_at_fpr(y_true, probs).items():
        print(f"{k}: TPR={v['tpr']:.4f}, FPR={v['fpr']}, threshold={v['threshold']}")


def plot_loss(train_losses, val_losses, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title("Training / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss 그래프 저장: {save_path}")


def plot_prob_distribution(y_true, probs, out_dir):
    human_probs = probs[y_true == 0]
    bot_probs = probs[y_true == 1]
    plt.figure(figsize=(10, 5))
    plt.hist(human_probs, bins=30, alpha=0.6, label="Human")
    plt.hist(bot_probs, bins=30, alpha=0.6, label="Bot")
    plt.title("Predicted Bot Probability Distribution")
    plt.xlabel("Predicted Bot Probability")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "prob_distribution_test.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"확률 분포 그래프 저장: {save_path}")


def plot_threshold_metrics(threshold_df, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(threshold_df["threshold"], threshold_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(threshold_df["threshold"], threshold_df["f1_bot"], marker="o", label="Bot F1")
    plt.plot(threshold_df["threshold"], threshold_df["human_block_rate"], marker="o", label="Human Block Rate")
    plt.plot(threshold_df["threshold"], threshold_df["bot_miss_rate"], marker="o", label="Bot Miss Rate")
    plt.title("Threshold Metrics")
    plt.xlabel("Threshold")
    plt.ylabel("Score / Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "threshold_metrics_test.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Threshold 그래프 저장: {save_path}")


def plot_roc_curve(y_true, probs, out_dir):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate / Human Block Rate")
    plt.ylabel("True Positive Rate / Bot Recall")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "roc_curve_test.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC 그래프 저장: {save_path}")


def plot_pr_curve(y_true, probs, out_dir):
    if len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC={pr_auc:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall / Bot Detection Rate")
    plt.ylabel("Precision / Bot Prediction Trust")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "pr_curve_test.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PR 그래프 저장: {save_path}")


def aggregate_flashlight_scores(scores, allow_total=0.60, block_total=1.40, suspicious_threshold=0.20, high_risk_threshold=0.55, extreme_threshold=0.80):
    if len(scores) == 0:
        raise ValueError("scores must not be empty.")
    total_score = float(sum(scores))
    avg_score = float(total_score / len(scores))
    max_score = float(max(scores))
    min_score = float(min(scores))
    suspicious_count = int(sum(1 for s in scores if s >= suspicious_threshold))
    high_risk_count = int(sum(1 for s in scores if s >= high_risk_threshold))
    if total_score < allow_total and max_score < 0.35:
        decision = "allow"
    elif total_score >= block_total or high_risk_count >= 2 or max_score >= extreme_threshold:
        decision = "block"
    else:
        decision = "challenge_again"
    return {
        "scores": [round(float(s), 4) for s in scores],
        "total_score": round(total_score, 4),
        "avg_score": round(avg_score, 4),
        "max_score": round(max_score, 4),
        "min_score": round(min_score, 4),
        "suspicious_count": suspicious_count,
        "high_risk_count": high_risk_count,
        "decision": decision,
        "policy": {
            "allow_total": allow_total,
            "block_total": block_total,
            "suspicious_threshold": suspicious_threshold,
            "high_risk_threshold": high_risk_threshold,
            "extreme_threshold": extreme_threshold,
        },
    }


class MouseBotDetector:
    def __init__(self, model_path: str, normalizer_path: str, metadata_path: str, device: Optional[torch.device] = None):
        self.device = device or get_device("auto")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.normalizer = joblib.load(normalizer_path)
        self.model = MouseGRUModel(
            seq_size=len(SEQ_FEATURES),
            static_size=len(STATIC_FEATURES),
            hidden=self.metadata["hidden"],
            layers=self.metadata["layers"],
            dropout=self.metadata["dropout"],
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.threshold = float(self.metadata["best_threshold"])

    @torch.no_grad()
    def predict_one(self, sample: Dict) -> Dict:
        seq = self.normalizer.transform_seq(sample)
        static = self.normalizer.transform_static(sample)
        x_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(seq)], dtype=torch.long).to(self.device)
        x_static = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x_seq, lengths, x_static)
        prob = torch.sigmoid(logits)[0].detach().cpu().item()
        pred = "bot" if prob >= self.threshold else "human"
        return {
            "bot_probability": float(prob),
            "threshold": self.threshold,
            "prediction": pred,
            "risk_score": float(prob),
        }


def check_data_path(data_path: str):
    if not os.path.exists(data_path):
        print("\n[에러] 데이터 파일을 찾을 수 없습니다.")
        print(f"입력한 경로: {data_path}")
        raise FileNotFoundError(data_path)


def print_dataset_diagnostics(all_data: List[Dict], out_dir: str):
    rows = []
    for d in all_data:
        rows.append({
            "label": d.get("label"),
            "user_id": d.get("user_id"),
            "bot_type": d.get("bot_type"),
            "image_id": d.get("image_id"),
            "original_file": d.get("original_file"),
            "source_file": d.get("source_file"),
        })
    df = pd.DataFrame(rows)
    print("\n[데이터 진단]")
    print("전체 데이터 수:", len(df))
    print("label 분포:", df["label"].value_counts(dropna=False).to_dict())
    print("user_id 고유 개수:", df["user_id"].nunique(dropna=False))
    print("bot_type 고유 개수:", df["bot_type"].nunique(dropna=False))
    print("image_id 고유 개수:", df["image_id"].nunique(dropna=False))
    print("original_file 고유 개수:", df["original_file"].nunique(dropna=False))
    print("source_file 고유 개수:", df["source_file"].nunique(dropna=False))
    diag_path = os.path.join(out_dir, "dataset_diagnostics.csv")
    df.to_csv(diag_path, index=False)
    print(f"데이터 진단 CSV 저장: {diag_path}")


def main():
    parser = argparse.ArgumentParser(description="Sentient-CAPTCHA Mouse Behavior GRU Trainer - Server GPU Final")
    parser.add_argument("--data", type=str, required=True, help="학습 데이터 JSON 경로")
    parser.add_argument("--out-dir", type=str, default="./runs/mouse_gru_final", help="결과 저장 폴더")
    parser.add_argument("--epochs", type=int, default=30, help="최대 epoch 수")
    parser.add_argument("--batch-size", type=int, default=128, help="배치 크기")
    parser.add_argument("--hidden", type=int, default=32, help="GRU hidden size")
    parser.add_argument("--layers", type=int, default=1, help="GRU layer 수")
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout 비율")
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="AdamW weight decay")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="gradient clipping max norm")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="early stopping 최소 개선폭")
    parser.add_argument("--monitor", type=str, default="val_auc", choices=["val_auc", "val_loss"], help="early stopping 기준")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--group-key", type=str, default="auto", choices=["auto", "user_id", "bot_type", "image_id", "original_file", "source_file"], help="그룹 분리 기준")
    parser.add_argument("--max-human-block-rate", type=float, default=0.20, help="threshold 선택 시 허용할 최대 사람 오탐률")
    parser.add_argument("--seq-noise-std", type=float, default=0.01, help="train sequence noise augmentation std")
    parser.add_argument("--static-noise-std", type=float, default=0.005, help="train static noise augmentation std")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0 등")
    parser.add_argument("--require-gpu", action="store_true", help="GPU 없으면 실행 중단")
    parser.add_argument("--print-every", type=int, default=1, help="몇 epoch마다 로그 출력할지")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = get_device(args.device)
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("require-gpu 옵션이 켜져 있지만 CUDA GPU를 찾지 못했습니다.")
    print_device_info(device)

    check_data_path(args.data)
    with open(args.data, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"\n전체 데이터 수: {len(all_data)}")
    all_labels = [int(d.get("label", 0)) for d in all_data]
    print("전체 label count:", pd.Series(all_labels).value_counts().to_dict())
    if len(set(all_labels)) < 2:
        raise ValueError("label이 한 종류만 있습니다. 0=human, 1=bot 데이터가 모두 필요합니다.")

    print_dataset_diagnostics(all_data, args.out_dir)
    train_raw, val_raw, test_raw = make_train_val_test_split(all_data, args.group_key, args.seed)

    normalizer = MouseFeatureNormalizer()
    normalizer.fit(train_raw)

    train_loader = make_loader(train_raw, normalizer, args.batch_size, True, True, args.seq_noise_std, args.static_noise_std, args.num_workers, device)
    val_loader = make_loader(val_raw, normalizer, args.batch_size, False, False, 0.0, 0.0, args.num_workers, device)
    test_loader = make_loader(test_raw, normalizer, args.batch_size, False, False, 0.0, 0.0, args.num_workers, device)

    model = MouseGRUModel(len(SEQ_FEATURES), len(STATIC_FEATURES), args.hidden, args.layers, args.dropout).to(device)
    pos_weight = get_pos_weight(train_raw, device)
    print(f"pos_weight: {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if args.monitor == "val_auc" else "min",
        factor=0.5,
        patience=2,
    )

    train_losses, val_losses, history_rows = [], [], []
    best_score = -float("inf") if args.monitor == "val_auc" else float("inf")
    best_model_state, best_epoch, patience_count = None, 0, 0

    print("\n학습 시작")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_probs, val_labels = predict_probs(model, val_loader, device)
        val_auc = safe_roc_auc(val_labels, val_probs)
        val_pr_auc = safe_pr_auc(val_labels, val_probs)
        val_auc_value = float(val_auc) if val_auc is not None else 0.0
        val_pr_auc_value = float(val_pr_auc) if val_pr_auc is not None else 0.0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_score = val_auc_value if args.monitor == "val_auc" else val_loss
        if args.monitor == "val_auc":
            improved = current_score > best_score + args.min_delta
        else:
            improved = current_score < best_score - args.min_delta

        if improved:
            best_score = current_score
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        scheduler.step(current_score)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc_value,
            "val_pr_auc": val_pr_auc_value,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
        }
        history_rows.append(row)

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val ROC-AUC: {val_auc_value:.4f} "
                f"Val PR-AUC: {val_pr_auc_value:.4f} "
                f"Best Epoch: {best_epoch} "
                f"Patience: {patience_count}/{args.patience}"
            )
        if patience_count >= args.patience:
            print(f"\nEarly stopping 발생. epoch={epoch}, best_epoch={best_epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model 로드 완료: epoch={best_epoch}, monitor={args.monitor}, best_score={best_score:.4f}")

    history_df = pd.DataFrame(history_rows)
    history_path = os.path.join(args.out_dir, "train_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"학습 히스토리 저장: {history_path}")

    val_probs, val_labels = predict_probs(model, val_loader, device)
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    val_threshold_df = evaluate_thresholds(val_labels, val_probs, thresholds)
    best_threshold, threshold_reason = choose_best_threshold(val_threshold_df, args.max_human_block_rate)
    print("\n========== Validation Threshold 결과 ==========")
    print(val_threshold_df.to_string(index=False))
    print(f"\n선택된 threshold: {best_threshold}")
    print(f"선택 이유: {threshold_reason}")
    val_threshold_csv_path = os.path.join(args.out_dir, "threshold_metrics_validation.csv")
    val_threshold_df.to_csv(val_threshold_csv_path, index=False)

    test_probs, test_labels = predict_probs(model, test_loader, device)
    print_eval_report("Test", test_labels, test_probs, best_threshold)
    test_threshold_df = evaluate_thresholds(test_labels, test_probs, thresholds)
    print("\n========== Test Threshold 결과 ==========")
    print(test_threshold_df.to_string(index=False))
    test_threshold_csv_path = os.path.join(args.out_dir, "threshold_metrics_test.csv")
    test_threshold_df.to_csv(test_threshold_csv_path, index=False)

    test_preds = (test_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    test_auc = safe_roc_auc(test_labels, test_probs)
    test_pr_auc = safe_pr_auc(test_labels, test_probs)
    test_summary = {
        "best_epoch": best_epoch,
        "monitor": args.monitor,
        "best_monitor_score": float(best_score),
        "best_threshold": best_threshold,
        "threshold_reason": threshold_reason,
        "test_roc_auc": float(test_auc) if test_auc is not None else None,
        "test_pr_auc": float(test_pr_auc) if test_pr_auc is not None else None,
        "test_accuracy": float(accuracy_score(test_labels, test_preds)),
        "test_precision_bot": float(precision_score(test_labels, test_preds, zero_division=0)),
        "test_recall_bot": float(recall_score(test_labels, test_preds, zero_division=0)),
        "test_f1_bot": float(f1_score(test_labels, test_preds, zero_division=0)),
        "test_human_block_rate": float(fp / max(fp + tn, 1)),
        "test_bot_miss_rate": float(fn / max(fn + tp, 1)),
        "confusion_matrix": {
            "tn_human_correct": int(tn),
            "fp_human_blocked": int(fp),
            "fn_bot_missed": int(fn),
            "tp_bot_detected": int(tp),
        },
        "tpr_at_fpr": calculate_tpr_at_fpr(test_labels, test_probs),
        "hyperparameters": vars(args),
    }
    summary_path = os.path.join(args.out_dir, "final_summary.json")
    save_json(test_summary, summary_path)
    print("\n========== 최종 요약 ==========")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))
    print(f"최종 요약 저장: {summary_path}")

    model_path = os.path.join(args.out_dir, "mouse_gru_server_final.pth")
    normalizer_path = os.path.join(args.out_dir, "mouse_normalizer_server_final.joblib")
    metadata_path = os.path.join(args.out_dir, "mouse_metadata_server_final.json")
    service_policy_path = os.path.join(args.out_dir, "three_attempt_service_policy.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump(normalizer, normalizer_path)
    metadata = {
        "model_name": "MouseGRUModel",
        "seq_features": SEQ_FEATURES,
        "static_features": STATIC_FEATURES,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "threshold_reason": threshold_reason,
        "group_key": args.group_key,
        "label_rule": "0=human, 1=bot",
        "max_human_block_rate": args.max_human_block_rate,
        "data_path": args.data,
        "normalization": "StandardScaler fitted on train only",
        "overfitting_controls": {
            "early_stopping": True,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "gradient_clipping": args.grad_clip,
            "seq_noise_std": args.seq_noise_std,
            "static_noise_std": args.static_noise_std,
            "layer_norm": True,
        },
    }
    save_json(metadata, metadata_path)

    service_policy = {
        "description": "3회 손전등 CAPTCHA 수행 후 bot_probability를 누적하여 최종 판정",
        "example": aggregate_flashlight_scores([0.18, 0.42, 0.61]),
        "policy_function": {
            "allow": "total_score < 0.60 and max_score < 0.35",
            "block": "total_score >= 1.40 or high_risk_count >= 2 or max_score >= 0.80",
            "challenge_again": "otherwise",
        },
    }
    save_json(service_policy, service_policy_path)

    print("\n========== 저장 완료 ==========")
    print(f"Model       : {model_path}")
    print(f"Normalizer  : {normalizer_path}")
    print(f"Metadata    : {metadata_path}")
    print(f"Val Metrics : {val_threshold_csv_path}")
    print(f"Test Metrics: {test_threshold_csv_path}")
    print(f"Summary     : {summary_path}")
    print(f"3회 정책    : {service_policy_path}")

    plot_loss(train_losses, val_losses, args.out_dir)
    plot_prob_distribution(test_labels, test_probs, args.out_dir)
    plot_threshold_metrics(test_threshold_df, args.out_dir)
    plot_roc_curve(test_labels, test_probs, args.out_dir)
    plot_pr_curve(test_labels, test_probs, args.out_dir)

    print("\n완료. 결과 저장 폴더:")
    print(args.out_dir)


if __name__ == "__main__":
    main()
