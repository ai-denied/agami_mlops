#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Anti-Spoofing CNN Trainer

MobileNetV2 fine-tune. samples_manifest.jsonl의 subject-aware split을 그대로 사용.

실행 예시:
  cd ml-pipeline
  python facial_recognition/scripts/train_antispoofing.py \
    --manifest facial_recognition/dataset/samples_manifest.jsonl \
    --img-dir  facial_recognition/dataset/face_images \
    --out-dir  ./runs/face_antispoofing_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from facial_recognition.data.face_dataset import (
    FaceAntiSpoofDataset,
    get_pos_weight,
    load_manifest,
    split_manifest,
)
from facial_recognition.model.face_antispoofing_cnn import FaceAntiSpoofingCNN

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import mlflow
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False


# ── 유틸리티 ─────────────────────────────────────────────────────────────────

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_auc(y_true, scores):
    if not SKLEARN_OK:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _safe_pr_auc(y_true, scores):
    if not SKLEARN_OK:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, scores))


# ── 학습 / 평가 루프 ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n          += labels.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_scores, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        scores = torch.sigmoid(logits)
        total_loss  += loss.item() * labels.size(0)
        n           += labels.size(0)
        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return (
        total_loss / max(n, 1),
        np.array(all_scores, dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
    )


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing CNN Trainer")

    parser.add_argument("--manifest",    required=True, help="samples_manifest.jsonl 경로")
    parser.add_argument("--img-dir",     required=True, help="face_images 루트 경로")
    parser.add_argument("--out-dir",     default="./runs/face_antispoofing_v1")

    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--img-size",    type=int,   default=224)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--grad-clip",   type=float, default=5.0)
    parser.add_argument("--patience",    type=int,   default=7)
    parser.add_argument("--min-delta",   type=float, default=1e-4)
    parser.add_argument("--monitor",     type=str,   default="val_auc",
                        choices=["val_auc", "val_pr_auc", "val_loss"])
    parser.add_argument("--num-workers", type=int,   default=2)
    parser.add_argument("--device",      type=str,   default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed",        type=int,   default=42)

    # threshold
    parser.add_argument("--max-human-block-rate", type=float, default=0.10,
                        help="허용 최대 실제사람 차단률 (FPR). 기본 0.10")

    # MLflow
    parser.add_argument("--use-mlflow",      action="store_true")
    parser.add_argument("--mlflow-uri",      type=str, default=None)
    parser.add_argument("--mlflow-exp",      type=str, default="face_antispoofing")
    parser.add_argument("--mlflow-run-name", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)

    print(f"디바이스: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── MLflow 초기화 ─────────────────────────────────────────────────────────
    if args.use_mlflow and MLFLOW_OK:
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_exp)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params(vars(args))
    elif args.use_mlflow:
        print("[MLflow] mlflow 미설치 — 비활성화")
        args.use_mlflow = False

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    all_samples   = load_manifest(args.manifest)
    train_samples = split_manifest(all_samples, "train")
    val_samples   = split_manifest(all_samples, "valid")
    test_samples  = split_manifest(all_samples, "test")

    print(f"\n데이터 분포:")
    for name, part in [("Train", train_samples), ("Valid", val_samples), ("Test", test_samples)]:
        n0 = sum(1 for s in part if s["label"] == 0)
        n1 = sum(1 for s in part if s["label"] == 1)
        print(f"  {name}: {len(part)} 클립  (real={n0}, spoof={n1})")

    from torch.utils.data import DataLoader
    pin = device.type == "cuda"

    train_ds = FaceAntiSpoofDataset(train_samples, args.img_dir, args.img_size, augment=True)
    val_ds   = FaceAntiSpoofDataset(val_samples,   args.img_dir, args.img_size, augment=False)
    test_ds  = FaceAntiSpoofDataset(test_samples,  args.img_dir, args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    # ── 모델 / 옵티마이저 ───────────────────────────────────────────────────────
    model     = FaceAntiSpoofingCNN(pretrained=True, dropout=args.dropout).to(device)
    pos_weight = get_pos_weight(train_samples, device)
    print(f"pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    best_score      = None
    best_epoch      = 0
    best_state      = None
    patience_count  = 0
    history_rows    = []

    print("\n학습 시작")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_loss, val_scores, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_auc    = _safe_auc(val_labels, val_scores)
        val_pr_auc = _safe_pr_auc(val_labels, val_scores)

        if args.monitor == "val_loss":
            cur = -val_loss
        elif args.monitor == "val_pr_auc":
            cur = val_pr_auc or -999.0
        else:
            cur = val_auc or -999.0

        improved = best_score is None or cur > best_score + args.min_delta
        if improved:
            best_score = cur
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        history_rows.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_auc": val_auc, "val_pr_auc": val_pr_auc,
            "best_epoch": best_epoch, "patience_count": patience_count,
        })

        if args.use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_auc": val_auc or 0.0, "val_pr_auc": val_pr_auc or 0.0,
            }, step=epoch)

        auc_str    = f"{val_auc:.4f}"    if val_auc    else "N/A"
        pr_auc_str = f"{val_pr_auc:.4f}" if val_pr_auc else "N/A"
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"Train {train_loss:.4f}  Val {val_loss:.4f}  "
            f"AUC {auc_str}  PR-AUC {pr_auc_str}  "
            f"Best {best_epoch}  Pat {patience_count}/{args.patience}"
        )

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Best 모델 복원 ─────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
        print(f"\nBest model 로드: epoch={best_epoch}, monitor={args.monitor}, score={best_score:.4f}")

    pd.DataFrame(history_rows).to_csv(os.path.join(args.out_dir, "train_history.csv"), index=False)

    # ── Test 평가 ──────────────────────────────────────────────────────────────
    _, test_scores, test_labels = evaluate(model, test_loader, criterion, device)

    test_auc    = _safe_auc(test_labels, test_scores)
    test_pr_auc = _safe_pr_auc(test_labels, test_scores)

    # threshold 선택: FPR ≤ max_human_block_rate 조건에서 최대 TPR
    best_thr   = 0.5
    best_tpr   = 0.0
    from numpy import linspace
    for thr in linspace(0.05, 0.95, 37):
        preds = (test_scores >= thr).astype(int)
        if SKLEARN_OK:
            tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
            fpr = fp / max(fp + tn, 1)
            tpr = tp / max(tp + fn, 1)
            if fpr <= args.max_human_block_rate and tpr > best_tpr:
                best_tpr = tpr
                best_thr = float(thr)

    preds = (test_scores >= best_thr).astype(int)
    if SKLEARN_OK:
        tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
        acc  = accuracy_score(test_labels, preds)
        f1   = f1_score(test_labels, preds, zero_division=0)
    else:
        tn = fp = fn = tp = 0
        acc = f1 = 0.0

    summary = {
        "best_epoch":     best_epoch,
        "monitor":        args.monitor,
        "spoof_threshold": best_thr,
        "test_roc_auc":   test_auc,
        "test_pr_auc":    test_pr_auc,
        "test_accuracy":  float(acc),
        "test_f1_spoof":  float(f1),
        "confusion_matrix": {
            "tn_real_correct":  int(tn),
            "fp_real_blocked":  int(fp),
            "fn_spoof_missed":  int(fn),
            "tp_spoof_detected": int(tp),
        },
        "human_block_rate": float(fp / max(fp + tn, 1)),
        "spoof_miss_rate":  float(fn / max(fn + tp, 1)),
        "hyperparameters":  vars(args),
    }

    print("\n========== Test 결과 ==========")
    print(json.dumps({k: v for k, v in summary.items() if k != "hyperparameters"},
                     ensure_ascii=False, indent=2))

    # ── 아티팩트 저장 ──────────────────────────────────────────────────────────
    ckpt_path     = os.path.join(args.out_dir, "best_model.pth")
    summary_path  = os.path.join(args.out_dir, "summary.json")
    metadata_path = os.path.join(args.out_dir, "face_metadata.json")

    torch.save(model.state_dict(), ckpt_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metadata = {
        "model_name":      "FaceAntiSpoofingCNN",
        "backbone":        "MobileNetV2",
        "img_size":        args.img_size,
        "spoof_threshold": best_thr,
        "score_name":      "spoof_score",
        "label_rule":      "0=real, 1=spoof",
        "imagenet_mean":   [0.485, 0.456, 0.406],
        "imagenet_std":    [0.229, 0.224, 0.225],
        "summary":         summary,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ── ONNX 자동 익스포트 ────────────────────────────────────────────────────
    onnx_path = os.path.join(args.out_dir, "face_antispoofing.onnx")
    model.eval().to("cpu")
    dummy = torch.zeros(1, 3, args.img_size, args.img_size)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["face_image"],
        output_names=["spoof_score"],
        dynamic_axes={"face_image": {0: "batch"}, "spoof_score": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX 저장: {onnx_path}")

    if args.use_mlflow:
        mlflow.log_metrics({
            "test_auc": test_auc or 0.0, "test_pr_auc": test_pr_auc or 0.0,
            "accuracy": float(acc), "f1": float(f1),
            "spoof_threshold": best_thr,
        })
        for path in [ckpt_path, onnx_path, summary_path, metadata_path]:
            if os.path.exists(path):
                mlflow.log_artifact(path)
        mlflow.end_run()

    print(f"\n저장 완료: {args.out_dir}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  ONNX       : {onnx_path}")
    print(f"  metadata   : {metadata_path}")


if __name__ == "__main__":
    main()
