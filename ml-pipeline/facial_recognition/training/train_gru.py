"""
GRU 얼굴 활성도 모델 학습 루프.

python -m facial_recognition.scripts.train_gru --data face_clip_data.npz --out runs/gru_v1
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from facial_recognition.model.face_liveness_gru import FaceLivenessGRU
from facial_recognition.data.face_clip_dataset import (
    load_npz,
    split_dataset,
    metadata_frame,
)
from facial_recognition.evaluation.metrics import (
    binary_metrics,
    choose_threshold,
    grouped_metrics,
    safe_auc,
)


# ── 추론 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: FaceLivenessGRU, loader: DataLoader, device: str):
    model.eval()
    probs, ys = [], []
    for x, y, lengths in loader:
        x       = x.to(device)
        lengths = lengths.to(device)
        logits  = model(x, lengths)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


# ── 학습 에폭 ─────────────────────────────────────────────────────────────────

def run_epoch(
    model: FaceLivenessGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.train()
    losses = []
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss   = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


# ── 메인 학습 함수 ────────────────────────────────────────────────────────────

def train(
    data_path: str,
    out_dir: str,
    feature_mode: str = "all",
    min_seq_len: int = 1,
    min_face_rate: float = 0.0,
    epochs: int = 80,
    batch_size: int = 64,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.3,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    threshold_strategy: str = "best_f1",
    max_frr: float | None = None,
    device: str = "auto",
    seed: int = 42,
    num_threads: int = 1,
) -> Path:
    """
    GRU 모델을 학습하고 아티팩트를 out_dir에 저장한다.

    Returns
    -------
    out : Path — 출력 디렉토리
    """
    torch.set_num_threads(num_threads)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 데이터 준비 ────────────────────────────────────────────────────────────
    raw_data = load_npz(data_path)
    (
        ds_train, ds_valid, ds_test,
        scaler, selected_features, selected_idx,
        df, train_idx, valid_idx, test_idx,
    ) = split_dataset(raw_data, feature_mode, min_seq_len, min_face_rate)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)

    # ── 모델 / 옵티마이저 ─────────────────────────────────────────────────────
    input_dim = ds_train.x.shape[-1]
    model = FaceLivenessGRU(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    y_train  = ds_train.y.numpy().astype(int)
    neg      = max(1, int((y_train == 0).sum()))
    pos      = max(1, int((y_train == 1).sum()))
    pos_wt   = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    loss_fn  = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device}  features={selected_features}")
    print(f"train={len(ds_train)}  valid={len(ds_valid)}  test={len(ds_test)}")
    print(f"pos_weight={pos_wt.item():.4f}  params={n_params}")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    history    = []
    best_score = -1.0
    best_state = None
    bad        = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
        val_y, val_prob = predict(model, valid_loader, device)
        val_auc = safe_auc(val_y, val_prob)
        val_m   = binary_metrics(val_y, val_prob, threshold=0.5)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "val_f1_0_5":  val_m["f1_spoof"],
            "val_far_0_5": val_m["far_attack_pass_rate"],
            "val_frr_0_5": val_m["frr_genuine_reject_rate"],
        })

        print(
            f"epoch={epoch:03d}  loss={train_loss:.4f}  "
            f"val_auc={val_auc:.4f}  f1={val_m['f1_spoof']:.4f}  "
            f"far={val_m['far_attack_pass_rate']:.4f}  frr={val_m['frr_genuine_reject_rate']:.4f}"
        )

        score = val_auc if not np.isnan(val_auc) else val_m["f1_spoof"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            print(f"early stopping at epoch={epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── 임계값 선택 ───────────────────────────────────────────────────────────
    val_y, val_prob = predict(model, valid_loader, device)
    threshold = choose_threshold(val_y, val_prob, strategy=threshold_strategy, max_frr=max_frr)

    # ── 평가 ──────────────────────────────────────────────────────────────────
    result_rows = []
    group_rows  = []

    for split_name, loader, sub_df in [
        ("valid", valid_loader, df.iloc[valid_idx]),
        ("test",  test_loader,  df.iloc[test_idx]),
    ]:
        t0 = time.perf_counter()
        yy, prob = predict(model, loader, device)
        infer_ms = (time.perf_counter() - t0) / max(1, len(yy)) * 1000

        m = binary_metrics(yy, prob, threshold=threshold)
        m.update({
            "model": f"gru_{feature_mode}",
            "split": split_name,
            "inference_time_ms_per_sample": infer_ms,
            "parameter_count": n_params,
            "feature_mode": feature_mode,
            "selected_feature_count": len(selected_features),
        })
        result_rows.append(m)

        for col in ["attack_type", "source_group"]:
            if col in sub_df.columns:
                group_rows.append(grouped_metrics(
                    yy, prob, sub_df[col].values,
                    threshold=threshold,
                    group_name=col,
                    model_name=f"gru_{feature_mode}",
                    split_name=split_name,
                ))

    # ── 저장 ──────────────────────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    pd.DataFrame(result_rows).to_csv(out / "model_results.csv", index=False)
    if group_rows:
        pd.concat(group_rows, ignore_index=True).to_csv(out / "group_results.csv", index=False)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim":        input_dim,
        "hidden_size":      hidden_size,
        "num_layers":       num_layers,
        "dropout":          dropout,
        "threshold":        threshold,
        "selected_features": selected_features,
        "selected_idx":     selected_idx,
        "scaler_mean":      scaler.mean_.tolist(),
        "scaler_scale":     scaler.scale_.tolist(),
        "pos_weight":       float(pos_wt.item()),
    }, out / "best_gru.pt")

    joblib.dump(scaler, out / "seq_scaler.joblib")

    import json
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data": data_path,
            "out": out_dir,
            "feature_mode": feature_mode,
            "min_seq_len": min_seq_len,
            "min_face_rate": min_face_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "threshold_strategy": threshold_strategy,
            "device_used": device,
            "selected_features": selected_features,
            "threshold": threshold,
            "parameter_count": n_params,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {out}/")
    print(f"  best_gru.pt  seq_scaler.joblib  run_config.json")
    print(f"  model_results.csv  group_results.csv  train_history.csv")
    return out
