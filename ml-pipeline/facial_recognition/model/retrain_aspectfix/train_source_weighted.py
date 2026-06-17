"""
v3(종횡비 수정 + 시간 정규화) 데이터 위에 R_live_clip source-weighted sampling을
적용한 재학습 + threshold sweep 실험.

배경: RETROSPECTIVE_facial_recognition_pipeline.md 8장 참고.
  v1(버그) R_live FRR  95.24%
  v3(버그 수정 + 시간정규화, weight 없음) R_live FRR  80.95% (threshold=best_f1)
  → 81%는 여전히 운영 기준에 부적합. R_live_clip을 train 샘플러에서 더 자주
    뽑도록 가중치를 줘서 추가 개선 여지가 있는지 확인한다.

같은 hidden/lr/epochs/patience, 같은 split(이미 npz에 고정됨)을 유지하고
train DataLoader의 sampler만 WeightedRandomSampler로 바꾼다.

Usage:
  python train_source_weighted.py --weight 5 --out runs/gru_v3_aspectfix_w5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from facial_recognition.model.face_liveness_gru import FaceLivenessGRU
from facial_recognition.data.face_clip_dataset import load_npz, split_dataset, metadata_frame
from facial_recognition.evaluation.metrics import (
    binary_metrics, choose_threshold, grouped_metrics, safe_auc, threshold_sweep,
)
from facial_recognition.training.train_gru import predict, run_epoch


DATA_PATH = Path(__file__).parent / "face_clip_data_time_norm.npz"


def make_weighted_sampler(train_df: pd.DataFrame, target: str, weight: float):
    weights = np.ones(len(train_df), dtype=np.float64)
    mask = train_df["source_group"].astype(str).values == target
    weights[mask] *= float(weight)
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=float, required=True, help="R_live_clip 샘플링 가중치")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_npz(str(DATA_PATH))
    (
        ds_train, ds_valid, ds_test,
        scaler, selected_features, selected_idx,
        df, train_idx, valid_idx, test_idx,
    ) = split_dataset(raw, feature_mode="all", min_seq_len=1, min_face_rate=0.0)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    n_r_live = int((train_df["source_group"] == "R_live_clip").sum())
    print(f"train R_live_clip 샘플 수: {n_r_live} / {len(train_df)}  (weight x{args.weight})")

    sampler = make_weighted_sampler(train_df, "R_live_clip", args.weight)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler)
    valid_loader = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False)

    device = args.device
    input_dim = ds_train.x.shape[-1]
    model = FaceLivenessGRU(
        input_dim=input_dim, hidden_size=args.hidden_size, num_layers=1, dropout=args.dropout
    ).to(device)

    y_train = ds_train.y.numpy().astype(int)
    neg = max(1, int((y_train == 0).sum()))
    pos = max(1, int((y_train == 1).sum()))
    pos_wt = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device}  train={len(ds_train)}  valid={len(ds_valid)}  test={len(ds_test)}  params={n_params}")

    history = []
    best_score = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
        val_y, val_prob = predict(model, valid_loader, device)
        val_auc = safe_auc(val_y, val_prob)
        val_m = binary_metrics(val_y, val_prob, threshold=0.5)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_auc": val_auc,
            "val_f1_0_5": val_m["f1_spoof"],
            "val_far_0_5": val_m["far_attack_pass_rate"],
            "val_frr_0_5": val_m["frr_genuine_reject_rate"],
        })
        print(f"epoch={epoch:03d} loss={train_loss:.4f} val_auc={val_auc:.4f} "
              f"f1={val_m['f1_spoof']:.4f} far={val_m['far_attack_pass_rate']:.4f} "
              f"frr={val_m['frr_genuine_reject_rate']:.4f}")

        score = val_auc if not np.isnan(val_auc) else val_m["f1_spoof"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            print(f"early stopping at epoch={epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_y, val_prob = predict(model, valid_loader, device)
    threshold = choose_threshold(val_y, val_prob, strategy="best_f1")

    result_rows, group_rows = [], []
    sweep_rows = []
    for split_name, loader, sub_df in [
        ("valid", valid_loader, df.iloc[valid_idx]),
        ("test",  test_loader,  df.iloc[test_idx]),
    ]:
        yy, prob = predict(model, loader, device)
        m = binary_metrics(yy, prob, threshold=threshold)
        m.update({"model": f"gru_v3_w{args.weight}", "split": split_name,
                   "parameter_count": n_params})
        result_rows.append(m)

        for col in ["attack_type", "source_group"]:
            group_rows.append(grouped_metrics(
                yy, prob, sub_df[col].values, threshold=threshold,
                group_name=col, model_name=f"gru_v3_w{args.weight}", split_name=split_name,
            ))

        # threshold sweep 0.20~0.95 (0.05 step), source_group/attack_type 별로도 기록
        sweep_th = np.round(np.arange(0.20, 0.951, 0.05), 2)
        for th in sweep_th:
            row = {"split": split_name, "threshold": float(th), "weight": args.weight}
            overall = binary_metrics(yy, prob, threshold=th)
            row["overall_roc_auc"] = overall["roc_auc"]
            row["overall_frr"] = overall["frr_genuine_reject_rate"]
            row["overall_attack_block_rate"] = overall["attack_block_rate"]

            sg = sub_df["source_group"].values
            at = sub_df["attack_type"].values

            r_live_mask = sg == "R_live_clip"
            if r_live_mask.sum() > 0:
                r_live_m = binary_metrics(yy[r_live_mask], prob[r_live_mask], threshold=th)
                row["r_live_frr"] = r_live_m["frr_genuine_reject_rate"]
                row["r_live_n"] = int(r_live_mask.sum())

            live_mask = yy == 0
            if live_mask.sum() > 0:
                live_m = binary_metrics(yy[live_mask], prob[live_mask], threshold=th)
                row["live_overall_frr"] = live_m["frr_genuine_reject_rate"]

            for atk in ["print", "replay"]:
                atk_mask = at == atk
                if atk_mask.sum() > 0:
                    atk_m = binary_metrics(yy[atk_mask], prob[atk_mask], threshold=th)
                    row[f"{atk}_block_rate"] = atk_m["attack_block_rate"]

            sweep_rows.append(row)

    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    pd.DataFrame(result_rows).to_csv(out / "model_results.csv", index=False)
    pd.concat(group_rows, ignore_index=True).to_csv(out / "group_results.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(out / "threshold_sweep.csv", index=False)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim, "hidden_size": args.hidden_size, "num_layers": 1,
        "dropout": args.dropout, "threshold": threshold,
        "selected_features": selected_features, "selected_idx": selected_idx,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "pos_weight": float(pos_wt.item()),
    }, out / "best_gru.pt")
    joblib.dump(scaler, out / "seq_scaler.joblib")

    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data": str(DATA_PATH), "out": str(out), "feature_mode": "all",
            "source_weight_target": "R_live_clip", "source_weight": args.weight,
            "epochs": args.epochs, "hidden_size": args.hidden_size, "lr": args.lr,
            "patience": args.patience, "threshold": threshold,
            "selected_features": selected_features, "parameter_count": n_params,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n완료: {out}  (best_f1 threshold={threshold:.3f})")


if __name__ == "__main__":
    main()
