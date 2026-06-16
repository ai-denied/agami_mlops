import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_utils import (
    load_npz,
    metadata_frame,
    select_seq_features,
    fit_seq_scaler,
    transform_seq_with_lengths,
)
from metrics import binary_metrics, choose_threshold, grouped_metrics, safe_auc
from train_gru import GRUClassifier, SeqDataset, predict, run_epoch, count_params


def ensure_source_group(df):
    """
    metadata_frame에서 source_group이 없을 경우 sample_id 기준으로 대략 추론.
    """
    if "source_group" in df.columns:
        return df

    def infer_source_group(sample_id):
        sid = str(sample_id)
        if sid.startswith("ATK"):
            return "ATK_external_clip"
        if sid.startswith("R"):
            return "R_live_clip"
        return "S_dataset_sequence"

    if "sample_id" in df.columns:
        df["source_group"] = df["sample_id"].apply(infer_source_group)
    else:
        df["source_group"] = "unknown"

    return df


def make_weighted_sampler(train_df, source_weight_target, source_weight, live_weight=1.0, spoof_weight=1.0):
    """
    train set 내부에서 특정 source를 더 자주 샘플링하기 위한 sampler.

    기본:
    - 모든 샘플 weight = 1
    - R_live_clip 같은 target source는 weight를 source_weight배
    - 필요하면 live/spoof class weight도 추가 가능
    """
    weights = np.ones(len(train_df), dtype=np.float64)

    if "y" in train_df.columns:
        weights[train_df["y"].values == 0] *= float(live_weight)
        weights[train_df["y"].values == 1] *= float(spoof_weight)

    if "source_group" in train_df.columns:
        mask = train_df["source_group"].astype(str).values == str(source_weight_target)
        weights[mask] *= float(source_weight)

    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler, weights.numpy()


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)

    p.add_argument(
        "--feature-mode",
        default="all",
        choices=[
            "all",
            "no_abs",
            "motion_only",
            "eye_mouth",
            "head",
            "no_abs_motion_head",
        ],
    )

    p.add_argument("--min-seq-len", type=int, default=1)
    p.add_argument("--min-face-rate", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=12)

    p.add_argument(
        "--threshold-strategy",
        default="best_f1",
        choices=["best_f1", "eer_like", "low_far", "default"],
    )
    p.add_argument("--max-frr", type=float, default=None)

    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-threads", type=int, default=1)

    # Source-balanced options
    p.add_argument("--source-weight-target", default="R_live_clip")
    p.add_argument("--source-weight", type=float, default=5.0)
    p.add_argument("--live-weight", type=float, default=1.0)
    p.add_argument("--spoof-weight", type=float, default=1.0)

    # Optional loss adjustment
    p.add_argument("--pos-weight-scale", type=float, default=1.0)

    args = p.parse_args()

    torch.set_num_threads(args.num_threads)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.data)
    df = metadata_frame(data)
    df = ensure_source_group(df)

    keep = np.ones(len(df), dtype=bool)
    keep &= data["seq_lengths"].astype(int) >= args.min_seq_len

    if "face_detect_rates" in data:
        keep &= data["face_detect_rates"].astype(float) >= args.min_face_rate

    x_seq = data["x_seq"][keep].astype(np.float32)
    y = data["y"][keep].astype(int)
    lengths = data["seq_lengths"][keep].astype(int)

    df = df[keep].reset_index(drop=True)
    df["y"] = y

    feature_names = [str(x) for x in data["seq_feature_names"]]
    selected_idx = select_seq_features(feature_names, args.feature_mode)
    selected_features = [feature_names[i] for i in selected_idx]

    x_seq = x_seq[:, :, selected_idx]

    train_idx = np.where(df["split"].values == "train")[0]
    valid_idx = np.where(df["split"].values == "valid")[0]
    test_idx = np.where(df["split"].values == "test")[0]

    scaler = fit_seq_scaler(x_seq, lengths, train_idx)
    x_seq = transform_seq_with_lengths(x_seq, lengths, scaler)

    x_train, y_train, len_train = x_seq[train_idx], y[train_idx], lengths[train_idx]
    x_valid, y_valid, len_valid = x_seq[valid_idx], y[valid_idx], lengths[valid_idx]
    x_test, y_test, len_test = x_seq[test_idx], y[test_idx], lengths[test_idx]

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)

    sampler, sample_weights = make_weighted_sampler(
        train_df=train_df,
        source_weight_target=args.source_weight_target,
        source_weight=args.source_weight,
        live_weight=args.live_weight,
        spoof_weight=args.spoof_weight,
    )

    train_loader = DataLoader(
        SeqDataset(x_train, y_train, len_train),
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
    )

    valid_loader = DataLoader(
        SeqDataset(x_valid, y_valid, len_valid),
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        SeqDataset(x_test, y_test, len_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = GRUClassifier(
        input_dim=x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    neg = max(1, int((y_train == 0).sum()))
    pos = max(1, int((y_train == 1).sum()))

    base_pos_weight = neg / pos
    final_pos_weight = base_pos_weight * float(args.pos_weight_scale)

    pos_weight = torch.tensor(
        [final_pos_weight],
        dtype=torch.float32,
        device=device,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"device={device}")
    print(f"feature_mode={args.feature_mode}")
    print(f"selected_features={selected_features}")
    print(f"train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")
    print(f"params={count_params(model)}")
    print(f"base_pos_weight={base_pos_weight:.4f}, final_pos_weight={final_pos_weight:.4f}")
    print(f"source_weight_target={args.source_weight_target}, source_weight={args.source_weight}")
    print("\n[train source distribution]")
    print(train_df.groupby(["source_group", "y"]).size().to_string())
    print("\n[train sample weight summary]")
    print(pd.Series(sample_weights).describe().to_string())

    history = []
    best_score = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)

        val_y, val_prob = predict(model, valid_loader, device)
        val_auc = safe_auc(val_y, val_prob)
        val_metrics = binary_metrics(val_y, val_prob, threshold=0.5)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "val_f1_0_5": val_metrics["f1_spoof"],
            "val_far_0_5": val_metrics["far_attack_pass_rate"],
            "val_frr_0_5": val_metrics["frr_genuine_reject_rate"],
        }

        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"val_auc={val_auc:.4f} "
            f"val_f1={val_metrics['f1_spoof']:.4f} "
            f"far={val_metrics['far_attack_pass_rate']:.4f} "
            f"frr={val_metrics['frr_genuine_reject_rate']:.4f}"
        )

        score = val_auc if not np.isnan(val_auc) else val_metrics["f1_spoof"]

        if score > best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

        if bad >= args.patience:
            print(f"early stopping at epoch={epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_y, val_prob = predict(model, valid_loader, device)

    threshold = choose_threshold(
        val_y,
        val_prob,
        strategy=args.threshold_strategy,
        max_frr=args.max_frr,
    )

    result_rows = []
    group_rows = []

    for split_name, loader, subdf in [
        ("valid", valid_loader, df.iloc[valid_idx]),
        ("test", test_loader, df.iloc[test_idx]),
    ]:
        t0 = time.perf_counter()
        yy, prob = predict(model, loader, device)
        infer_time = (time.perf_counter() - t0) / max(1, len(yy))

        m = binary_metrics(yy, prob, threshold=threshold)
        m.update({
            "model": f"gru_source_weighted_{args.feature_mode}",
            "split": split_name,
            "inference_time_ms_per_sample": infer_time * 1000,
            "parameter_count": count_params(model),
            "feature_mode": args.feature_mode,
            "selected_feature_count": len(selected_features),
            "min_seq_len": args.min_seq_len,
            "min_face_rate": args.min_face_rate,
            "threshold_strategy": args.threshold_strategy,
            "source_weight_target": args.source_weight_target,
            "source_weight": args.source_weight,
            "live_weight": args.live_weight,
            "spoof_weight": args.spoof_weight,
            "pos_weight_scale": args.pos_weight_scale,
        })
        result_rows.append(m)

        for col in ["attack_type", "source_group"]:
            gm = grouped_metrics(
                yy,
                prob,
                subdf[col].values,
                threshold=threshold,
                group_name=col,
                model_name=f"gru_source_weighted_{args.feature_mode}",
                split_name=split_name,
            )
            gm["source_weight_target"] = args.source_weight_target
            gm["source_weight"] = args.source_weight
            gm["live_weight"] = args.live_weight
            gm["spoof_weight"] = args.spoof_weight
            gm["pos_weight_scale"] = args.pos_weight_scale
            group_rows.append(gm)

    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    pd.DataFrame(result_rows).to_csv(out / "model_results.csv", index=False)

    if group_rows:
        pd.concat(group_rows, ignore_index=True).to_csv(out / "group_results.csv", index=False)

    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "selected_features": selected_features,
        "selected_idx": selected_idx,
        "seq_feature_names": feature_names,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "threshold": threshold,
        "input_dim": x_train.shape[-1],
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "base_pos_weight": float(base_pos_weight),
        "pos_weight": float(pos_weight.item()),
        "source_weighting": {
            "source_weight_target": args.source_weight_target,
            "source_weight": args.source_weight,
            "live_weight": args.live_weight,
            "spoof_weight": args.spoof_weight,
            "pos_weight_scale": args.pos_weight_scale,
        },
    }, out / "best_gru.pt")

    joblib.dump(scaler, out / "seq_scaler.joblib")

    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            **vars(args),
            "device_used": device,
            "selected_features": selected_features,
            "threshold": threshold,
            "parameter_count": count_params(model),
            "base_pos_weight": float(base_pos_weight),
            "final_pos_weight": float(final_pos_weight),
        }, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(out / "model_results.csv")
    print(out / "group_results.csv")
    print(out / "best_gru.pt")
    print(out / "seq_scaler.joblib")
    print(out / "run_config.json")
    print(f"\nselected threshold={threshold}")


if __name__ == "__main__":
    main()
