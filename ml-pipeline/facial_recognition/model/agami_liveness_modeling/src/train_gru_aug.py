import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from data_utils import (
    load_npz,
    metadata_frame,
    select_seq_features,
    fit_seq_scaler,
    transform_seq_with_lengths,
)
from metrics import binary_metrics, choose_threshold, grouped_metrics, safe_auc


class AugSeqDataset(Dataset):
    """
    GRU input dataset.
    label: 0=live, 1=spoof

    Augmentation is applied only when augment=True.
    Important:
    - Only valid frames within seq_length are augmented.
    - Padding frames remain zero.
    - Validation/test should use augment=False.
    """

    def __init__(
        self,
        x,
        y,
        lengths,
        augment=False,
        aug_prob=1.0,
        noise_std=0.0,
        time_mask_prob=0.0,
        time_mask_mode="zero",
        feature_drop_prob=0.0,
        feature_drop_indices=None,
    ):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.lengths = np.asarray(lengths, dtype=np.int64)

        self.augment = bool(augment)
        self.aug_prob = float(aug_prob)
        self.noise_std = float(noise_std)
        self.time_mask_prob = float(time_mask_prob)
        self.time_mask_mode = str(time_mask_mode)
        self.feature_drop_prob = float(feature_drop_prob)
        self.feature_drop_indices = feature_drop_indices or []

    def __len__(self):
        return len(self.y)

    def _augment_one(self, x, length):
        x = x.copy()
        length = int(length)

        if length <= 0:
            return x

        valid = x[:length, :]

        # Apply augmentation with sample-level probability.
        if np.random.rand() > self.aug_prob:
            return x

        # 1) Gaussian noise in standardized feature space.
        # Since x is already standardized, 0.01~0.02 is weak noise.
        if self.noise_std > 0:
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_std,
                size=valid.shape,
            ).astype(np.float32)
            valid += noise

        # 2) Time masking: mask a small number of valid frames.
        # zero in standardized space means feature mean.
        if self.time_mask_prob > 0:
            mask_flags = np.random.rand(length) < self.time_mask_prob
            mask_idx = np.where(mask_flags)[0]

            if len(mask_idx) > 0:
                if self.time_mask_mode == "zero":
                    valid[mask_idx, :] = 0.0
                elif self.time_mask_mode == "mean":
                    mean_frame = valid.mean(axis=0, keepdims=True)
                    valid[mask_idx, :] = mean_frame
                else:
                    raise ValueError(f"Unknown time_mask_mode: {self.time_mask_mode}")

        # 3) Feature dropout: mainly for absolute-position features.
        # Dropping to zero means using train-set mean after standardization.
        if self.feature_drop_prob > 0 and len(self.feature_drop_indices) > 0:
            for j in self.feature_drop_indices:
                if np.random.rand() < self.feature_drop_prob:
                    valid[:, j] = 0.0

        x[:length, :] = valid
        return x

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        length = self.lengths[idx]

        if self.augment:
            x = self._augment_one(x, length)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(length, dtype=torch.long),
        )


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()

        packed = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        _, h_n = self.gru(packed)
        h_last = h_n[-1]
        logits = self.head(h_last).squeeze(-1)

        return logits


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    probs = []
    ys = []

    for x, y, lengths in loader:
        x = x.to(device)
        lengths = lengths.to(device)

        logits = model(x, lengths)
        prob = torch.sigmoid(logits).detach().cpu().numpy()

        probs.append(prob)
        ys.append(y.numpy())

    return np.concatenate(ys), np.concatenate(probs)


def run_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    losses = []

    for x, y, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        logits = model(x, lengths)
        loss = loss_fn(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_feature_drop_indices(selected_features, target):
    target = str(target).lower()

    if target == "none":
        return []

    if target == "abs":
        abs_names = {"nose_x", "nose_y", "cx", "cy"}
        return [i for i, name in enumerate(selected_features) if name in abs_names]

    if target == "all":
        return list(range(len(selected_features)))

    raise ValueError(f"Unknown feature_drop_target: {target}")


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

    # Augmentation options
    p.add_argument("--augment", action="store_true")
    p.add_argument("--aug-prob", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--time-mask-prob", type=float, default=0.0)
    p.add_argument("--time-mask-mode", default="zero", choices=["zero", "mean"])
    p.add_argument("--feature-drop-prob", type=float, default=0.0)
    p.add_argument("--feature-drop-target", default="abs", choices=["none", "abs", "all"])

    # Optional loss adjustment.
    # 1.0 = existing behavior.
    # Lower than 1.0 makes model slightly less aggressive toward spoof class.
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

    keep = np.ones(len(df), dtype=bool)
    keep &= data["seq_lengths"].astype(int) >= args.min_seq_len

    if "face_detect_rates" in data:
        keep &= data["face_detect_rates"].astype(float) >= args.min_face_rate

    x_seq = data["x_seq"][keep].astype(np.float32)
    y = data["y"][keep].astype(int)
    lengths = data["seq_lengths"][keep].astype(int)
    df = df[keep].reset_index(drop=True)

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

    feature_drop_indices = get_feature_drop_indices(
        selected_features,
        args.feature_drop_target,
    )

    train_loader = DataLoader(
        AugSeqDataset(
            x_train,
            y_train,
            len_train,
            augment=args.augment,
            aug_prob=args.aug_prob,
            noise_std=args.noise_std,
            time_mask_prob=args.time_mask_prob,
            time_mask_mode=args.time_mask_mode,
            feature_drop_prob=args.feature_drop_prob,
            feature_drop_indices=feature_drop_indices,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        AugSeqDataset(x_valid, y_valid, len_valid, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        AugSeqDataset(x_test, y_test, len_test, augment=False),
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

    history = []
    best_score = -1.0
    best_state = None
    bad = 0

    print(f"device={device}")
    print(f"feature_mode={args.feature_mode}")
    print(f"selected_features={selected_features}")
    print(f"train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")
    print(f"params={count_params(model)}")
    print(f"base_pos_weight={base_pos_weight:.4f}, final_pos_weight={final_pos_weight:.4f}")
    print(
        "augmentation="
        f"{args.augment}, aug_prob={args.aug_prob}, noise_std={args.noise_std}, "
        f"time_mask_prob={args.time_mask_prob}, time_mask_mode={args.time_mask_mode}, "
        f"feature_drop_prob={args.feature_drop_prob}, "
        f"feature_drop_target={args.feature_drop_target}, "
        f"feature_drop_indices={feature_drop_indices}"
    )

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
            "model": f"gru_aug_{args.feature_mode}",
            "split": split_name,
            "inference_time_ms_per_sample": infer_time * 1000,
            "parameter_count": count_params(model),
            "feature_mode": args.feature_mode,
            "selected_feature_count": len(selected_features),
            "min_seq_len": args.min_seq_len,
            "min_face_rate": args.min_face_rate,
            "threshold_strategy": args.threshold_strategy,
            "augment": args.augment,
            "aug_prob": args.aug_prob,
            "noise_std": args.noise_std,
            "time_mask_prob": args.time_mask_prob,
            "time_mask_mode": args.time_mask_mode,
            "feature_drop_prob": args.feature_drop_prob,
            "feature_drop_target": args.feature_drop_target,
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
                model_name=f"gru_aug_{args.feature_mode}",
                split_name=split_name,
            )
            gm["augment"] = args.augment
            gm["noise_std"] = args.noise_std
            gm["time_mask_prob"] = args.time_mask_prob
            gm["feature_drop_prob"] = args.feature_drop_prob
            gm["feature_drop_target"] = args.feature_drop_target
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
        "augmentation": {
            "augment": args.augment,
            "aug_prob": args.aug_prob,
            "noise_std": args.noise_std,
            "time_mask_prob": args.time_mask_prob,
            "time_mask_mode": args.time_mask_mode,
            "feature_drop_prob": args.feature_drop_prob,
            "feature_drop_target": args.feature_drop_target,
            "feature_drop_indices": feature_drop_indices,
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
            "feature_drop_indices": feature_drop_indices,
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
