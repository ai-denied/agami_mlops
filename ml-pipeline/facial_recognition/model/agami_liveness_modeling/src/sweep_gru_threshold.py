import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_utils import load_npz, metadata_frame, transform_seq_with_lengths
from metrics import binary_metrics, threshold_sweep, grouped_metrics
from train_gru import GRUClassifier, SeqDataset, predict


def pick_thresholds(valid_sweep: pd.DataFrame):
    picks = []

    def add_pick(name, row):
        if row is None:
            return
        picks.append({
            "rule": name,
            "threshold": float(row["threshold"]),
            "valid_accuracy": float(row["accuracy"]),
            "valid_f1_spoof": float(row["f1_spoof"]),
            "valid_far_attack_pass_rate": float(row["far_attack_pass_rate"]),
            "valid_frr_genuine_reject_rate": float(row["frr_genuine_reject_rate"]),
            "valid_attack_block_rate": float(row["attack_block_rate"]),
            "valid_genuine_pass_rate": float(row["genuine_pass_rate"]),
        })

    # 1. F1 최고
    add_pick("best_f1", valid_sweep.loc[valid_sweep["f1_spoof"].idxmax()])

    # 2. FAR와 FRR이 가장 비슷한 지점
    eer_idx = (valid_sweep["far_attack_pass_rate"] - valid_sweep["frr_genuine_reject_rate"]).abs().idxmin()
    add_pick("eer_like", valid_sweep.loc[eer_idx])

    # 3. 정상 거절률 30% 이하에서 FAR 최소
    cand = valid_sweep[valid_sweep["frr_genuine_reject_rate"] <= 0.30]
    if len(cand) > 0:
        cand = cand.sort_values(
            ["far_attack_pass_rate", "f1_spoof", "accuracy"],
            ascending=[True, False, False]
        )
        add_pick("low_far_with_frr_under_30", cand.iloc[0])

    # 4. 정상 통과율 80% 이상에서 FAR 최소
    cand = valid_sweep[valid_sweep["genuine_pass_rate"] >= 0.80]
    if len(cand) > 0:
        cand = cand.sort_values(
            ["far_attack_pass_rate", "f1_spoof", "accuracy"],
            ascending=[True, False, False]
        )
        add_pick("genuine_pass_80", cand.iloc[0])

    # 5. 정상 통과율 90% 이상에서 FAR 최소
    cand = valid_sweep[valid_sweep["genuine_pass_rate"] >= 0.90]
    if len(cand) > 0:
        cand = cand.sort_values(
            ["far_attack_pass_rate", "f1_spoof", "accuracy"],
            ascending=[True, False, False]
        )
        add_pick("genuine_pass_90", cand.iloc[0])

    # 6. 기본 threshold 0.50
    default_row = valid_sweep.iloc[(valid_sweep["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]
    add_pick("default_0_50", default_row)

    return pd.DataFrame(picks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "threshold_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ckpt_path = run_dir / "best_gru.pt"
    scaler_path = run_dir / "seq_scaler.joblib"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    scaler = joblib.load(scaler_path)

    run_args = ckpt.get("args", {})
    min_seq_len = int(run_args.get("min_seq_len", 1))
    min_face_rate = float(run_args.get("min_face_rate", 0.0))

    data = load_npz(args.data)
    df = metadata_frame(data)

    keep = np.ones(len(df), dtype=bool)
    keep &= data["seq_lengths"].astype(int) >= min_seq_len
    if "face_detect_rates" in data:
        keep &= data["face_detect_rates"].astype(float) >= min_face_rate

    x_seq = data["x_seq"][keep].astype(np.float32)
    y = data["y"][keep].astype(int)
    lengths = data["seq_lengths"][keep].astype(int)
    df = df[keep].reset_index(drop=True)

    selected_idx = [int(i) for i in ckpt["selected_idx"]]
    selected_features = [str(x) for x in ckpt["selected_features"]]
    x_seq = x_seq[:, :, selected_idx]
    x_seq = transform_seq_with_lengths(x_seq, lengths, scaler)

    valid_idx = np.where(df["split"].values == "valid")[0]
    test_idx = np.where(df["split"].values == "test")[0]

    model = GRUClassifier(
        input_dim=int(ckpt["input_dim"]),
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    split_outputs = {}

    for split_name, idx in [("valid", valid_idx), ("test", test_idx)]:
        loader = DataLoader(
            SeqDataset(x_seq[idx], y[idx], lengths[idx]),
            batch_size=args.batch_size,
            shuffle=False
        )

        yy, prob = predict(model, loader, device)

        pred_df = df.iloc[idx].copy().reset_index(drop=True)
        pred_df["y_true"] = yy
        pred_df["prob_spoof"] = prob
        pred_df.to_csv(out_dir / f"predictions_{split_name}.csv", index=False)

        sweep = threshold_sweep(
            yy,
            prob,
            thresholds=np.round(np.arange(0.01, 1.00, 0.01), 2)
        )
        sweep["split"] = split_name
        sweep["model"] = run_dir.name
        sweep["feature_mode"] = run_args.get("feature_mode", "")
        sweep["hidden_size"] = run_args.get("hidden_size", "")
        sweep["dropout"] = run_args.get("dropout", "")
        sweep["lr"] = run_args.get("lr", "")
        sweep["weight_decay"] = run_args.get("weight_decay", "")
        sweep["selected_feature_count"] = len(selected_features)
        sweep.to_csv(out_dir / f"threshold_sweep_{split_name}.csv", index=False)

        split_outputs[split_name] = {
            "y": yy,
            "prob": prob,
            "df": pred_df,
            "sweep": sweep,
        }

    valid_sweep = split_outputs["valid"]["sweep"]
    picked = pick_thresholds(valid_sweep)
    picked.to_csv(out_dir / "picked_thresholds_from_valid.csv", index=False)

    eval_rows = []
    group_rows = []

    for _, pick in picked.iterrows():
        rule = pick["rule"]
        th = float(pick["threshold"])

        for split_name in ["valid", "test"]:
            yy = split_outputs[split_name]["y"]
            prob = split_outputs[split_name]["prob"]
            subdf = split_outputs[split_name]["df"]

            m = binary_metrics(yy, prob, threshold=th)
            m.update({
                "rule": rule,
                "model": run_dir.name,
                "split": split_name,
            })
            eval_rows.append(m)

            for group_col in ["attack_type", "source_group"]:
                gm = grouped_metrics(
                    yy,
                    prob,
                    subdf[group_col].values,
                    threshold=th,
                    group_name=group_col,
                    model_name=run_dir.name,
                    split_name=split_name,
                )
                gm["rule"] = rule
                group_rows.append(gm)

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_dir / "picked_thresholds_eval_valid_test.csv", index=False)

    if group_rows:
        pd.concat(group_rows, ignore_index=True).to_csv(
            out_dir / "picked_thresholds_group_results.csv",
            index=False
        )

    print(f"\nSaved threshold sweep results to: {out_dir}")
    print("\n[Picked thresholds from validation]")
    print(picked.to_string(index=False))

    print("\n[Picked thresholds evaluated on valid/test]")
    show_cols = [
        "rule", "split", "threshold", "accuracy", "f1_spoof",
        "far_attack_pass_rate", "frr_genuine_reject_rate",
        "attack_block_rate", "genuine_pass_rate",
        "tn", "fp", "fn", "tp"
    ]
    print(eval_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
