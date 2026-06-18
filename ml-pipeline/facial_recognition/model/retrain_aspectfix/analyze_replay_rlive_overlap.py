"""
replay 공격과 R_live_clip의 spoof_score 분포가 왜 겹치는지 원인 분석.

배경: RETROSPECTIVE 9-5 — "R_live FRR<15%와 attack block>70~80%를 동시에
만족하는 weight×threshold 조합이 없다"는 결론의 병목이 "replay의 spoof_score
분포가 R_live(진짜 사람)와 너무 가깝다"는 것이었다. 이 스크립트는 그 주장을
1) spoof_score 분포 시각화, 2) 피처 단위 분리도(Cohen's d)로 직접 검증한다.

대상 모델: runs/gru_v3_aspectfix_w8 (가장 균형 잡힌 후보).
출력: analysis_replay_rlive/ 아래 png 2장 + csv 1장.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from facial_recognition.model.face_liveness_gru import FaceLivenessGRU
from facial_recognition.data.face_clip_dataset import load_npz, split_dataset
from facial_recognition.training.train_gru import predict

RUN_DIR = Path(__file__).parent / "runs" / "gru_v3_aspectfix_w8"
DATA_PATH = Path(__file__).parent / "face_clip_data_time_norm.npz"
OUT_DIR = Path(__file__).parent / "analysis_replay_rlive"
OUT_DIR.mkdir(exist_ok=True)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def main():
    ckpt = torch.load(RUN_DIR / "best_gru.pt", map_location="cpu")
    raw = load_npz(str(DATA_PATH))
    (ds_train, ds_valid, ds_test, scaler, selected_features, selected_idx,
     df, train_idx, valid_idx, test_idx) = split_dataset(raw, feature_mode="all", min_seq_len=1, min_face_rate=0.0)

    model = FaceLivenessGRU(
        input_dim=ckpt["input_dim"], hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"], dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])

    from torch.utils.data import DataLoader
    rows = []
    for split_name, ds, idx in [("valid", ds_valid, valid_idx), ("test", ds_test, test_idx)]:
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        yy, prob = predict(model, loader, "cpu")
        sub = df.iloc[idx].reset_index(drop=True)
        sub["spoof_score"] = prob
        sub["split"] = split_name
        rows.append(sub)
    scored = pd.concat(rows, ignore_index=True)
    scored.to_csv(OUT_DIR / "per_sample_scores.csv", index=False)

    # ── 1) spoof_score 분포: R_live(진짜) vs replay(공격) vs print(공격) vs S_dataset(진짜) ──
    groups = {
        "R_live_clip (real)": scored[(scored.source_group == "R_live_clip") & (scored.label == 0)].spoof_score,
        "S_dataset (real)":   scored[(scored.source_group == "S_dataset_sequence") & (scored.label == 0)].spoof_score,
        "replay (attack)":    scored[scored.attack_type == "replay"].spoof_score,
        "print (attack)":     scored[scored.attack_type == "print"].spoof_score,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    for name, vals in groups.items():
        ax.hist(vals, bins=bins, alpha=0.45, label=f"{name} (n={len(vals)})", density=True)
    ax.axvline(ckpt["threshold"], color="black", linestyle="--", label=f"model threshold={ckpt['threshold']:.2f}")
    ax.set_xlabel("spoof_score")
    ax.set_ylabel("density")
    ax.set_title("gru_v3_aspectfix_w8: spoof_score distribution (valid+test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spoof_score_distribution.png", dpi=150)
    plt.close(fig)

    d_rlive_replay = cohens_d(groups["R_live_clip (real)"].values, groups["replay (attack)"].values)
    d_sdataset_replay = cohens_d(groups["S_dataset (real)"].values, groups["replay (attack)"].values)
    d_rlive_print = cohens_d(groups["R_live_clip (real)"].values, groups["print (attack)"].values)
    print(f"Cohen's d  R_live vs replay   = {d_rlive_replay:.3f}")
    print(f"Cohen's d  S_dataset vs replay = {d_sdataset_replay:.3f}")
    print(f"Cohen's d  R_live vs print     = {d_rlive_print:.3f}")

    # ── 2) 피처 단위 분리도: 어떤 피처가 R_live와 replay를 가장 못 가르는가 ──
    x_seq_full = raw["x_seq"]
    seq_lengths = raw["seq_lengths"].astype(int)
    feature_names = [str(f) for f in raw["seq_feature_names"]]

    def clip_mean_features(global_idx: np.ndarray) -> pd.DataFrame:
        out = []
        for i in global_idx:
            l = max(1, int(seq_lengths[i]))
            out.append(x_seq_full[i, :l, :].mean(axis=0))
        return pd.DataFrame(out, columns=feature_names)

    r_live_idx = df[(df.source_group == "R_live_clip") & (df.label == 0)]["index"].values
    replay_idx = df[df.attack_type == "replay"]["index"].values
    sdataset_idx = df[(df.source_group == "S_dataset_sequence") & (df.label == 0)]["index"].values

    feat_r_live = clip_mean_features(r_live_idx)
    feat_replay = clip_mean_features(replay_idx)
    feat_sdataset = clip_mean_features(sdataset_idx)

    sep_rows = []
    for feat in feature_names:
        sep_rows.append({
            "feature": feat,
            "d_rlive_vs_replay": cohens_d(feat_r_live[feat].values, feat_replay[feat].values),
            "d_sdataset_vs_replay": cohens_d(feat_sdataset[feat].values, feat_replay[feat].values),
            "rlive_mean": feat_r_live[feat].mean(),
            "replay_mean": feat_replay[feat].mean(),
            "sdataset_mean": feat_sdataset[feat].mean(),
        })
    sep_df = pd.DataFrame(sep_rows).sort_values("d_rlive_vs_replay", key=lambda s: s.abs())
    sep_df.to_csv(OUT_DIR / "feature_separability_rlive_vs_replay.csv", index=False)

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    y_pos = np.arange(len(sep_df))
    ax2.barh(y_pos, sep_df["d_rlive_vs_replay"].abs(), color="tab:orange", alpha=0.8, label="|d| R_live vs replay")
    ax2.barh(y_pos, sep_df["d_sdataset_vs_replay"].abs(), color="tab:blue", alpha=0.5, label="|d| S_dataset vs replay")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sep_df["feature"])
    ax2.set_xlabel("|Cohen's d|  (real vs replay, per-clip mean feature)")
    ax2.set_title("Which features fail to separate R_live from replay\n(shorter orange bar = R_live looks more like replay)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "feature_separability.png", dpi=150)
    plt.close(fig2)

    print(f"\n저장 위치: {OUT_DIR}/")
    print("  spoof_score_distribution.png")
    print("  feature_separability.png")
    print("  feature_separability_rlive_vs_replay.csv")
    print("  per_sample_scores.csv")
    print("\n가장 분리 안 되는(|d| 작은) 피처 top5:")
    print(sep_df.head(5)[["feature", "d_rlive_vs_replay", "d_sdataset_vs_replay"]].to_string(index=False))
    print("\n가장 잘 분리되는(|d| 큰) 피처 top5:")
    print(sep_df.tail(5)[["feature", "d_rlive_vs_replay", "d_sdataset_vs_replay"]].to_string(index=False))


if __name__ == "__main__":
    main()
