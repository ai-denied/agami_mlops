import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_npz, metadata_frame


def as_py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def feature_stats_from_seq(x_seq, lengths, names):
    rows = []
    for j, name in enumerate(names):
        vals = []
        for i in range(len(x_seq)):
            l = int(lengths[i])
            if l > 0:
                vals.append(x_seq[i, :l, j])
        vals = np.concatenate(vals) if vals else np.array([])
        if len(vals) == 0:
            row = {"feature": str(name), "count": 0}
        else:
            row = {
                "feature": str(name),
                "count": int(len(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "zero_ratio": float(np.mean(vals == 0)),
                "p01": float(np.percentile(vals, 1)),
                "p99": float(np.percentile(vals, 99)),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def feature_stats_static(x, names):
    rows = []
    for j, name in enumerate(names):
        vals = x[:, j]
        rows.append({
            "feature": str(name),
            "count": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "zero_ratio": float(np.mean(vals == 0)),
            "p01": float(np.percentile(vals, 1)),
            "p99": float(np.percentile(vals, 99)),
        })
    return pd.DataFrame(rows)


def split_label_table(df):
    return pd.crosstab(df["split"], df["label"]).rename(columns={0: "live_0", 1: "spoof_1"})


def split_attack_table(df):
    return pd.crosstab(df["split"], df["attack_type"])


def group_label_table(df, col):
    return pd.crosstab(df[col], df["label"]).rename(columns={0: "live_0", 1: "spoof_1"})


def leakage_report(df, group_col):
    rows = []
    for gid, sub in df.groupby(group_col):
        splits = sorted(sub["split"].unique().tolist())
        if len(splits) > 1:
            rows.append({
                group_col: gid,
                "splits": ",".join(splits),
                "n": int(len(sub)),
                "live_0": int((sub["label"] == 0).sum()),
                "spoof_1": int((sub["label"] == 1).sum()),
            })
    return pd.DataFrame(rows).sort_values("n", ascending=False) if rows else pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.data)
    df = metadata_frame(data)

    report = {
        "data_path": args.data,
        "keys": list(data.keys()),
        "shapes": {k: list(v.shape) for k, v in data.items() if hasattr(v, "shape")},
        "dtypes": {k: str(v.dtype) for k, v in data.items() if hasattr(v, "dtype")},
        "n_samples": int(len(data["y"])),
        "nan_count_x_seq": int(np.isnan(data["x_seq"]).sum()) if "x_seq" in data else None,
        "inf_count_x_seq": int(np.isinf(data["x_seq"]).sum()) if "x_seq" in data else None,
        "nan_count_x_static": int(np.isnan(data["x_static"]).sum()) if "x_static" in data else None,
        "inf_count_x_static": int(np.isinf(data["x_static"]).sum()) if "x_static" in data else None,
        "label_counts": {str(k): int(v) for k, v in pd.Series(data["y"]).value_counts().sort_index().items()},
        "sample_id_duplicate_count": int(df["sample_id"].duplicated().sum()),
    }

    if "seq_lengths" in data:
        seq = data["seq_lengths"].astype(int)
        report["seq_length"] = {
            "min": int(seq.min()), "max": int(seq.max()), "mean": float(seq.mean()),
            "median": float(np.median(seq)), "lt_5": int((seq < 5).sum()),
            "lt_8": int((seq < 8).sum()), "full_16": int((seq >= 16).sum())
        }

        x_seq = data["x_seq"]
        bad_padding = 0
        zero_valid_frames = 0
        for i, l in enumerate(seq):
            if l < x_seq.shape[1] and not np.allclose(x_seq[i, l:, :], 0):
                bad_padding += 1
            if l > 0:
                frame_zero = np.all(np.isclose(x_seq[i, :l, :], 0), axis=1)
                zero_valid_frames += int(frame_zero.sum())
        report["padding"] = {
            "bad_padding_samples": int(bad_padding),
            "zero_valid_frame_count": int(zero_valid_frames),
        }

    if "face_detect_rates" in data:
        f = data["face_detect_rates"].astype(float)
        report["face_detect_rate"] = {
            "min": float(f.min()), "max": float(f.max()), "mean": float(f.mean()),
            "median": float(np.median(f)), "lt_0_8": int((f < 0.8).sum()),
            "eq_1_0": int(np.isclose(f, 1.0).sum()),
        }
        low = df[f < 0.8].copy()
        low.to_csv(out / "low_face_detect_samples.csv", index=False)

    split_label_table(df).to_csv(out / "split_label_counts.csv")
    split_attack_table(df).to_csv(out / "split_attack_counts.csv")
    group_label_table(df, "source_group").to_csv(out / "source_group_label_counts.csv")
    group_label_table(df, "root_id").to_csv(out / "root_id_label_counts.csv")

    subject_leak = leakage_report(df[df["subject_id"].astype(str) != "None"], "subject_id")
    root_leak = leakage_report(df, "root_id")
    subject_leak.to_csv(out / "subject_split_overlap.csv", index=False)
    root_leak.to_csv(out / "root_split_overlap.csv", index=False)

    if "seq_feature_names" in data:
        seq_stats = feature_stats_from_seq(
            data["x_seq"],
            data.get("seq_lengths", np.full(len(data["y"]), data["x_seq"].shape[1])),
            data["seq_feature_names"],
        )
        seq_stats.to_csv(out / "seq_feature_stats.csv", index=False)

    if "agg_feature_names" in data:
        st_stats = feature_stats_static(data["x_static"], data["agg_feature_names"])
        st_stats.to_csv(out / "static_feature_stats.csv", index=False)

    df.to_csv(out / "metadata.csv", index=False)

    with open(out / "dataset_check_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=as_py)

    md = []
    md.append("# Dataset Check Report\n")
    md.append(f"- Data path: `{args.data}`")
    md.append(f"- Samples: **{report['n_samples']}**")
    md.append(f"- x_seq shape: `{report['shapes'].get('x_seq')}`")
    md.append(f"- x_static shape: `{report['shapes'].get('x_static')}`")
    md.append(f"- y shape: `{report['shapes'].get('y')}`")
    md.append(f"- NaN x_seq: `{report.get('nan_count_x_seq')}` / Inf x_seq: `{report.get('inf_count_x_seq')}`")
    md.append(f"- NaN x_static: `{report.get('nan_count_x_static')}` / Inf x_static: `{report.get('inf_count_x_static')}`")
    md.append(f"- Label counts: `{report['label_counts']}`")
    if "seq_length" in report:
        md.append(f"- Seq length stats: `{report['seq_length']}`")
    if "face_detect_rate" in report:
        md.append(f"- Face detect rate stats: `{report['face_detect_rate']}`")
    md.append(f"- Sample ID duplicates: `{report['sample_id_duplicate_count']}`")
    md.append(f"- Subject split overlap rows: `{len(subject_leak)}`")
    md.append(f"- Root split overlap rows: `{len(root_leak)}`")
    md.append("\n생성된 CSV 파일에서 split/source/root별 분포와 feature 통계를 확인하세요.\n")

    (out / "dataset_check_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
