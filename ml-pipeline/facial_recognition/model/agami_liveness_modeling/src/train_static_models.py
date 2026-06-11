import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import load_npz, metadata_frame
from metrics import binary_metrics, choose_threshold, grouped_metrics


def get_optional_models(y_train):
    models = {}

    try:
        from xgboost import XGBClassifier
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=neg / pos,
            random_state=42,
            n_jobs=-1,
        )
    except Exception as e:
        print(f"[skip] xgboost unavailable: {e}")

    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    except Exception as e:
        print(f"[skip] lightgbm unavailable: {e}")

    return models


def prob_spoof_from_model(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        return (s - s.min()) / (s.max() - s.min() + 1e-8)
    raise ValueError("Model has neither predict_proba nor decision_function")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold-strategy", default="best_f1", choices=["best_f1", "eer_like", "low_far", "default"])
    p.add_argument("--max-frr", type=float, default=None)
    p.add_argument("--min-seq-len", type=int, default=1)
    p.add_argument("--min-face-rate", type=float, default=0.0)
    p.add_argument("--include-boosting", action="store_true", help="Also train XGBoost/LightGBM if installed.")
    args = p.parse_args()

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    data = load_npz(args.data)
    df = metadata_frame(data)

    keep = np.ones(len(df), dtype=bool)
    if "seq_lengths" in data:
        keep &= data["seq_lengths"].astype(int) >= args.min_seq_len
    if "face_detect_rates" in data:
        keep &= data["face_detect_rates"].astype(float) >= args.min_face_rate

    x = data["x_static"][keep].astype(np.float32)
    y = data["y"][keep].astype(int)
    df = df[keep].reset_index(drop=True)

    train_idx = np.where(df["split"].values == "train")[0]
    valid_idx = np.where(df["split"].values == "valid")[0]
    test_idx = np.where(df["split"].values == "test")[0]

    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    base_models = {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        "svm_rbf": SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    if args.include_boosting:
        base_models.update(get_optional_models(y_train))

    result_rows = []
    group_rows = []
    thresholds = {}

    for name, clf in base_models.items():
        print(f"\n[train] {name}")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clf),
        ])

        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_time = time.perf_counter() - start

        valid_prob = prob_spoof_from_model(model, x_valid)
        threshold = choose_threshold(y_valid, valid_prob, strategy=args.threshold_strategy, max_frr=args.max_frr)
        thresholds[name] = threshold

        for split_name, xx, yy, subdf in [
            ("valid", x_valid, y_valid, df.iloc[valid_idx]),
            ("test", x_test, y_test, df.iloc[test_idx]),
        ]:
            t0 = time.perf_counter()
            prob = prob_spoof_from_model(model, xx)
            infer_time = (time.perf_counter() - t0) / max(1, len(xx))

            m = binary_metrics(yy, prob, threshold)
            m.update({
                "model": name,
                "split": split_name,
                "train_time_sec": train_time,
                "inference_time_ms_per_sample": infer_time * 1000,
                "threshold_strategy": args.threshold_strategy,
            })
            result_rows.append(m)

            for col in ["attack_type", "source_group"]:
                gm = grouped_metrics(
                    yy, prob, subdf[col].values,
                    threshold=threshold,
                    group_name=col,
                    model_name=name,
                    split_name=split_name,
                )
                group_rows.append(gm)

        joblib.dump(model, out / "models" / f"{name}.joblib")

    pd.DataFrame(result_rows).to_csv(out / "model_results.csv", index=False)
    if group_rows:
        pd.concat(group_rows, ignore_index=True).to_csv(out / "group_results.csv", index=False)

    with open(out / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(out / "model_results.csv")
    print(out / "group_results.csv")


if __name__ == "__main__":
    main()
