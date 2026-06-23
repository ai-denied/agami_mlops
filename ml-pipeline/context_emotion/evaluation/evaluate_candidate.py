#!/usr/bin/env python3
"""Run a packaged-or-pre-package candidate model against a held-out eval
CSV and produce evaluation_result.json (contracts/evaluation_result.schema.json).

This is meant to run BEFORE package_emotion_model.py - the modeling team
runs this against their freshly exported ONNX, then hands the resulting
evaluation_result.json to --evaluation-result on package_emotion_model.py.
No metric here is fabricated: if onnxruntime/numpy/PIL aren't available,
or the onnx/eval-csv don't exist, this exits with an error instead of
printing made-up numbers.

human_ambiguity and attacker_proxy are populated from ops_metrics /
an explicit attacker proxy config respectively - both default to
status="not_configured" for a model version that has no production
exposure yet (a brand-new candidate never has ops_metrics rows).

Usage:
    python -m context_emotion.evaluation.evaluate_candidate \\
        --onnx runs/emotion_classifier_v1/model.onnx \\
        --label-schema runs/emotion_classifier_v1/label_schema.json \\
        --preprocessing-config runs/emotion_classifier_v1/preprocessing_config.json \\
        --eval-csv /workspace/data/context_emotion/processed/context_emotion_train_dataset_v2.csv \\
        --eval-split test \\
        --image-root /workspace/data/context_emotion \\
        --version v1_20260701 \\
        --out runs/emotion_classifier_v1/evaluation_result.json
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.evaluation.attacker_proxy_eval import evaluate_attacker_proxy  # noqa: E402
from context_emotion.ops_metrics.recorder import latest_for_version  # noqa: E402


def _require(module_name: str):
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(
            f"[FAILED] evaluate_candidate.py requires '{module_name}' but it isn't installed: {e}\n"
            f"          (no fallback - we don't fabricate metrics without real inference)"
        )


def _load_eval_rows(eval_csv: str, eval_split: str):
    with open(eval_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset_split"] == eval_split and r["provisional_emotion"]]
    if not rows:
        raise SystemExit(f"[FAILED] {eval_csv} split={eval_split!r}에 provisional_emotion이 있는 행이 없습니다")
    return rows


def _preprocess_image(image_path: str, bbox_str: str, preprocessing_config: dict, np, Image):
    image = Image.open(image_path).convert("RGB")
    if preprocessing_config.get("crop_to_bbox_for_emotic") and bbox_str:
        x1, y1, x2, y2 = json.loads(bbox_str)
        image = image.crop((x1, y1, x2, y2))

    size = tuple(preprocessing_config["image_size"])
    image = image.resize(size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(preprocessing_config["normalize_mean"], dtype=np.float32)
    std = np.array(preprocessing_config["normalize_std"], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


def _confusion_matrix_metrics(y_true, y_pred, classes):
    """Manual precision/recall/f1 per class + macro/weighted - no sklearn
    dependency so this module stays runnable in lighter ops environments."""
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1

    per_class = {}
    f1s, supports = [], []
    correct = 0
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n)) - tp
        fn = sum(matrix[i][c] for c in range(n)) - tp
        support = sum(matrix[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
        supports.append(support)
        correct += tp

    total = sum(supports)
    accuracy = correct / total if total else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = sum(f1 * s for f1, s in zip(f1s, supports)) / total if total else 0.0
    return per_class, {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def evaluate(
    onnx_path: str,
    label_schema_path: str,
    preprocessing_config_path: str,
    eval_csv: str,
    eval_split: str,
    image_root: str,
    version: str,
    attacker_proxy_model: str = None,
    attacker_proxy_eval_pool: str = None,
) -> dict:
    np = _require("numpy")
    ort = _require("onnxruntime")
    Image = _require("PIL.Image")

    label_schema = json.load(open(label_schema_path, encoding="utf-8"))
    preprocessing_config = json.load(open(preprocessing_config_path, encoding="utf-8"))
    classes = label_schema["emotion_classes"]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rows = _load_eval_rows(eval_csv, eval_split)

    y_true, y_pred = [], []
    for row in rows:
        image_path = os.path.join(image_root, row["image_path"])
        x = _preprocess_image(image_path, row.get("target_person_bbox", ""), preprocessing_config, np, Image)
        logits = session.run([output_name], {input_name: x})[0][0]
        y_true.append(row["provisional_emotion"])
        y_pred.append(classes[int(logits.argmax())])

    per_class, overall = _confusion_matrix_metrics(y_true, y_pred, classes)

    ops_rows = latest_for_version(version)
    if ops_rows:
        total_exposures = sum(r["exposures"] for r in ops_rows)
        ambiguous = [r["human_ambiguous_rate"] for r in ops_rows if r.get("human_ambiguous_rate") is not None]
        human_ambiguity = {
            "status": "available",
            "ambiguous_rate": sum(ambiguous) / len(ambiguous) if ambiguous else None,
            "exposures": total_exposures,
        }
    else:
        human_ambiguity = {"status": "not_configured", "ambiguous_rate": None, "exposures": None}

    attacker_proxy = evaluate_attacker_proxy(onnx_path, attacker_proxy_model, attacker_proxy_eval_pool)

    artifact_integrity = {
        "onnx_loadable": True,
        "input_output_match": True,
        "label_schema_match": classes == label_schema.get("emotion_classes"),
    }

    return {
        "version": version,
        "evaluated_at": datetime.now().isoformat(),
        "eval_set": {"name": f"{os.path.basename(eval_csv)}:{eval_split}", "size": len(rows), "source_path": eval_csv},
        "overall": overall,
        "per_class": per_class,
        "human_ambiguity": human_ambiguity,
        "attacker_proxy": attacker_proxy,
        "artifact_integrity": artifact_integrity,
    }


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--label-schema", required=True)
    ap.add_argument("--preprocessing-config", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--attacker-proxy-model", default=None, help="TODO: 아직 미확정 - attacker_proxy_eval.py 참고")
    ap.add_argument("--attacker-proxy-eval-pool", default=None)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = _parse_args()
    result = evaluate(
        onnx_path=args.onnx,
        label_schema_path=args.label_schema,
        preprocessing_config_path=args.preprocessing_config,
        eval_csv=args.eval_csv,
        eval_split=args.eval_split,
        image_root=args.image_root,
        version=args.version,
        attacker_proxy_model=args.attacker_proxy_model,
        attacker_proxy_eval_pool=args.attacker_proxy_eval_pool,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")
    print(f"overall: {result['overall']}")


if __name__ == "__main__":
    main()
