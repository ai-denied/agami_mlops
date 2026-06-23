"""Shared mock-artifact builders for the ops-skeleton tests.

These build the smallest possible valid (or deliberately invalid) set of
the 6 candidate files so tests never need a real trained model - per the
task constraint, no test pretends to have real model performance, it just
exercises the package/compare/promote/rollback *mechanics*.
"""
import json
import os

from context_emotion.common.constants import EMOTION_CLASSES, SITUATION_CLASSES

VERSION = "v_test"


def write_valid_candidate_inputs(dir_path: str, version: str = VERSION) -> dict:
    """Writes the 5 *input* files (not manifest.json - package_emotion_model.py
    generates that) package_emotion_model.py expects via --onnx/--metadata/...
    Returns the dict of paths to pass to package()."""
    os.makedirs(dir_path, exist_ok=True)

    onnx_path = os.path.join(dir_path, "model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(b"not a real onnx file - tests use skip_validate=True / mock onnxruntime")

    metadata_path = os.path.join(dir_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "EmotionClassifier",
            "version": version,
            "framework": "pytorch",
            "label_schema_version": "v2",
            "input_spec": {"name": "image", "shape": ["batch", 3, 224, 224], "dtype": "float32"},
            "output_spec": {"name": "logits", "shape": ["batch", len(EMOTION_CLASSES)], "dtype": "float32"},
            "trained_at": "2026-07-01",
            "training_dataset_version": "context_emotion_train_dataset_v2",
            "checkpoint_source": "runs/mock/best.pt",
        }, f)

    label_schema_path = os.path.join(dir_path, "label_schema.json")
    with open(label_schema_path, "w", encoding="utf-8") as f:
        json.dump({"version": "v2", "emotion_classes": EMOTION_CLASSES, "situation_classes": SITUATION_CLASSES}, f)

    preprocessing_config_path = os.path.join(dir_path, "preprocessing_config.json")
    with open(preprocessing_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "image_size": [224, 224],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "crop_to_bbox_for_emotic": True,
        }, f)

    evaluation_result_path = os.path.join(dir_path, "evaluation_result.json")
    with open(evaluation_result_path, "w", encoding="utf-8") as f:
        json.dump(mock_evaluation_result(version), f)

    return {
        "onnx": onnx_path,
        "metadata": metadata_path,
        "label_schema": label_schema_path,
        "preprocessing_config": preprocessing_config_path,
        "evaluation_result": evaluation_result_path,
    }


def mock_evaluation_result(version: str, overall_macro_f1: float = 0.70, per_class_overrides: dict = None) -> dict:
    """A syntactically valid evaluation_result.json with made-up-but-labeled-
    as-mock numbers, used only to exercise compare/promotion_gate logic in
    tests - never written by production code without a real model behind it."""
    per_class = {
        cls: {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 50}
        for cls in EMOTION_CLASSES
    }
    if per_class_overrides:
        for cls, metrics in per_class_overrides.items():
            per_class[cls] = metrics

    return {
        "version": version,
        "evaluated_at": "2026-07-01T00:00:00",
        "eval_set": {"name": "mock:test", "size": 700, "source_path": "mock"},
        "overall": {"accuracy": overall_macro_f1, "macro_f1": overall_macro_f1, "weighted_f1": overall_macro_f1},
        "per_class": per_class,
        "human_ambiguity": {"status": "not_configured", "ambiguous_rate": None, "exposures": None},
        "attacker_proxy": {"status": "not_configured", "attacker_solve_rate": None, "error_type_breakdown": None, "proxy_model_version": None},
        "artifact_integrity": {"onnx_loadable": True, "input_output_match": True, "label_schema_match": True},
    }
