"""Shared model-store paths + artifact contract checks for context_emotion.

Every script that touches model-store/context_emotion/ (package, validate,
promote, rollback, smoke test) goes through this module instead of
hardcoding paths - mirrors how flashlight's scripts each redefine
_CANDIDATES_DIR/_CURRENT_DIR locally, except here it's centralized once
since this skeleton has more moving parts (5 scripts vs flashlight's 3).
"""
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import yaml

from context_emotion.common.constants import EMOTION_CLASSES

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTEXT_EMOTION_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_CONTEXT_EMOTION_ROOT, ".."))
_RUNTIME_CONTRACT_PATH = os.path.join(_CONTEXT_EMOTION_ROOT, "config", "runtime_contract.yaml")
_PROMOTION_POLICY_PATH = os.path.join(_CONTEXT_EMOTION_ROOT, "config", "promotion_policy.yaml")


def load_runtime_contract() -> dict:
    with open(_RUNTIME_CONTRACT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_promotion_policy() -> dict:
    with open(_PROMOTION_POLICY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_store_paths(store_root_override: Optional[str] = None, contract: Optional[dict] = None) -> Tuple[str, str, str, str]:
    """(store_root, candidates_dir, current_dir, archive_dir).

    store_root_override lets tests point the whole model-store at a tmp_path
    instead of the real ml-pipeline/model-store/context_emotion/ - every
    deployment/* function takes this same override so test_promote_rollback.py
    never touches the real store."""
    contract = contract or load_runtime_contract()
    store_cfg = contract["model_store"]
    store_root = store_root_override or os.path.join(_ML_PIPELINE_ROOT, store_cfg["root"])
    candidates_dir = os.path.join(store_root, store_cfg["candidates_dirname"])
    current_dir = os.path.join(store_root, store_cfg["current_dirname"])
    archive_dir = os.path.join(store_root, store_cfg["archive_dirname"])
    return store_root, candidates_dir, current_dir, archive_dir


STORE_ROOT, CANDIDATES_DIR, CURRENT_DIR, ARCHIVE_DIR = resolve_store_paths()
REQUIRED_FILES: List[str] = load_runtime_contract()["candidate_required_files"]


def candidate_dir(version: str, candidates_dir: str = CANDIDATES_DIR) -> str:
    return os.path.join(candidates_dir, version)


def is_current_dir(path: str, current_dir: str = CURRENT_DIR) -> bool:
    """Safety check used by package/validate scripts so nothing writes
    into current/ except promote_model.py's atomic swap."""
    return os.path.realpath(path) == os.path.realpath(current_dir)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_required_files(artifact_dir: str) -> List[str]:
    """Returns the list of missing required filenames (empty = all present)."""
    return [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(artifact_dir, f))]


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_model_metadata(metadata: dict) -> List[str]:
    """Minimal required-key check against contracts/model_metadata.schema.json.
    Returns a list of problem strings (empty = OK). No jsonschema dependency -
    this repo doesn't otherwise need it, so the check stays hand-rolled and
    intentionally shallow (existence + type of the few fields other scripts
    actually read)."""
    problems = []
    required = [
        "model_name", "version", "framework", "label_schema_version",
        "input_spec", "output_spec", "trained_at", "training_dataset_version",
        "checkpoint_source",
    ]
    for key in required:
        if key not in metadata:
            problems.append(f"metadata.json missing required key: {key}")

    for spec_key in ("input_spec", "output_spec"):
        spec = metadata.get(spec_key)
        if isinstance(spec, dict):
            for sub in ("name", "shape", "dtype"):
                if sub not in spec:
                    problems.append(f"metadata.json {spec_key} missing '{sub}'")
    return problems


def validate_label_schema(label_schema: dict, metadata: dict) -> List[str]:
    """Enforces contracts/label_schema_contract.md: emotion_classes must
    exactly match common/constants.py EMOTION_CLASSES in order, and
    label_schema_version must agree with metadata.json."""
    problems = []

    classes = label_schema.get("emotion_classes")
    if classes != EMOTION_CLASSES:
        problems.append(
            "label_schema.json emotion_classes does not match "
            f"context_emotion.common.constants.EMOTION_CLASSES exactly "
            f"(order-sensitive). got={classes!r}"
        )

    schema_version = label_schema.get("version")
    meta_version = metadata.get("label_schema_version")
    if schema_version != meta_version:
        problems.append(
            f"label_schema.json version ({schema_version!r}) != "
            f"metadata.json label_schema_version ({meta_version!r})"
        )

    output_spec = metadata.get("output_spec") or {}
    shape = output_spec.get("shape") or []
    if shape and isinstance(classes, list) and shape[-1] != len(classes):
        problems.append(
            f"metadata.json output_spec.shape last dim ({shape[-1] if shape else None}) "
            f"!= len(emotion_classes) ({len(classes) if isinstance(classes, list) else None})"
        )

    return problems


def validate_preprocessing_config(preprocessing_config: dict, contract: Optional[dict] = None) -> List[str]:
    contract = contract or load_runtime_contract()
    required_keys = contract["preprocessing_config_required_keys"]
    return [
        f"preprocessing_config.json missing required key: {k}"
        for k in required_keys
        if k not in preprocessing_config
    ]


def validate_version_consistency(expected_version: str, metadata: dict, evaluation_result: Optional[dict]) -> List[str]:
    """Nothing else checks that metadata.json's own 'version' field, an
    evaluation_result.json's own 'version' field, and the candidates/{X}/
    directory name (X) actually agree. Without this, a stale
    evaluation_result.json from a different model could get packaged
    alongside a brand-new onnx under a directory name that matches
    neither - compare_candidate.py would still "work" (it only ever reads
    the directory name) and nobody would notice."""
    problems = []
    meta_version = metadata.get("version")
    if meta_version != expected_version:
        problems.append(
            f"metadata.json version ({meta_version!r}) != candidate directory/--version ({expected_version!r})"
        )
    if evaluation_result is not None:
        eval_version = evaluation_result.get("version")
        if eval_version != expected_version:
            problems.append(
                f"evaluation_result.json version ({eval_version!r}) != candidate directory/--version ({expected_version!r})"
            )
    return problems


def validate_onnx_hash_consistency(onnx_path: str, evaluation_result: Optional[dict]) -> List[str]:
    """evaluate_candidate.py records the sha256 of the exact onnx file it
    ran inference against (evaluation_result.json's onnx_sha256). If the
    onnx being packaged/promoted doesn't match, evaluation_result.json
    describes a DIFFERENT model than the one about to go live - the
    numbers in it (and any compare_candidate.py gate decision based on
    them) would be meaningless. This is the main defense against
    "evaluation_result.json만 믿으면 안 되는" cases."""
    if evaluation_result is None:
        return ["cannot check onnx hash: evaluation_result.json missing/invalid"]

    recorded_hash = evaluation_result.get("onnx_sha256")
    if not recorded_hash:
        return ["evaluation_result.json has no onnx_sha256 - was it produced by an older evaluate_candidate.py?"]

    actual_hash = sha256_of(onnx_path)
    if actual_hash != recorded_hash:
        return [
            f"model.onnx sha256 ({actual_hash[:12]}...) != evaluation_result.json onnx_sha256 ({recorded_hash[:12]}...) "
            f"- this evaluation_result.json was NOT produced from this exact onnx file"
        ]
    return []


def validate_artifact_dir(artifact_dir: str, expected_version: Optional[str] = None) -> Dict[str, List[str]]:
    """Full contract validation of a candidate/current directory.
    Returns {category: [problem, ...]} - empty lists mean that category passed.
    Used by package_emotion_model.py, validate_model_artifacts.py,
    promote_model.py and the artifact_integrity gate in promotion_gate.py.

    Each category only depends on its own file (plus metadata.json for
    label_schema's cross-check) - so a modeling team can self-check their
    4 core files with validate_model_artifacts.py before evaluation_result.json
    /manifest.json even exist (see MLOPS_OPERATION_DESIGN.md section 9 step 1).
    A missing file fails that file's own category instead of being silently
    skipped as a false pass.

    expected_version is optional precisely so validate_model_artifacts.py's
    pre-packaging self-check (run before a version name is even decided)
    doesn't have to supply one - but package_emotion_model.py / promote_model.py
    / compare_candidate.py always pass it, since that's the only point where
    "does this onnx/metadata/evaluation_result actually agree on which
    version this is" can be caught."""
    missing = set(check_required_files(artifact_dir))
    result: Dict[str, List[str]] = {"required_files": sorted(missing)}

    metadata = None
    if "metadata.json" in missing:
        result["metadata"] = ["metadata.json missing - see required_files"]
    else:
        metadata = load_json(os.path.join(artifact_dir, "metadata.json"))
        result["metadata"] = validate_model_metadata(metadata)

    if "label_schema.json" in missing:
        result["label_schema"] = ["label_schema.json missing - see required_files"]
    elif metadata is None:
        result["label_schema"] = ["cannot cross-check label_schema.json: metadata.json missing/invalid"]
    else:
        label_schema = load_json(os.path.join(artifact_dir, "label_schema.json"))
        result["label_schema"] = validate_label_schema(label_schema, metadata)

    if "preprocessing_config.json" in missing:
        result["preprocessing_config"] = ["preprocessing_config.json missing - see required_files"]
    else:
        preprocessing_config = load_json(os.path.join(artifact_dir, "preprocessing_config.json"))
        result["preprocessing_config"] = validate_preprocessing_config(preprocessing_config)

    evaluation_result = None
    if "evaluation_result.json" not in missing:
        evaluation_result = load_json(os.path.join(artifact_dir, "evaluation_result.json"))

    if expected_version is not None:
        if metadata is None:
            result["version_consistency"] = ["cannot check version consistency: metadata.json missing/invalid"]
        else:
            result["version_consistency"] = validate_version_consistency(expected_version, metadata, evaluation_result)

        if "model.onnx" in missing:
            result["onnx_hash_consistency"] = ["cannot check onnx hash: model.onnx missing - see required_files"]
        else:
            result["onnx_hash_consistency"] = validate_onnx_hash_consistency(
                os.path.join(artifact_dir, "model.onnx"), evaluation_result
            )

    return result


def artifact_dir_is_valid(artifact_dir: str, expected_version: Optional[str] = None) -> bool:
    problems = validate_artifact_dir(artifact_dir, expected_version)
    return all(len(v) == 0 for v in problems.values())
