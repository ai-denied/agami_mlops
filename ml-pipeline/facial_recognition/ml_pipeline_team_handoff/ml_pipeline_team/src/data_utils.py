import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def to_str_array(arr) -> np.ndarray:
    return np.asarray(arr).astype(str)


def infer_source_group(sample_id: str) -> str:
    s = str(sample_id)
    if s.startswith("ATK"):
        return "ATK_external_clip"
    if s.startswith("R"):
        return "R_live_clip"
    if s.startswith("S"):
        return "S_dataset_sequence"
    return "unknown"


def infer_root_id(sample_id: str) -> str:
    s = str(sample_id)

    m = re.match(r"^(ATK\d+)_(print|replay|mask|spoof|live)", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    if s.startswith("S"):
        return s.split("_")[0]

    if "_clip" in s:
        return re.sub(r"_clip\d+.*$", "", s)

    return s.split("_")[0]


def split_indices(splits: np.ndarray) -> Dict[str, np.ndarray]:
    splits = to_str_array(splits)
    return {
        "train": np.where(splits == "train")[0],
        "valid": np.where((splits == "valid") | (splits == "val") | (splits == "validation"))[0],
        "test": np.where(splits == "test")[0],
    }


def select_seq_features(feature_names: List[str], mode: str) -> List[int]:
    names = [str(x) for x in feature_names]
    mode = mode.lower()

    groups = {
        "abs": {"nose_x", "nose_y", "cx", "cy"},
        "motion": {
            "nose_dx", "nose_dy", "center_dx", "center_dy", "nose_speed",
            "ear_velocity", "mar_velocity", "yaw_velocity", "pitch_velocity", "roll_velocity"
        },
        "eye_mouth": {"ear", "mar", "smile_w", "ear_velocity", "mar_velocity"},
        "head": {"roll", "yaw", "pitch", "yaw_velocity", "pitch_velocity", "roll_velocity"},
    }

    if mode == "all":
        selected = set(names)
    elif mode == "no_abs":
        selected = set(names) - groups["abs"]
    elif mode == "motion_only":
        selected = groups["motion"]
    elif mode == "eye_mouth":
        selected = groups["eye_mouth"]
    elif mode == "head":
        selected = groups["head"]
    elif mode == "no_abs_motion_head":
        selected = set(names) - groups["abs"]
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    idx = [i for i, n in enumerate(names) if n in selected]
    if not idx:
        raise ValueError(f"No features selected for mode={mode}. Available={names}")
    return idx


def fit_seq_scaler(
    x_seq: np.ndarray,
    lengths: np.ndarray,
    train_idx: np.ndarray,
) -> StandardScaler:
    frames = []
    for i in train_idx:
        l = int(lengths[i])
        if l > 0:
            frames.append(x_seq[i, :l, :])
    valid_frames = np.concatenate(frames, axis=0)
    scaler = StandardScaler()
    scaler.fit(valid_frames)
    return scaler


def transform_seq_with_lengths(
    x_seq: np.ndarray,
    lengths: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    out = np.zeros_like(x_seq, dtype=np.float32)
    for i in range(len(x_seq)):
        l = int(lengths[i])
        if l > 0:
            out[i, :l, :] = scaler.transform(x_seq[i, :l, :]).astype(np.float32)
    return out


def metadata_frame(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(data["y"])
    sample_ids = to_str_array(data.get("sample_ids", np.arange(n).astype(str)))
    df = pd.DataFrame({
        "index": np.arange(n),
        "sample_id": sample_ids,
        "label": data["y"].astype(int),
        "split": to_str_array(data.get("splits", np.array([""] * n))),
        "attack_type": to_str_array(data.get("attack_types", np.array([""] * n))),
        "subject_id": to_str_array(data.get("subject_ids", np.array([""] * n))),
    })
    df["source_group"] = df["sample_id"].map(infer_source_group)
    df["root_id"] = df["sample_id"].map(infer_root_id)

    for key in ["seq_lengths", "valid_frame_counts", "face_detect_rates", "sessions", "illuminations", "devices"]:
        if key in data:
            df[key] = data[key]
    return df
