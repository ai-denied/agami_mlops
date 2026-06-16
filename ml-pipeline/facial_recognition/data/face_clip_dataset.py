"""
NPZ 기반 GRU 학습 데이터셋 및 전처리 유틸리티.

NPZ 포맷 (face_clip_data.npz):
  x_seq            : (N, max_seq_len, n_features) float32
  y                : (N,) int   — 0=live, 1=spoof
  seq_lengths      : (N,) int   — 실제 유효 프레임 수
  splits           : (N,) str   — "train" | "valid" | "test"
  seq_feature_names: (n_features,) str
  sample_ids       : (N,) str   — 선택적
  attack_types     : (N,) str   — 선택적
  subject_ids      : (N,) str   — 선택적
  face_detect_rates: (N,) float — 선택적
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def to_str_array(arr) -> np.ndarray:
    return np.asarray(arr).astype(str)


# ── 소스 그룹 추론 ────────────────────────────────────────────────────────────

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


# ── 피처 선택 ─────────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "abs":        {"nose_x", "nose_y", "cx", "cy"},
    "motion":     {"nose_dx", "nose_dy", "center_dx", "center_dy", "nose_speed",
                   "ear_velocity", "mar_velocity", "yaw_velocity", "pitch_velocity", "roll_velocity"},
    "eye_mouth":  {"ear", "mar", "smile_w", "ear_velocity", "mar_velocity"},
    "head":       {"roll", "yaw", "pitch", "yaw_velocity", "pitch_velocity", "roll_velocity"},
}


def select_seq_features(feature_names: List[str], mode: str) -> List[int]:
    names = [str(x) for x in feature_names]
    mode  = mode.lower()

    if mode == "all":
        selected = set(names)
    elif mode == "no_abs":
        selected = set(names) - FEATURE_GROUPS["abs"]
    elif mode == "motion_only":
        selected = FEATURE_GROUPS["motion"]
    elif mode == "eye_mouth":
        selected = FEATURE_GROUPS["eye_mouth"]
    elif mode == "head":
        selected = FEATURE_GROUPS["head"]
    elif mode == "no_abs_motion_head":
        selected = set(names) - FEATURE_GROUPS["abs"]
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    idx = [i for i, n in enumerate(names) if n in selected]
    if not idx:
        raise ValueError(f"No features selected for mode={mode}. Available={names}")
    return idx


# ── 스케일러 ──────────────────────────────────────────────────────────────────

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


# ── 메타데이터 프레임 ─────────────────────────────────────────────────────────

def metadata_frame(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    n          = len(data["y"])
    sample_ids = to_str_array(data.get("sample_ids", np.arange(n).astype(str)))
    df = pd.DataFrame({
        "index":       np.arange(n),
        "sample_id":   sample_ids,
        "label":       data["y"].astype(int),
        "split":       to_str_array(data.get("splits",       np.array([""] * n))),
        "attack_type": to_str_array(data.get("attack_types", np.array([""] * n))),
        "subject_id":  to_str_array(data.get("subject_ids",  np.array([""] * n))),
    })
    df["source_group"] = df["sample_id"].map(infer_source_group)
    df["root_id"]      = df["sample_id"].map(infer_root_id)

    for key in ["seq_lengths", "valid_frame_counts", "face_detect_rates",
                "sessions", "illuminations", "devices"]:
        if key in data:
            df[key] = data[key]
    return df


# ── PyTorch 데이터셋 ──────────────────────────────────────────────────────────

class FaceClipDataset(Dataset):
    """NPZ에서 로드한 GRU 시퀀스 데이터셋."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        self.x       = torch.tensor(x,       dtype=torch.float32)
        self.y       = torch.tensor(y,       dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.lengths[idx]


def split_dataset(
    data: Dict[str, np.ndarray],
    feature_mode: str = "all",
    min_seq_len: int = 1,
    min_face_rate: float = 0.0,
) -> tuple:
    """NPZ 데이터를 로드하여 train/valid/test 데이터셋과 메타 정보를 반환한다."""
    df = metadata_frame(data)

    keep = np.ones(len(df), dtype=bool)
    keep &= data["seq_lengths"].astype(int) >= min_seq_len
    if "face_detect_rates" in data:
        keep &= data["face_detect_rates"].astype(float) >= min_face_rate

    x_seq       = data["x_seq"][keep].astype(np.float32)
    y           = data["y"][keep].astype(int)
    lengths     = data["seq_lengths"][keep].astype(int)
    df          = df[keep].reset_index(drop=True)
    feature_names = [str(f) for f in data["seq_feature_names"]]

    selected_idx      = select_seq_features(feature_names, feature_mode)
    selected_features = [feature_names[i] for i in selected_idx]
    x_seq             = x_seq[:, :, selected_idx]

    train_idx = np.where(df["split"].values == "train")[0]
    valid_idx = np.where(df["split"].values == "valid")[0]
    test_idx  = np.where(df["split"].values == "test")[0]

    scaler = fit_seq_scaler(x_seq, lengths, train_idx)
    x_seq  = transform_seq_with_lengths(x_seq, lengths, scaler)

    ds_train = FaceClipDataset(x_seq[train_idx], y[train_idx], lengths[train_idx])
    ds_valid = FaceClipDataset(x_seq[valid_idx], y[valid_idx], lengths[valid_idx])
    ds_test  = FaceClipDataset(x_seq[test_idx],  y[test_idx],  lengths[test_idx])

    return ds_train, ds_valid, ds_test, scaler, selected_features, selected_idx, df, train_idx, valid_idx, test_idx
