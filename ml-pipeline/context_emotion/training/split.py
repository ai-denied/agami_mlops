"""Train/val/test loading for context_emotion_train_dataset_v2.csv.

Unlike flashlight (training/split.py there computes a fresh group split
every run), the split here is already fixed upstream by dataset_split /
split_group_id in build_train_dataset_v2.py - this module just loads each
partition and derives class weights from it.
"""
from collections import Counter
from typing import Dict, List

import torch

from context_emotion.common.constants import EMOTION_CLASSES
from context_emotion.data.dataset import load_rows


def load_splits(train_csv: str) -> Dict[str, List[Dict]]:
    splits = {name: load_rows(train_csv, name) for name in ("train", "val", "test")}
    for name, rows in splits.items():
        counts = Counter(r["provisional_emotion"] for r in rows)
        print(f"[{name}] {len(rows)} rows -> {dict(counts)}")
    return splits


def class_weights(train_rows: List[Dict], device) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss - the low_resource
    classes flagged in label_distribution_v2.md (sadness, aversion,
    embarrassment) need this or they'll be drowned out by the 800-cap
    classes."""
    counts = Counter(r["provisional_emotion"] for r in train_rows)
    total = sum(counts.values())
    weights = [
        total / (len(EMOTION_CLASSES) * counts.get(cls, 1))
        for cls in EMOTION_CLASSES
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)
