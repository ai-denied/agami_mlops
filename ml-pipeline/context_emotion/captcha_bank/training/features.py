"""Feature extraction for CAPTCHA pool attacker proxy models.

Two feature sets are defined:

choice_features (20-dim)
    Features visible to a metadata-only attacker who sees the 4-choice set
    but NOT the image.  Measures how much information leaks from the choice
    set alone — a baseline lower bound on attacker solve rate.

difficulty_features (4-dim)
    Features for predicting question difficulty (attack_hardness) from pool
    metadata.  Used to train the security_ranker model.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from context_emotion.captcha_bank.choice_generation import (
    EMOTION_GROUPS,
    EMOTION_TO_GROUP,
    EMOTIONS,
    generate_choices,
)

# ── Feature name lists (for explainability / logging) ────────────────────────

CHOICE_FEATURE_NAMES: list[str] = [f"choice_{e}" for e in EMOTIONS]            # 14
GROUP_FEATURE_NAMES:  list[str] = [f"group_count_{g}" for g in EMOTION_GROUPS]  # 6
ATTACKER_FEATURE_NAMES = CHOICE_FEATURE_NAMES + GROUP_FEATURE_NAMES             # 20

DIFFICULTY_FEATURE_NAMES = [
    "emotion_idx_norm",
    "group_idx_norm",
    "aux_count_norm",
    "attack_hardness",
]  # 4

_GROUPS = list(EMOTION_GROUPS.keys())


def choice_features(row: dict, seed: Optional[int] = None) -> np.ndarray:
    """20-dim features derived from the 4-choice set (no image signal).

    The seed is fixed per sample_id so features are deterministic across
    train/eval runs.
    """
    effective_seed = (
        seed if seed is not None
        else abs(hash(str(row.get("sample_id", "")))) % (2 ** 31)
    )
    choices = generate_choices(row, seed=effective_seed)

    in_choices = [1.0 if e in choices else 0.0 for e in EMOTIONS]
    group_counts = [
        float(sum(1 for e in EMOTION_GROUPS[g] if e in choices))
        for g in _GROUPS
    ]
    return np.array(in_choices + group_counts, dtype=np.float32)


def difficulty_features(row: dict) -> np.ndarray:
    """4-dim features for attack_hardness regression."""
    emotion = row.get("final_emotion", "")
    emotion_idx = EMOTIONS.index(emotion) / len(EMOTIONS) if emotion in EMOTIONS else 0.5

    group = EMOTION_TO_GROUP.get(emotion, "")
    group_idx = _GROUPS.index(group) / len(_GROUPS) if group in _GROUPS else 0.5

    aux_count = min(len(_parse_list(row.get("aux_emotions", "[]"))), 5) / 5.0

    try:
        hardness = float(row.get("attack_hardness", 0.5))
        # attack_hardness is computed from VLMs — normalise to [0, 1]
        hardness = max(0.0, min(1.0, hardness))
    except (ValueError, TypeError):
        hardness = 0.5

    return np.array([emotion_idx, group_idx, aux_count, hardness], dtype=np.float32)


def build_attacker_matrix(rows: list[dict], seed: Optional[int] = None) -> np.ndarray:
    return np.vstack([choice_features(r, seed=seed) for r in rows])


def build_difficulty_matrix(rows: list[dict]) -> np.ndarray:
    return np.vstack([difficulty_features(r) for r in rows])


def emotion_labels(rows: list[dict]) -> list[str]:
    return [r.get("final_emotion", "") for r in rows]


def attack_hardness_targets(rows: list[dict]) -> np.ndarray:
    out = []
    for r in rows:
        try:
            out.append(max(0.0, min(1.0, float(r.get("attack_hardness", 0.5)))))
        except (ValueError, TypeError):
            out.append(0.5)
    return np.array(out, dtype=np.float32)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _parse_list(text: str) -> list[str]:
    try:
        v = json.loads(text or "[]")
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []
