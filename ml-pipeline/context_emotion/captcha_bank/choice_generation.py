#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

EMOTIONS = ["happiness", "calm", "anticipation", "affection", "anger", "fear", "sadness", "disconnection", "suffering", "aversion", "embarrassment", "confidence", "confusion", "yearning"]

EMOTION_GROUPS = {
    "positive": ["happiness", "affection", "confidence", "anticipation"],
    "calm": ["calm"],
    "distress": ["sadness", "suffering", "disconnection", "embarrassment"],
    "threat": ["fear", "anger", "aversion"],
    "confusion": ["confusion"],
    "yearning": ["yearning"],
}

EMOTION_TO_GROUP = {emotion: group for group, emotions in EMOTION_GROUPS.items() for emotion in emotions}

FALLBACK_CONFUSIONS = {
    "happiness": ["affection", "confidence", "anticipation", "calm"],
    "affection": ["happiness", "calm", "yearning", "confidence"],
    "confidence": ["happiness", "anticipation", "anger", "affection"],
    "anticipation": ["happiness", "confidence", "confusion", "yearning"],
    "calm": ["happiness", "affection", "disconnection", "sadness"],
    "sadness": ["suffering", "disconnection", "yearning", "fear"],
    "suffering": ["sadness", "disconnection", "fear", "anger"],
    "disconnection": ["sadness", "suffering", "calm", "confusion"],
    "embarrassment": ["confusion", "sadness", "disconnection", "fear"],
    "fear": ["sadness", "confusion", "anger", "suffering"],
    "anger": ["aversion", "fear", "disconnection", "confidence"],
    "aversion": ["anger", "fear", "disconnection", "embarrassment"],
    "confusion": ["fear", "disconnection", "embarrassment", "anticipation"],
    "yearning": ["sadness", "affection", "anticipation", "calm"],
}


def parse_aux(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return [v.strip() for v in str(value).split(";") if v.strip()]
    return [str(v) for v in parsed if str(v).strip()] if isinstance(parsed, list) else []


def add_choice(choices: list[str], emotion: str, final_emotion: str) -> None:
    if emotion and emotion in EMOTIONS and emotion != final_emotion and emotion not in choices:
        choices.append(emotion)


def generate_choices(row: dict, seed: int | None = None) -> list[str]:
    rng = random.Random(seed if seed is not None else row.get("sample_id", ""))
    final = row["final_emotion"]
    choices = [final]

    # 1. Human auxiliary emotions are useful near-misses and allow partial credit.
    for emotion in parse_aux(row.get("aux_emotions", "[]")):
        add_choice(choices, emotion, final)
        if len(choices) >= 2:
            break

    # 2. Add wrong labels actually chosen by attackers.
    for key in ["qwen_emotion", "smolvlm_emotion", "self_attack_emotion"]:
        add_choice(choices, row.get(key, ""), final)
        if len(choices) >= 4:
            break

    # 3. Fill with same-group or manually close emotions.
    for emotion in FALLBACK_CONFUSIONS.get(final, []):
        add_choice(choices, emotion, final)
        if len(choices) >= 4:
            break

    group = EMOTION_TO_GROUP.get(final)
    for emotion in EMOTION_GROUPS.get(group, []):
        add_choice(choices, emotion, final)
        if len(choices) >= 4:
            break

    # 4. Last-resort fill with globally valid labels.
    for emotion in EMOTIONS:
        add_choice(choices, emotion, final)
        if len(choices) >= 4:
            break

    final_choices = choices[:4]
    rng.shuffle(final_choices)
    return final_choices


def choice_credit(selected: str, row: dict) -> float:
    if selected == row.get("final_emotion", ""):
        return 1.0
    if selected in parse_aux(row.get("aux_emotions", "[]")):
        return 0.5
    return 0.0


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
