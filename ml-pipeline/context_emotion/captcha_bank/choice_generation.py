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
    # aux_emotions는 선택지에서 제외 — 정답과 구별하기 어려운 근접 감정이
    # 오보기로 나오면 사용자 혼란을 유발하므로 선택지 풀에서 배제한다.
    # (scoring의 choice_credit에서는 여전히 참조되지 않지만 로그 분석용으로 유지)
    excluded = set(parse_aux(row.get("aux_emotions", "[]")))
    choices = [final]

    # 1. Wrong labels actually chosen by attackers.
    for key in ["qwen_emotion", "smolvlm_emotion", "self_attack_emotion"]:
        e = row.get(key, "")
        if e not in excluded:
            add_choice(choices, e, final)
        if len(choices) >= 4:
            break

    # 2. Fill with same-group or manually close emotions.
    for emotion in FALLBACK_CONFUSIONS.get(final, []):
        if emotion not in excluded:
            add_choice(choices, emotion, final)
        if len(choices) >= 4:
            break

    group = EMOTION_TO_GROUP.get(final)
    for emotion in EMOTION_GROUPS.get(group, []):
        if emotion not in excluded:
            add_choice(choices, emotion, final)
        if len(choices) >= 4:
            break

    # 3. Last-resort fill with globally valid labels.
    for emotion in EMOTIONS:
        if emotion not in excluded:
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
