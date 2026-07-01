#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

EMOTIONS = ["happiness", "calm", "anticipation", "affection", "anger", "fear", "sadness", "disconnection", "suffering", "aversion", "embarrassment", "confidence", "confusion", "yearning"]

EMOTION_GROUPS = {
    "positive": ["happiness", "affection", "confidence", "anticipation", "calm"],
    "distress": ["sadness", "suffering", "disconnection", "embarrassment", "confusion", "yearning"],
    "threat": ["fear", "anger", "aversion"],
}

EMOTION_TO_GROUP = {emotion: group for group, emotions in EMOTION_GROUPS.items() for emotion in emotions}

# 정답과 같은 그룹인 오답을 선택지에서 배제할 확률.
# 1.0(항상 배제)은 attacker_pass_rate를 67.6%->94.7%까지 끌어올리는 것으로
# 확인돼(choice_features가 선택지 그룹 구성만으로 정답을 추론), 0.3으로 낮춰
# 사람 체감 개선과 공격 저항력 사이 절충점을 취한다.
SAME_GROUP_EXCLUSION_RATE = 0.3

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
    # aux_emotions는 항상 제외한다. 같은 그룹 오답은 SAME_GROUP_EXCLUSION_RATE
    # 확률로만 제외한다 — 매번 배제하면 "선택지에 정답 그룹이 하나만 있다"는
    # 패턴이 100% 확정적이 되어 메타데이터만 보는 공격자에게 정답이 그대로
    # 드러난다 (SAME_GROUP_EXCLUSION_RATE 설명 참고). 배제 여부는 sample_id
    # 기반 난수로 결정해 shuffle에 쓰는 seed(요청마다 달라짐)와 분리한다.
    excluded = set(parse_aux(row.get("aux_emotions", "[]")))
    excl_rng = random.Random(f"{row.get('sample_id', '')}_group_excl")
    if excl_rng.random() < SAME_GROUP_EXCLUSION_RATE:
        excluded |= {e for e in EMOTION_GROUPS.get(EMOTION_TO_GROUP.get(final), []) if e != final}
    choices = [final]

    # 1. Wrong labels actually chosen by attackers.
    for key in ["qwen_emotion", "smolvlm_emotion", "self_attack_emotion"]:
        e = row.get(key, "")
        if e not in excluded:
            add_choice(choices, e, final)
        if len(choices) >= 4:
            break

    # 2. Fill with manually curated close-but-different-group emotions.
    for emotion in FALLBACK_CONFUSIONS.get(final, []):
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
