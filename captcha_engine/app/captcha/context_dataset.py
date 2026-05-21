"""
감정 맥락 추론(context_inference) 캡챠 데이터셋
==================================================
WBS 2.1.3: 감정별 이미지 그룹 카탈로그.

설계 노트
---------
- 챌린지 1회 = 서로 다른 감정 그룹에서 1장씩 sample 한 N문제 시퀀스.
- placeholder URL 은 감정별 hex 색상 + text 라벨 (`?text=joy_1`) 을 포함해
  데모/QA 가 정답을 시각적으로 추론할 수 있도록 함. 실제 이미지로 교체 시
  항목별 image_url 만 갱신.
- context_hint 는 디버깅/QA 전용. ContextChallengeSpec 에 해당 필드가
  존재하지 않으므로 클라이언트로는 구조적으로 노출되지 않음.
"""

from __future__ import annotations

import secrets
from typing import Final


# 감정별 hex 색상 (placeholder 배경). 색상으로도 감정 구분 가능.
_EMOTION_HEX: Final[dict[str, str]] = {
    "joy":      "FFD700",  # 황금
    "sadness":  "4682B4",  # 청색
    "anger":    "DC143C",  # 적색
    "fear":     "4B0082",  # 인디고
    "surprise": "FFA500",  # 주황
    "disgust":  "556B2F",  # 진녹색
    "contempt": "708090",  # 슬레이트 그레이
}


def _placeholder(emotion: str, n: int) -> str:
    """placehold.co 의 색상 URL 생성. text 파라미터로 정답 추론 가능 (데모용)."""
    hex_color = _EMOTION_HEX[emotion]
    return f"https://placehold.co/400x400/{hex_color}/000?text={emotion}_{n}"


def _group(emotion: str, hints: list[str]) -> list[dict]:
    """감정 1개에 대한 이미지 그룹 빌더. hint 개수만큼 항목 생성."""
    return [
        {
            "id": f"{emotion}_{i+1:03d}",
            "image_url": _placeholder(emotion, i + 1),
            "context_hint": h,
        }
        for i, h in enumerate(hints)
    ]


# 감정별 이미지 그룹. 각 그룹 4장 (hard=4문제 시 한 감정만 뽑힐 일은 없지만 여유 보유).
EMOTION_GROUPS: Final[dict[str, list[dict]]] = {
    "joy": _group("joy", [
        "훈련 후 동료들과 미소짓는 군인",
        "결혼식 단상 위 환하게 웃는 신부",
        "졸업장 받고 환호하는 학생",
        "선물 받고 활짝 웃는 어린이",
    ]),
    "sadness": _group("sadness", [
        "장례식장에서 고개 숙인 가족",
        "비 오는 거리, 우산 없이 어깨 늘어뜨린 사람",
        "이별 통보 후 창밖을 보는 사람",
        "병상 옆 손을 잡고 눈물 흘리는 가족",
    ]),
    "anger": _group("anger", [
        "접촉 사고 후 차에서 내려 언쟁하는 두 사람",
        "경기 종료 직후 주먹 쥐고 소리지르는 선수",
        "회의실에서 책상을 내리치는 임원",
        "교통 체증 속에서 경적을 누르는 운전자",
    ]),
    "fear": _group("fear", [
        "어두운 골목에서 뒤를 돌아보는 사람",
        "공포 영화 한 장면을 보며 손으로 입을 가린 관객",
        "롤러코스터 정점에서 눈 감은 탑승자",
        "지진 후 길거리에서 떨고 있는 시민",
    ]),
    "surprise": _group("surprise", [
        "생일 선물 상자 열고 눈 커진 사람",
        "깜짝 파티에서 들어선 주인공",
        "예상 못한 합격 통보를 본 직후",
        "프러포즈 반지를 본 순간",
    ]),
    "disgust": _group("disgust", [
        "상한 음식 냄새에 코 막은 표정",
        "벌레가 들어간 음료 컵을 본 표정",
        "쓰레기통 옆 악취에 얼굴을 찌푸린 사람",
        "곰팡이 핀 빵을 발견한 직후",
    ]),
    "contempt": _group("contempt", [
        "팔짱 끼고 한쪽 입꼬리만 올린 비웃음",
        "거만한 표정으로 상대를 내려다보는 인물",
        "허세 부리는 동료를 쳐다보는 차가운 시선",
        "후배의 실수를 비꼬듯 보는 선배",
    ]),
}


# ---------------------------------------------------------------------------
# 신규: N문제 시퀀스용 sampler
# ---------------------------------------------------------------------------

def get_question_set(
    rng: secrets.SystemRandom | None = None,
    count: int = 2,
) -> list[dict]:
    """
    서로 다른 N개 감정 그룹을 균등 무작위로 고른 뒤 각 그룹에서 1장씩 sample.

    Parameters
    ----------
    rng   : 외부 주입 가능. None 이면 새 SystemRandom.
    count : 문제 수. EMOTION_GROUPS 의 감정 개수 이하여야 함.

    Returns
    -------
    list[dict] : 각 원소 = {"id", "image_url", "correct_emotion"}.
                 출제 순서(index) 는 호출 측이 반환 리스트 인덱스로 결정.
    """
    rng = rng or secrets.SystemRandom()
    emotions = list(EMOTION_GROUPS.keys())
    if count > len(emotions):
        raise ValueError(
            f"감정 카탈로그 크기({len(emotions)}) 보다 많은 문제({count}) 를 요청함."
        )
    chosen_emotions = rng.sample(emotions, k=count)
    questions: list[dict] = []
    for emotion in chosen_emotions:
        pool = EMOTION_GROUPS[emotion]
        item = rng.choice(pool)
        questions.append({
            "id": item["id"],
            "image_url": item["image_url"],
            "correct_emotion": emotion,
        })
    return questions


# ---------------------------------------------------------------------------
# DEPRECATED: 이전 인터랙션(N장 동시 노출 또는 단일 이미지) API
# ---------------------------------------------------------------------------

# DEPRECATED: 단일 챌린지 = N문제 시퀀스 도입(2026-05) 이후 사용 안 함.
def get_random_emotion_set(
    rng: secrets.SystemRandom | None = None,
    count: int = 2,
) -> tuple[str, list[dict]]:
    """[DEPRECATED] 한 챌린지가 N장의 같은 감정 이미지를 보여주던 시절 API.

    하위 호환을 위해 시그니처 보존 — 임의 감정 1개 그룹에서 count 장 sample.
    신규 코드는 get_question_set 사용.
    """
    rng = rng or secrets.SystemRandom()
    emotion = rng.choice(list(EMOTION_GROUPS.keys()))
    pool = EMOTION_GROUPS[emotion]
    if count > len(pool):
        raise ValueError(
            f"감정 그룹 '{emotion}' 크기({len(pool)}) 보다 많은 이미지({count}) 를 요청함."
        )
    items = rng.sample(pool, k=count)
    return emotion, items


# DEPRECATED: 단일 이미지 시절 API.
def get_random_context_item(rng: secrets.SystemRandom | None = None) -> dict:
    """[DEPRECATED] 단일 이미지 + 4지선다 시절 API. 신규 코드는 get_question_set 사용."""
    rng = rng or secrets.SystemRandom()
    emotion, items = get_random_emotion_set(rng, count=1)
    item = items[0]
    others = [e for e in EMOTION_GROUPS.keys() if e != emotion]
    decoys = rng.sample(others, k=3)
    return {
        "id": item["id"],
        "image_url": item["image_url"],
        "correct_emotion": emotion,
        "choices": [emotion, *decoys],
        "context_hint": item["context_hint"],
    }


# ---------------------------------------------------------------------------
# CLI 동작 확인 — python -m app.captcha.context_dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    for count in (2, 3, 4):
        questions = get_question_set(count=count)
        print(f"=== count={count} ===")
        print(json.dumps(questions, ensure_ascii=False, indent=2))
        print()
