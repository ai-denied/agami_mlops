"""
감정 맥락 추론 캡챠 동적 생성기
==================================
WBS 2.1.3: 1챌린지 = N개 문제 시퀀스. 각 문제는 1장 이미지 + 4지선다.

설계 원칙
---------
1. flashlight_generator / face_generator 와 동일한 (spec, answer) 페어 반환 패턴.
2. 정답은 spec 의 ContextQuestion 에 들어가지 않고 answer.correct_answers 에만 보관.
3. 매 문제마다 choices 독립 셔플 → "1번 보기가 항상 정답" 류 학습 방어.
4. 한 챌린지의 N문제는 서로 다른 감정 그룹에서 1장씩 sample (감정 중복 없음).
5. 난이도는 question_count(2/3/4) + time_limit 으로 조절.

이 모듈의 책임 경계
-------------------
- [O] N개 문제 sampling, choices 셔플, 만료시간 계산
- [O] (spec, answer) 페어 반환
- [X] 정답 검증              -> app/captcha/verifier.py:check_context_hit
- [X] Redis 저장 / API 응답   -> app/cache/challenge_store, app/api/public.py
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Final

from app.captcha.challenge_types import (
    ChallengeKind,
    ContextChallengeAnswer,
    ContextChallengeSpec,
    ContextQuestion,
    Difficulty,
    Emotion,
)
from app.captcha.context_dataset import EMOTION_GROUPS, get_question_set


# ---------------------------------------------------------------------------
# 난이도 프로필
#   question_count : 출제 문제 수.
#   time_limit_sec / hint_after_sec : 전체 시간 압박.
# ---------------------------------------------------------------------------

DIFFICULTY_PROFILES: Final[dict[Difficulty, dict]] = {
    Difficulty.EASY: {
        "question_count": 2,
        "time_limit_sec": 30,
        "hint_after_sec": 12,
    },
    Difficulty.MEDIUM: {
        "question_count": 3,
        "time_limit_sec": 30,
        "hint_after_sec": 12,
    },
    Difficulty.HARD: {
        "question_count": 4,
        "time_limit_sec": 30,
        "hint_after_sec": None,
    },
}


# 4지선다 미끼 수 (정답 1 + 미끼 3 = 4)
_DECOY_COUNT: Final[int] = 3


# ---------------------------------------------------------------------------
# 메인 생성 함수
# ---------------------------------------------------------------------------

def generate_context_challenge(
    difficulty: Difficulty = Difficulty.MEDIUM,
    *,
    rng: secrets.SystemRandom | None = None,
    now: datetime | None = None,
) -> tuple[ContextChallengeSpec, ContextChallengeAnswer]:
    """
    감정 맥락 추론 캡챠 1챌린지 (=N문제 시퀀스) 생성.

    Returns
    -------
    (spec, answer)
        spec   : 클라이언트로 보낼 사양. questions 리스트, correct_answers 없음.
        answer : 서버 보관용 정답. correct_answers (출제 순서) 만 보관.
    """
    rng = rng or secrets.SystemRandom()
    now = now or datetime.now(timezone.utc)
    profile = DIFFICULTY_PROFILES[difficulty]
    count = profile["question_count"]

    # 1) 서로 다른 N개 감정 그룹에서 1장씩 sample
    raw_questions = get_question_set(rng, count=count)

    # 2) 매 문제마다 미끼 3개 + 정답 1개 셔플 → ContextQuestion 생성
    all_emotions = list(EMOTION_GROUPS.keys())
    questions: list[ContextQuestion] = []
    correct_answers: list[str] = []
    for idx, q in enumerate(raw_questions):
        correct = q["correct_emotion"]
        others = [e for e in all_emotions if e != correct]
        decoys = rng.sample(others, k=_DECOY_COUNT)
        choices_raw = [correct, *decoys]
        rng.shuffle(choices_raw)
        questions.append(ContextQuestion(
            index=idx,
            image_url=q["image_url"],
            choices=[Emotion(c) for c in choices_raw],
        ))
        correct_answers.append(correct)

    challenge_id = secrets.token_urlsafe(16)
    expires_at = now + timedelta(seconds=profile["time_limit_sec"] + 10)

    spec = ContextChallengeSpec(
        challenge_id=challenge_id,
        kind=ChallengeKind.CONTEXT_INFERENCE,
        difficulty=difficulty,
        issued_at=now,
        expires_at=expires_at,
        questions=questions,
        total_count=count,
        time_limit_sec=profile["time_limit_sec"],
        hint_after_sec=profile["hint_after_sec"],
    )

    answer = ContextChallengeAnswer(
        challenge_id=challenge_id,
        correct_answers=correct_answers,
        created_at=now,
        expires_at=expires_at,
    )

    return spec, answer


# ---------------------------------------------------------------------------
# CLI 동작 확인 — python -m app.captcha.context_generator
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    for diff in Difficulty:
        spec, answer = generate_context_challenge(diff)
        print(f"=== {diff.value.upper()} ===")
        print("[client spec]")
        print(json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print("[server answer]")
        print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print()
