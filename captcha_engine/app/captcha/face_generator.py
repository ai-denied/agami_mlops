"""
안면 미션 캡챠 동적 생성기
============================
WBS 2.1.2: 사용자가 카메라 앞에서 일련의 안면 동작을 수행하도록 지시.

설계 원칙
---------
1. flashlight_generator 와 동일한 인터페이스 패턴: (spec, answer) 튜플 반환,
   순수 함수, secrets.SystemRandom 사용.
2. 정답은 expected_instruction_types 만 보관. 실제 동작 자동 감지(MediaPipe)는
   팀원 합류 후 별도 모듈로 추가 → 본 단계는 클라이언트가 보고하는
   completed_instructions 를 그대로 비교하는 임시 로직으로 운영.
3. 지시 종류는 FaceInstructionType enum 의 모든 값 중 중복 없이 sample.

이 모듈의 책임 경계
-------------------
- [O] challenge_id 발급, 지시 종류/개수/시간 결정
- [O] (spec, answer) 페어 반환
- [X] 카메라 영상 분석          -> 팀원 MediaPipe 모듈 담당 (예정)
- [X] Redis 저장 / API 응답     -> WBS #43 의 challenge_store / public.py 담당
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Final

from app.captcha.challenge_types import (
    ChallengeKind,
    Difficulty,
    FaceChallengeAnswer,
    FaceChallengeSpec,
    FaceInstruction,
    FaceInstructionType,
    FACE_INSTRUCTION_LABELS,
)


# ---------------------------------------------------------------------------
# 난이도별 프로필
# ---------------------------------------------------------------------------

DIFFICULTY_PROFILES: Final[dict[Difficulty, dict]] = {
    Difficulty.EASY: {
        "instruction_count": 1,
        "duration_per_instruction_sec": 3,
        "time_limit_sec": 30,
        "hint_after_sec": 12,
        "tolerance_sec": 1.5,
    },
    Difficulty.MEDIUM: {
        "instruction_count": 2,
        "duration_per_instruction_sec": 3,  # 사용자 사양 "각 동작 2.5초" → 정수 보존 위해 3
        "time_limit_sec": 25,
        "hint_after_sec": 10,
        "tolerance_sec": 1.0,
    },
    Difficulty.HARD: {
        "instruction_count": 3,
        "duration_per_instruction_sec": 2,
        "time_limit_sec": 20,
        "hint_after_sec": None,
        "tolerance_sec": 0.8,
    },
}


# ---------------------------------------------------------------------------
# 메인 생성 함수
# ---------------------------------------------------------------------------

def generate_face_challenge(
    difficulty: Difficulty = Difficulty.MEDIUM,
    *,
    rng: secrets.SystemRandom | None = None,
    now: datetime | None = None,
) -> tuple[FaceChallengeSpec, FaceChallengeAnswer]:
    """
    안면 미션 캡챠 1개 인스턴스를 생성한다.

    Returns
    -------
    (spec, answer)
        spec   : 클라이언트로 보낼 사양 (지시 목록 포함)
        answer : 서버 보관용 정답 (expected_instruction_types 만 보관)
    """
    rng = rng or secrets.SystemRandom()
    now = now or datetime.now(timezone.utc)
    profile = DIFFICULTY_PROFILES[difficulty]

    count: int = profile["instruction_count"]
    duration: int = profile["duration_per_instruction_sec"]

    # 지시 종류는 중복 없이 sample.
    instruction_pool = list(FaceInstructionType)
    if count > len(instruction_pool):
        raise ValueError(
            f"지시 카탈로그({len(instruction_pool)}) 보다 많은 지시({count})를 요청함."
        )
    chosen: list[FaceInstructionType] = rng.sample(instruction_pool, k=count)

    instructions = [
        FaceInstruction(
            type=t,
            label=FACE_INSTRUCTION_LABELS[t],
            duration_sec=duration,
        )
        for t in chosen
    ]

    challenge_id = secrets.token_urlsafe(16)
    expires_at = now + timedelta(seconds=profile["time_limit_sec"] + 10)

    spec = FaceChallengeSpec(
        challenge_id=challenge_id,
        kind=ChallengeKind.FACE_MISSION,
        difficulty=difficulty,
        issued_at=now,
        expires_at=expires_at,
        instructions=instructions,
        time_limit_sec=profile["time_limit_sec"],
        hint_after_sec=profile["hint_after_sec"],
    )

    answer = FaceChallengeAnswer(
        challenge_id=challenge_id,
        expected_instruction_types=[t.value for t in chosen],
        tolerance_sec=profile["tolerance_sec"],
        created_at=now,
        expires_at=expires_at,
    )

    return spec, answer


# ---------------------------------------------------------------------------
# CLI 동작 확인
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    for diff in Difficulty:
        spec, answer = generate_face_challenge(diff)
        print(f"=== {diff.value.upper()} ===")
        print("[client spec]")
        print(json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print("[server answer]")
        print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print()
