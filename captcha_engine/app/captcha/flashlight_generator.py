"""
손전등 캡챠 동적 생성기
========================
WBS #41: 캡챠 문제(Challenge) 동적 생성 로직

이미지 데이터셋 통합 후 흐름:
  - app/static/captcha_images/captcha_*.jpg (1000장) + 같은 이름의 JSON 라벨
  - 챌린지당 무작위 3장 선택, 각 이미지의 bbox 중심을 정답 좌표로 사용
  - DIFFICULTY_PROFILES 의 radius / time_limit / hint 만 난이도에 따라 변동

설계 원칙
---------
1. 정답은 spec 에 들어가지 않음. (spec, answer) 튜플로 분리 반환.
2. 암호학적 난수 (`secrets`) 사용.
3. 데이터셋은 모듈 lazy singleton (앱 시작 시 1회 인덱싱).
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Final

from app.captcha.challenge_types import (
    Difficulty,
    FlashlightChallengeAnswer,
    FlashlightChallengeSpec,
    FlashlightSubAnswer,
    FlashlightSubChallenge,
    FlashlightTargetHint,
    FlashlightVariant,
)
from app.captcha.flashlight_image_dataset import pick_random_entries


# ---------------------------------------------------------------------------
# 난이도 프로필.
# flashlight_radius: 손전등이 비추는 반경 (캔버스 짧은 변 기준 비율).
#   기존 값의 절반으로 축소 — 사용자 요청.
# time_limit_sec / hint_after_sec 은 그대로.
# decoy_count / edge_padding / min_separation / tolerance 는 이미지 캡챠로
# 전환되며 사용되지 않지만 (좌표/decoy 샘플링 미수행) 호환을 위해 키만 유지.
# ---------------------------------------------------------------------------

DIFFICULTY_PROFILES: Final[dict[Difficulty, dict]] = {
    Difficulty.EASY: {
        "flashlight_radius": 0.24,        # was 0.6
        "time_limit_sec": 60,
        "hint_after_sec": 15,
        "decoy_count": 0,
        "tolerance": 0.06,
        "edge_padding": 0.15,
        "min_separation": 0.25,
    },
    Difficulty.MEDIUM: {
        "flashlight_radius": 0.2,    # was 0.4667
        "time_limit_sec": 45,
        "hint_after_sec": 18,
        "decoy_count": 1,
        "tolerance": 0.05,
        "edge_padding": 0.10,
        "min_separation": 0.25,
    },
    Difficulty.HARD: {
        "flashlight_radius": 0.15,        # was 0.4
        "time_limit_sec": 30,
        "hint_after_sec": None,
        "decoy_count": 2,
        "tolerance": 0.04,
        "edge_padding": 0.08,
        "min_separation": 0.20,
    },
}


# ---------------------------------------------------------------------------
# 메인 생성 함수
# ---------------------------------------------------------------------------

def generate_flashlight_challenge(
    difficulty: Difficulty = Difficulty.MEDIUM,
    *,
    rng: secrets.SystemRandom | None = None,
    now: datetime | None = None,
) -> tuple[FlashlightChallengeSpec, FlashlightChallengeAnswer]:
    """
    손전등 캡챠 1개 챌린지 = 3장 묶음을 생성한다.

    이미지 데이터셋에서 서로 다른 3장을 무작위로 뽑고 각 이미지의 bbox 중심을
    정답 좌표로, bbox 폭/높이를 verifier 가 사각형 매칭에 사용하도록 보관한다.
    flashlight_radius / time_limit_sec / hint_after_sec / variant 은 번들 단위.

    Returns
    -------
    (spec, answer)
        spec   : 클라이언트로 보낼 사양 (sub_challenges 3개, 이미지 URL 포함).
        answer : 서버 보관용 정답 (sub_answers 3개, bbox 좌표 포함).
    """
    rng = rng or secrets.SystemRandom()
    now = now or datetime.now(timezone.utc)
    profile = DIFFICULTY_PROFILES[difficulty]

    entries = pick_random_entries(rng, k=3)

    sub_challenges: list[FlashlightSubChallenge] = []
    sub_answers: list[FlashlightSubAnswer] = []

    for i, entry in enumerate(entries):
        sub_challenges.append(
            FlashlightSubChallenge(
                index=i,
                image_url=entry.image_url(),
                target_hint=FlashlightTargetHint(
                    object_id=entry.target_object_id,
                    label=entry.target_label,
                    emoji="",  # 이미지 캡챠는 이모지 미사용
                ),
                decoys=[],
            )
        )
        sub_answers.append(
            FlashlightSubAnswer(
                index=i,
                correct_object_id=entry.target_object_id,
                correct_x=entry.center_x_norm,
                correct_y=entry.center_y_norm,
                tolerance=profile["tolerance"],  # bbox=0 fallback 시 사용
                bbox_w=entry.bbox_w_norm,
                bbox_h=entry.bbox_h_norm,
                image_url=entry.image_url(),       # 로그용 메타
                target_label=entry.target_label,   # 로그용 메타
            )
        )

    challenge_id = secrets.token_urlsafe(16)  # 128 bit, URL-safe
    expires_at = now + timedelta(seconds=profile["time_limit_sec"] + 10)

    spec = FlashlightChallengeSpec(
        challenge_id=challenge_id,
        difficulty=difficulty,
        issued_at=now,
        expires_at=expires_at,
        variant=FlashlightVariant.SINGLE_TARGET,  # 이미지 캡챠는 단일 정답
        sub_challenges=sub_challenges,
        flashlight_radius=profile["flashlight_radius"],
        time_limit_sec=profile["time_limit_sec"],
        hint_after_sec=profile["hint_after_sec"],
        canvas_aspect_w=4,
        canvas_aspect_h=3,
    )

    answer = FlashlightChallengeAnswer(
        challenge_id=challenge_id,
        sub_answers=sub_answers,
        created_at=now,
        expires_at=expires_at,
        # 로그용 메타 — 검증에는 미사용
        difficulty=difficulty.value,
        flashlight_radius=profile["flashlight_radius"],
        time_limit_sec=profile["time_limit_sec"],
        canvas_aspect_w=4,
        canvas_aspect_h=3,
    )

    return spec, answer


# ---------------------------------------------------------------------------
# CLI 형태 동작 확인 (python flashlight_generator.py 로 즉시 검증)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    for diff in Difficulty:
        spec, answer = generate_flashlight_challenge(diff)
        print(f"=== {diff.value.upper()} ===")
        print("[client spec]")
        print(json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print("[server answer]")
        print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print()
