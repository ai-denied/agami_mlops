"""
Tests for app/api/policy.py
============================
WBS #45: 순수 함수 (Redis/DB 의존 없음) 만 테스트.
실행: python tests/test_policy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.policy import (
    DEFAULT_PER_IP_LIMIT_PER_MIN,
    decide_difficulty,
    is_rate_limited,
)
from app.captcha.challenge_types import Difficulty


def test_rate_limit_threshold() -> None:
    # limit 정확히 같으면 통과 (limit-th 호출 허용)
    assert is_rate_limited(DEFAULT_PER_IP_LIMIT_PER_MIN, DEFAULT_PER_IP_LIMIT_PER_MIN) is False
    # limit + 1 이면 차단
    assert is_rate_limited(DEFAULT_PER_IP_LIMIT_PER_MIN + 1, DEFAULT_PER_IP_LIMIT_PER_MIN) is True
    # 0 호출 → 절대 차단 안 됨
    assert is_rate_limited(0, DEFAULT_PER_IP_LIMIT_PER_MIN) is False


def test_decide_difficulty_explicit_request_no_failures() -> None:
    # 요청 명시값이 있고 실패가 없으면 그대로
    assert decide_difficulty(Difficulty.EASY, Difficulty.MEDIUM, 0) == Difficulty.EASY
    assert decide_difficulty(Difficulty.HARD, Difficulty.EASY, 0) == Difficulty.HARD


def test_decide_difficulty_falls_back_to_tenant_default() -> None:
    assert decide_difficulty(None, Difficulty.MEDIUM, 0) == Difficulty.MEDIUM
    assert decide_difficulty(None, Difficulty.EASY, 0) == Difficulty.EASY


def test_decide_difficulty_bumps_on_failures() -> None:
    # 실패 1회: easy 요청도 medium 으로 상향
    assert decide_difficulty(Difficulty.EASY, Difficulty.EASY, 1) == Difficulty.MEDIUM
    # 실패 2회: 여전히 medium 단계
    assert decide_difficulty(None, Difficulty.EASY, 2) == Difficulty.MEDIUM
    # 실패 3회 이상: hard 강제 (요청값 무관)
    assert decide_difficulty(Difficulty.EASY, Difficulty.EASY, 3) == Difficulty.HARD
    assert decide_difficulty(Difficulty.HARD, Difficulty.HARD, 10) == Difficulty.HARD


def test_decide_difficulty_does_not_lower_explicit_request() -> None:
    # 요청이 hard 인데 실패 1회 → 여전히 hard (medium 으로 내리지 않음)
    assert decide_difficulty(Difficulty.HARD, Difficulty.EASY, 1) == Difficulty.HARD


if __name__ == "__main__":
    test_rate_limit_threshold()
    test_decide_difficulty_explicit_request_no_failures()
    test_decide_difficulty_falls_back_to_tenant_default()
    test_decide_difficulty_bumps_on_failures()
    test_decide_difficulty_does_not_lower_explicit_request()
    print("OK — all policy tests passed")
