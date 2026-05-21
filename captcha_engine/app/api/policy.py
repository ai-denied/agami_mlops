"""
Rate Limit + Dynamic Difficulty Policy
=======================================
WBS #45: 어뷰징 방지 정책의 순수 로직.

이 모듈은 Redis/DB 에 의존하지 않는 순수 함수만 둠.
사용 측(deps.py / public.py)은 카운터 값을 주입하고 결정만 받아간다.
이렇게 분리해두면 테스트가 단순해진다 (test_policy.py).
"""

from __future__ import annotations

from app.captcha.challenge_types import Difficulty


# ---------------------------------------------------------------------------
# Rate Limit
# ---------------------------------------------------------------------------

# IP 와 API key 양쪽에 동일한 1분 창 카운터를 둔다.
# - IP: 한 단말의 폭주를 막음 (한 명이 한 키를 도배하는 패턴)
# - API key: 한 키 전체의 폭주를 막음 (다중 IP 분산 공격에도 상한)
RATE_LIMIT_WINDOW_SEC = 60

# IP 단위 안전 상한. tenant_settings 의 분당 한도와는 별개로 IP 한 개가
# 분당 이 값을 넘기면 무조건 차단 (분산 봇넷이 아니라 단일 IP 폭주는 어뷰징 가정).
DEFAULT_PER_IP_LIMIT_PER_MIN = 30


def is_rate_limited(count: int, limit: int) -> bool:
    """카운터 값이 한도를 초과했는지. 한도 정확히 같으면 통과 (limit-th 호출까지 허용)."""
    return count > limit


# ---------------------------------------------------------------------------
# Dynamic Difficulty
# ---------------------------------------------------------------------------

# 같은 IP 의 최근 실패 횟수 누적 창. 정상 사용자가 1~2번 틀리는 건 봐주되,
# 그 이상 누적되면 봇/스크립트 의심 → 난이도 상향.
FAILURE_WINDOW_SEC = 600  # 10 분

# 난이도 상향 임계치. 보수적으로 잡음 (정상 사용자 UX 보호).
THRESHOLD_TO_MEDIUM = 1   # 1회 실패부터 medium
THRESHOLD_TO_HARD = 3     # 3회부터 hard


def decide_difficulty(
    requested: Difficulty | None,
    tenant_default: Difficulty,
    recent_failure_count: int,
) -> Difficulty:
    """
    난이도 결정 로직.

    우선순위:
    1. 클라이언트가 difficulty 를 명시한 경우 → 그대로 사용. 동적 상향 무시.
       (개발자/대시보드가 의도적으로 선택한 난이도이므로 존중. 프론트 드롭다운으로
        easy/medium/hard 를 선택하는 데모/대시보드 UX 보존.)
    2. 명시값 없으면 tenant_default 에서 시작, 같은 IP 의 누적 실패에 따라 동적 상향.
       (어뷰저는 보통 명시 안 하고 default 로 폭주하므로 anti-abuse 동작은 유지.)
    """
    if requested is not None:
        return requested

    if recent_failure_count >= THRESHOLD_TO_HARD:
        return Difficulty.HARD
    if recent_failure_count >= THRESHOLD_TO_MEDIUM:
        return _at_least(tenant_default, Difficulty.MEDIUM)
    return tenant_default


def _at_least(current: Difficulty, floor: Difficulty) -> Difficulty:
    """current 가 floor 보다 낮으면 floor 로 끌어올림."""
    order = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}
    return current if order[current] >= order[floor] else floor
