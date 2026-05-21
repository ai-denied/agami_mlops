"""
Redis-backed Challenge Store
=============================
WBS #42: Hot path 저장소.

이 모듈의 책임
---------------
- #41 가 만든 FlashlightChallengeAnswer 를 Redis 에 보관 (TTL 자동 만료)
- 검증 시점에 1회용으로 꺼내고 즉시 삭제 (GETDEL — 재사용 공격 방지)
- 챌린지 ID 충돌 방지 (이미 같은 ID 가 있으면 거부 — token_urlsafe 충돌 가능성은
  사실상 0 이지만 안전 장치)

이 모듈이 책임지지 않는 것
---------------------------
- 정답 검증 로직 자체 (#43)
- 거리 비교 (#43)
- 기록 영속화 (#43 가 verifications 테이블에 INSERT)

설계 노트
---------
- redis.asyncio 사용. FastAPI 의 async 핸들러와 자연스럽게 어울림.
- Redis 6.2+ 의 GETDEL 명령으로 atomic 한 read-and-delete 보장.
  (Redis 6.0 이하라면 Lua 스크립트로 대체 — TODO 표시)
- 직렬화는 Pydantic v2 의 model_dump_json() 사용. 사람이 디버깅하기 쉬움.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import redis.asyncio as redis_async

if TYPE_CHECKING:
    # 41번 산출물. 실제 import 는 사용 측에서.
    from app.captcha.challenge_types import (
        FaceChallengeAnswer,
        FlashlightChallengeAnswer,
    )

# challenge_id → kind 디스패치에 쓰는 union alias.
StoredAnswer = "FlashlightChallengeAnswer | FaceChallengeAnswer"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 키 네이밍 규약 (한 곳에서 관리)
# ---------------------------------------------------------------------------

KEY_PREFIX = "captcha"


def k_answer(challenge_id: str) -> str:
    """챌린지 정답 저장 키. 1회용."""
    return f"{KEY_PREFIX}:answer:{challenge_id}"


def k_apikey_cache(client_key: str) -> str:
    """API key 메타데이터 캐시 키. DB 조회 부하 감소."""
    return f"{KEY_PREFIX}:apikey:{client_key}"


def k_rate_ip(ip: str, window: str) -> str:
    """IP 단위 rate limit 카운터. window 예: '1m', '1h'."""
    return f"{KEY_PREFIX}:rate:ip:{ip}:{window}"


def k_rate_apikey(client_key: str, window: str) -> str:
    """API key 단위 rate limit 카운터."""
    return f"{KEY_PREFIX}:rate:apikey:{client_key}:{window}"


def k_token(challenge_id: str) -> str:
    """captcha_token 페이로드 저장 키. /v1/siteverify 가 1회용 GETDEL."""
    return f"{KEY_PREFIX}:token:{challenge_id}"


def k_fail_ip(ip: str) -> str:
    """IP 단위 누적 실패 횟수 카운터. 동적 난이도 상향에 사용 (#45)."""
    return f"{KEY_PREFIX}:fail:ip:{ip}"


# ---------------------------------------------------------------------------
# Challenge Store
# ---------------------------------------------------------------------------

class ChallengeStore:
    """
    Redis 위에 얹은 얇은 wrapper. 모든 메서드는 async.

    Usage
    -----
        store = ChallengeStore(redis_client)
        spec, answer = generate_flashlight_challenge()
        await store.save_answer(answer)            # 발급 시
        ...
        ans = await store.consume_answer(cid)      # 검증 시 (1회용)
    """

    def __init__(self, redis: redis_async.Redis) -> None:
        self.redis = redis

    # -----------------------------------------------------------------------
    # 정답 저장 / 회수
    # -----------------------------------------------------------------------

    async def save_answer(self, answer) -> None:
        """
        정답을 Redis 에 저장. TTL = expires_at - now.
        같은 challenge_id 가 이미 존재하면 RuntimeError (충돌 방지).

        answer 타입: FlashlightChallengeAnswer | FaceChallengeAnswer
        둘 다 challenge_id, expires_at, model_dump_json() 인터페이스가 동일하므로
        구조적 다형성으로 동작.
        """
        from datetime import datetime, timezone

        ttl_seconds = int((answer.expires_at - datetime.now(timezone.utc)).total_seconds())
        if ttl_seconds <= 0:
            raise ValueError(f"answer already expired: ttl={ttl_seconds}s")

        key = k_answer(answer.challenge_id)
        payload = answer.model_dump_json()

        # SET ... NX EX ttl : NX 옵션으로 기존 키가 있으면 실패.
        # 결과가 None 이면 이미 같은 키가 존재한다는 뜻.
        result = await self.redis.set(key, payload, nx=True, ex=ttl_seconds)
        if result is None:
            raise RuntimeError(
                f"challenge_id collision (extremely rare): {answer.challenge_id}"
            )

    async def consume_answer(self, challenge_id: str):
        """
        정답을 atomic 하게 꺼내고 동시에 삭제 (one-shot).
        존재하지 않거나 이미 만료/소비된 경우 None.

        반환: FlashlightChallengeAnswer | FaceChallengeAnswer
              | ContextChallengeAnswer | None
        kind 필드로 디스패치해 적절한 모델로 역직렬화.
        """
        import json

        from app.captcha.challenge_types import (
            ChallengeKind,
            ContextChallengeAnswer,
            FaceChallengeAnswer,
            FlashlightChallengeAnswer,
        )

        key = k_answer(challenge_id)
        raw = await self.redis.getdel(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            kind = data.get("kind")
            if kind == ChallengeKind.FLASHLIGHT.value:
                return FlashlightChallengeAnswer.model_validate(data)
            if kind == ChallengeKind.FACE_MISSION.value:
                return FaceChallengeAnswer.model_validate(data)
            if kind == ChallengeKind.CONTEXT_INFERENCE.value:
                return ContextChallengeAnswer.model_validate(data)
            logger.warning("unknown kind in stored answer for %s: %r", challenge_id, kind)
            return None
        except Exception:
            logger.exception("failed to parse stored answer for %s", challenge_id)
            return None

    async def peek_answer(self, challenge_id: str):
        """
        삭제 없이 읽기. 운영/디버깅 용도. 일반 검증 경로는 consume_answer 를 써야 함.
        """
        import json

        from app.captcha.challenge_types import (
            ChallengeKind,
            ContextChallengeAnswer,
            FaceChallengeAnswer,
            FlashlightChallengeAnswer,
        )

        raw = await self.redis.get(k_answer(challenge_id))
        if raw is None:
            return None
        data = json.loads(raw)
        kind = data.get("kind")
        if kind == ChallengeKind.FLASHLIGHT.value:
            return FlashlightChallengeAnswer.model_validate(data)
        if kind == ChallengeKind.FACE_MISSION.value:
            return FaceChallengeAnswer.model_validate(data)
        if kind == ChallengeKind.CONTEXT_INFERENCE.value:
            return ContextChallengeAnswer.model_validate(data)
        return None

    # -----------------------------------------------------------------------
    # API key 캐시 (DB 부하 완화)
    # -----------------------------------------------------------------------

    async def cache_api_key(self, client_key: str, payload_json: str, ttl: int = 300) -> None:
        await self.redis.set(k_apikey_cache(client_key), payload_json, ex=ttl)

    async def get_cached_api_key(self, client_key: str) -> str | None:
        return await self.redis.get(k_apikey_cache(client_key))

    async def invalidate_api_key(self, client_key: str) -> None:
        """API key revoke / rotate 시 즉시 캐시 제거."""
        await self.redis.delete(k_apikey_cache(client_key))

    # -----------------------------------------------------------------------
    # captcha_token 페이로드 저장 (#43 siteverify 가 사용)
    # -----------------------------------------------------------------------

    async def save_token(self, challenge_id: str, payload: dict, ttl: int = 120) -> None:
        """
        /v1/challenges/{cid}/answer 가 성공하면 호출.
        siteverify 시 GETDEL 로 단 1회 회수 → 재사용 차단.
        payload 예: {"verdict": "human", "confidence": 0.5, "hostname": "...", "ts": "..."}
        """
        import json
        await self.redis.set(k_token(challenge_id), json.dumps(payload), ex=ttl)

    async def consume_token(self, challenge_id: str) -> dict | None:
        """
        siteverify 가 호출. atomic GETDEL.
        존재 X / 이미 소비됨 → None (사용 측이 timeout-or-duplicate 로 처리).
        """
        import json
        raw = await self.redis.getdel(k_token(challenge_id))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            logger.exception("failed to parse stored token for %s", challenge_id)
            return None

    # -----------------------------------------------------------------------
    # Rate limit primitive (#45 가 사용)
    # -----------------------------------------------------------------------

    async def incr_failure(self, ip: str, window_seconds: int) -> int:
        """
        실패 1회 누적. 동적 난이도 상향용 (#45).
        반환값 = 창 안의 누적 실패 횟수.
        """
        return await self.incr_rate_counter(k_fail_ip(ip), window_seconds)

    async def get_failure_count(self, ip: str) -> int:
        """현재 IP 의 누적 실패 횟수. 만료되었으면 0."""
        raw = await self.redis.get(k_fail_ip(ip))
        if raw is None:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    async def incr_rate_counter(self, key: str, window_seconds: int) -> int:
        """
        Redis INCR + EXPIRE pipeline. atomic 하게 카운터를 1 증가시키고
        만약 이번이 첫 증가라면 TTL 을 설정.

        반환값 = 현재 누적 호출 수.
        """
        # pipeline 으로 묶어 round-trip 절약. transaction=False : INCR / EXPIRE 는
        # 각각 atomic 하므로 transaction 없이 OK (성능 ↑).
        async with self.redis.pipeline(transaction=False) as pipe:
            await pipe.incr(key)
            await pipe.expire(key, window_seconds)
            count, _ = await pipe.execute()
        return int(count)
