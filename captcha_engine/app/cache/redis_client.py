"""
Redis Async Client Factory
===========================
WBS #43: Redis 연결 풀 관리.

- main.py 의 lifespan 이 init_redis / close_redis 호출
- API 핸들러는 deps.get_redis() 의존성으로 클라이언트 주입받음

redis-py 5.0+ 의 redis.asyncio API 사용. close 는 aclose() 로 호출.
"""

from __future__ import annotations

import redis.asyncio as redis_async

from app.core.config import get_settings


_client: redis_async.Redis | None = None


async def init_redis() -> redis_async.Redis:
    """앱 시작 시 호출. ping 으로 연결 가능 여부도 확인."""
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    _client = redis_async.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=True,  # bytes 대신 str 반환 (JSON 파싱 편의)
    )
    # 연결 검증. 실패 시 RuntimeError 가 lifespan 까지 전파되어 앱 시작 차단.
    await _client.ping()
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def get_redis_client() -> redis_async.Redis:
    if _client is None:
        raise RuntimeError(
            "Redis client not initialized. Did the lifespan handler run?"
        )
    return _client
