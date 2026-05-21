"""
FastAPI Dependencies
====================
WBS #43: HTTP 핸들러가 주입받는 공통 의존성.

- get_db / get_redis / get_store : 인프라 자원
- verify_client_key : X-Captcha-Client-Key 헤더 검증
- verify_origin    : Origin 헤더가 tenant 의 allowed_origins 에 있는지 검증
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import redis.asyncio as redis_async
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.policy import (
    DEFAULT_PER_IP_LIMIT_PER_MIN,
    RATE_LIMIT_WINDOW_SEC,
    is_rate_limited,
)
from app.cache.challenge_store import (
    ChallengeStore,
    k_rate_apikey,
    k_rate_ip,
)
from app.cache.redis_client import get_redis_client
from app.core.config import get_settings
from app.db.models import AllowedOrigin, ApiKey, TenantSettings
from app.db.session import get_sessionmaker


# ---------------------------------------------------------------------------
# 인프라
# ---------------------------------------------------------------------------

async def get_db() -> AsyncIterator[AsyncSession]:
    """async DB 세션. 요청 단위 lifecycle."""
    sm = get_sessionmaker()
    async with sm() as session:
        yield session


def get_redis() -> redis_async.Redis:
    return get_redis_client()


def get_store(redis: redis_async.Redis = Depends(get_redis)) -> ChallengeStore:
    return ChallengeStore(redis)


# ---------------------------------------------------------------------------
# 인증
# ---------------------------------------------------------------------------

async def verify_client_key(
    x_captcha_client_key: str = Header(..., alias="X-Captcha-Client-Key"),
    db: AsyncSession = Depends(get_db),
) -> ApiKey:
    """
    헤더로 들어온 client_key 가 활성 API key 인지 확인.
    revoked 된 키는 거부.
    """
    stmt = select(ApiKey).where(
        ApiKey.client_key == x_captcha_client_key,
        ApiKey.revoked_at.is_(None),
    )
    api_key = (await db.execute(stmt)).scalar_one_or_none()
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_client_key", "message": "Unknown or revoked client key."},
        )
    return api_key


async def verify_origin(
    request: Request,
    api_key: ApiKey = Depends(verify_client_key),
    db: AsyncSession = Depends(get_db),
) -> ApiKey:
    """
    Origin 헤더가 tenant 의 allowed_origins 화이트리스트에 있는지 확인.
    Origin 이 없는 요청 (서버-서버, curl 테스트 등) 은 통과.
    프로덕션에서는 정책에 따라 강제할 수 있음.
    """
    origin = request.headers.get("origin")
    if not origin:
        return api_key

    stmt = select(AllowedOrigin).where(
        AllowedOrigin.tenant_id == api_key.tenant_id,
        AllowedOrigin.origin == origin,
    )
    if (await db.execute(stmt)).scalar_one_or_none() is None:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "origin_not_allowed",
                "message": f"Origin {origin} is not in this tenant's allowed list.",
            },
        )
    return api_key


# ---------------------------------------------------------------------------
# Rate Limit (#45)
# ---------------------------------------------------------------------------

async def enforce_rate_limit(
    request: Request,
    api_key: ApiKey = Depends(verify_origin),
    db: AsyncSession = Depends(get_db),
    store: ChallengeStore = Depends(get_store),
) -> ApiKey:
    """
    요청 단위 rate limit. IP 와 API key 양쪽을 1분 창으로 카운트.
    한도 초과 시 429 + Retry-After 헤더.

    한도 산정:
    - API key 한도 = tenant_settings.rate_limit_per_min (없으면 시스템 기본값)
    - IP 한도 = DEFAULT_PER_IP_LIMIT_PER_MIN (단일 IP 폭주 차단)
    """
    settings = get_settings()
    ip = request.client.host if request.client else "unknown"

    # 카운터 두 개를 동시에 증가 (round-trip 한 번에)
    ip_count = await store.incr_rate_counter(
        k_rate_ip(ip, "1m"), RATE_LIMIT_WINDOW_SEC
    )
    apikey_count = await store.incr_rate_counter(
        k_rate_apikey(api_key.client_key, "1m"), RATE_LIMIT_WINDOW_SEC
    )

    # tenant 별 한도 조회. 없으면 시스템 기본.
    ts = (
        await db.execute(
            select(TenantSettings).where(TenantSettings.tenant_id == api_key.tenant_id)
        )
    ).scalar_one_or_none()
    apikey_limit = ts.rate_limit_per_min if ts else settings.default_rate_limit_per_min

    if is_rate_limited(ip_count, DEFAULT_PER_IP_LIMIT_PER_MIN):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "rate_limit_exceeded",
                "message": "Too many requests from this IP. Please retry later.",
                "scope": "ip",
            },
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SEC)},
        )
    if is_rate_limited(apikey_count, apikey_limit):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "rate_limit_exceeded",
                "message": "Too many requests for this API key. Please retry later.",
                "scope": "api_key",
            },
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SEC)},
        )

    return api_key
