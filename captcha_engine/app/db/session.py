"""
Async DB Engine + Session Factory
==================================
WBS #43: PostgreSQL 비동기 연결 관리.

- main.py 의 lifespan 이 init_engine / dispose_engine 호출
- API 핸들러는 deps.get_db() 의존성으로 세션 주입받음

asyncpg 드라이버 사용 (config.py 의 database_url 이 'postgresql+asyncpg://').
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine() -> AsyncEngine:
    """앱 시작 시 1회 호출."""
    global _engine, _sessionmaker
    if _engine is not None:
        return _engine

    settings = get_settings()
    _engine = create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,  # stale connection 자동 감지
    )
    _sessionmaker = async_sessionmaker(
        _engine,
        expire_on_commit=False,  # commit 후에도 객체 속성 접근 가능 (FastAPI 응답 직렬화 용이)
    )
    return _engine


async def dispose_engine() -> None:
    """앱 종료 시 호출."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError(
            "DB engine not initialized. Did the lifespan handler run?"
        )
    return _sessionmaker
