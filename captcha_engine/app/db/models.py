"""
SQLAlchemy 2.0 ORM Models
==========================
WBS #42: PostgreSQL 매핑 레이어. db/schema.sql 과 1:1 대응.

스타일: SQLAlchemy 2.0 의 Mapped[] / mapped_column() 신문법.
PostgreSQL 전용 타입(UUID, INET, JSONB) 사용.

이 파일은 schema.sql 의 단순한 미러가 아니라 애플리케이션 레이어가
타입 힌트와 함께 사용할 수 있도록 매핑하는 게 핵심 가치임.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """모든 모델이 상속할 베이스. SQLAlchemy 2.0 권장 방식."""
    pass


# ---------------------------------------------------------------------------
# 1. Tenant
# ---------------------------------------------------------------------------

class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    billing_plan: Mapped[str] = mapped_column(
        String(32), nullable=False, default="free"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        CheckConstraint(
            "billing_plan IN ('free', 'standard', 'pro', 'enterprise')",
            name="ck_tenants_billing_plan",
        ),
    )

    # 관계 (선택 사항: 자주 같이 조회되면 lazy="select" 가 기본)
    users: Mapped[list["TenantUser"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    settings: Mapped["TenantSettings | None"] = relationship(
        back_populates="tenant", uselist=False, cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# 2. TenantUser
# ---------------------------------------------------------------------------

class TenantUser(Base):
    __tablename__ = "tenant_users"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    firebase_uid: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        CheckConstraint("role IN ('owner', 'admin', 'viewer')", name="ck_tenant_users_role"),
        Index("idx_tenant_users_tenant_id", "tenant_id"),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="users")


# ---------------------------------------------------------------------------
# 3. ApiKey
# ---------------------------------------------------------------------------

class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    client_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # secret_key 는 절대 평문 저장 X. SHA-256 hash (hex 64자) 또는 argon2.
    secret_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_api_keys_tenant_id", "tenant_id"),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")


# ---------------------------------------------------------------------------
# 4. AllowedOrigin
# ---------------------------------------------------------------------------

class AllowedOrigin(Base):
    __tablename__ = "allowed_origins"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    origin: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("tenant_id", "origin", name="uq_allowed_origins_tenant_origin"),
        Index("idx_allowed_origins_tenant_id", "tenant_id"),
    )


# ---------------------------------------------------------------------------
# 5. TenantSettings
# ---------------------------------------------------------------------------

class TenantSettings(Base):
    __tablename__ = "tenant_settings"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        primary_key=True,
    )
    default_difficulty: Mapped[str] = mapped_column(String(16), nullable=False, default="medium")
    enabled_kinds: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=lambda: ["flashlight"])
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        CheckConstraint("default_difficulty IN ('easy', 'medium', 'hard')", name="ck_tenant_settings_difficulty"),
        CheckConstraint("max_attempts > 0", name="ck_tenant_settings_max_attempts"),
        CheckConstraint("rate_limit_per_min > 0", name="ck_tenant_settings_rate_limit"),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="settings")


# ---------------------------------------------------------------------------
# 6. Challenge (audit log)
# ---------------------------------------------------------------------------

class Challenge(Base):
    """
    발급된 챌린지의 영속 로그.
    PK = #41 의 token_urlsafe 문자열 그대로. UUID 가 아님에 주의.
    """
    __tablename__ = "challenges"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    variant: Mapped[str | None] = mapped_column(String(32))
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    requester_ip: Mapped[Any | None] = mapped_column(INET)
    requester_origin: Mapped[str | None] = mapped_column(String(255))

    __table_args__ = (
        Index("idx_challenges_tenant_issued", "tenant_id", "issued_at"),
        Index("idx_challenges_kind", "tenant_id", "kind", "issued_at"),
    )


# ---------------------------------------------------------------------------
# 7. Verification (대시보드 통계의 원천)
# ---------------------------------------------------------------------------

class Verification(Base):
    __tablename__ = "verifications"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    challenge_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("challenges.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)  # 'human' | 'bot' | 'uncertain'
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    time_taken_ms: Mapped[int | None] = mapped_column(Integer)
    ai_model_score: Mapped[float | None] = mapped_column(Numeric(4, 3))
    # 행동 분석 요약 (raw 마우스 궤적 X. 집계된 피처만 저장 — 프라이버시).
    behavioral_summary: Mapped[dict | None] = mapped_column(JSONB)
    requester_ip: Mapped[Any | None] = mapped_column(INET)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        CheckConstraint("verdict IN ('human', 'bot', 'uncertain')", name="ck_verifications_verdict"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_verifications_confidence"),
        Index("idx_verifications_tenant_created", "tenant_id", "created_at"),
        Index("idx_verifications_challenge", "challenge_id"),
        Index("idx_verifications_tenant_success", "tenant_id", "success", "created_at"),
    )
