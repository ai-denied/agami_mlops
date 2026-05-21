"""
Application Settings
====================
WBS #42: 환경변수 기반 설정.

Pydantic v2 의 pydantic-settings 패키지 사용.
운영/개발 환경 분리는 환경변수로 (.env 또는 K8s Secret).

설치: pip install pydantic-settings
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # 일반
    # -----------------------------------------------------------------------
    app_env: str = Field(default="local", description="local | dev | prod")
    log_level: str = Field(default="INFO")

    # -----------------------------------------------------------------------
    # PostgreSQL
    # -----------------------------------------------------------------------
    # SQLAlchemy 비동기 드라이버: postgresql+asyncpg://...
    database_url: str = Field(
        default="postgresql+asyncpg://captcha:captcha@localhost:5432/captcha"
    )
    db_pool_size: int = Field(default=10)
    db_max_overflow: int = Field(default=10)

    # -----------------------------------------------------------------------
    # Redis
    # -----------------------------------------------------------------------
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_max_connections: int = Field(default=50)

    # -----------------------------------------------------------------------
    # 보안
    # -----------------------------------------------------------------------
    # API secret_key 검증용 HMAC pepper. Tenant-independent. 주기적 회전 권장.
    # 절대 git 에 커밋 X. K8s Secret / KMS 사용.
    api_key_hmac_pepper: str = Field(default="CHANGE_ME_IN_PRODUCTION")

    # captcha_token (사용자에게 발급되는 1회용 토큰) 의 HMAC 서명 키.
    # api_key_hmac_pepper 와 별도로 분리 (책임 분리, 회전 주기 다를 수 있음).
    captcha_token_secret: str = Field(default="CHANGE_ME_TOKEN_SECRET")

    # Firebase Admin SDK credentials (service account JSON 경로 또는 inline JSON)
    firebase_credentials_path: str | None = Field(default=None)

    # -----------------------------------------------------------------------
    # 캡챠 동작 정책
    # -----------------------------------------------------------------------
    # Tenant 가 별도 설정을 안 했을 때의 기본값
    default_difficulty: str = Field(default="medium")
    default_rate_limit_per_min: int = Field(default=60)

    # -----------------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------------
    # 콤마로 구분된 origin 문자열. K8s ConfigMap 에서 한 줄로 주입하기 쉽게 string.
    # 예: "http://localhost:5173,http://210.109.53.140"
    cors_origins: str = Field(default="http://localhost:5173")

    @property
    def cors_origins_list(self) -> list[str]:
        """CORSMiddleware 의 allow_origins 에 직접 넘길 수 있는 list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """싱글턴 캐시. FastAPI Depends 에서 사용."""
    return Settings()
