-- =====================================================================
-- Captcha Engine PostgreSQL Schema (WBS #42)
-- =====================================================================
-- PostgreSQL 14+ 기준. uuid-ossp 또는 pgcrypto 의 gen_random_uuid() 사용.
-- 모든 timestamp 는 TIMESTAMPTZ (UTC 저장).
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid()

-- ---------------------------------------------------------------------
-- 1. tenants : 서비스를 구매한 기업 고객 (멀티테넌시의 루트)
-- ---------------------------------------------------------------------
CREATE TABLE tenants (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(120)  NOT NULL,
    billing_plan    VARCHAR(32)   NOT NULL DEFAULT 'free'
                    CHECK (billing_plan IN ('free', 'standard', 'pro', 'enterprise')),
    is_active       BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------
-- 2. tenant_users : 대시보드에 로그인하는 실제 사용자 (Firebase 연결)
-- ---------------------------------------------------------------------
CREATE TABLE tenant_users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID          NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    firebase_uid    VARCHAR(128)  NOT NULL UNIQUE,
    email           VARCHAR(255)  NOT NULL,
    role            VARCHAR(16)   NOT NULL DEFAULT 'admin'
                    CHECK (role IN ('owner', 'admin', 'viewer')),
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_tenant_users_tenant_id ON tenant_users(tenant_id);

-- ---------------------------------------------------------------------
-- 3. api_keys : 기업이 발급받는 client_key + secret_key 쌍
-- ---------------------------------------------------------------------
-- client_key : public. 프론트엔드 위젯에 임베드. tenant 식별용.
-- secret_key : confidential. 해시만 저장. 발급 시점에만 평문 노출.
-- ---------------------------------------------------------------------
CREATE TABLE api_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID          NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name            VARCHAR(80)   NOT NULL,
    client_key      VARCHAR(64)   NOT NULL UNIQUE,
    secret_hash     VARCHAR(128)  NOT NULL,
    last_used_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    revoked_at      TIMESTAMPTZ
);
CREATE INDEX idx_api_keys_tenant_id ON api_keys(tenant_id);
-- client_key UNIQUE 제약은 자동으로 인덱스 생성. 별도 인덱스 불필요.

-- ---------------------------------------------------------------------
-- 4. allowed_origins : 캡챠를 임베드할 수 있는 도메인 화이트리스트
-- ---------------------------------------------------------------------
CREATE TABLE allowed_origins (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID          NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    origin          VARCHAR(255)  NOT NULL,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, origin)
);
CREATE INDEX idx_allowed_origins_tenant_id ON allowed_origins(tenant_id);

-- ---------------------------------------------------------------------
-- 5. tenant_settings : 기획서 2.2.2 "캡챠 커스터마이징"
-- ---------------------------------------------------------------------
CREATE TABLE tenant_settings (
    tenant_id           UUID PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    default_difficulty  VARCHAR(16)   NOT NULL DEFAULT 'medium'
                        CHECK (default_difficulty IN ('easy', 'medium', 'hard')),
    enabled_kinds       JSONB         NOT NULL DEFAULT '["flashlight"]'::jsonb,
    max_attempts        INTEGER       NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    rate_limit_per_min  INTEGER       NOT NULL DEFAULT 60 CHECK (rate_limit_per_min > 0),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------
-- 6. challenges : 발급된 챌린지 로그
-- ---------------------------------------------------------------------
-- PK 는 #41 generator 가 만든 token_urlsafe 문자열을 그대로 사용.
-- VARCHAR(32) 로 16바이트 token_urlsafe(22글자) + 마진.
-- ---------------------------------------------------------------------
CREATE TABLE challenges (
    id                  VARCHAR(32) PRIMARY KEY,
    tenant_id           UUID          NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    api_key_id          UUID          NOT NULL REFERENCES api_keys(id),
    kind                VARCHAR(32)   NOT NULL,
    variant             VARCHAR(32),
    difficulty          VARCHAR(16)   NOT NULL,
    issued_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ   NOT NULL,
    requester_ip        INET,
    requester_origin    VARCHAR(255)
);
-- 대시보드: 시간대별 트래픽, 캡챠 종류별 발급량
CREATE INDEX idx_challenges_tenant_issued ON challenges(tenant_id, issued_at DESC);
CREATE INDEX idx_challenges_kind ON challenges(tenant_id, kind, issued_at DESC);

-- ---------------------------------------------------------------------
-- 7. verifications : 사용자가 답을 제출한 결과 (대시보드 통계의 원천)
-- ---------------------------------------------------------------------
CREATE TABLE verifications (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    challenge_id        VARCHAR(32)   NOT NULL REFERENCES challenges(id) ON DELETE CASCADE,
    tenant_id           UUID          NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    success             BOOLEAN       NOT NULL,
    verdict             VARCHAR(16)   NOT NULL
                        CHECK (verdict IN ('human', 'bot', 'uncertain')),
    confidence          NUMERIC(4, 3) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    time_taken_ms       INTEGER,
    ai_model_score      NUMERIC(4, 3),
    behavioral_summary  JSONB,
    requester_ip        INET,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
-- 성공률 그래프
CREATE INDEX idx_verifications_tenant_created ON verifications(tenant_id, created_at DESC);
-- challenge → 검증 결과 역추적
CREATE INDEX idx_verifications_challenge ON verifications(challenge_id);
-- 성공/실패율 GROUP BY 가속
CREATE INDEX idx_verifications_tenant_success ON verifications(tenant_id, success, created_at DESC);

-- ---------------------------------------------------------------------
-- 향후 작업 (이번 단계 범위 외):
-- - verifications 의 created_at 월별 파티셔닝 (트래픽 100만건/월 이상일 때)
-- - 시간대별 통계를 위한 materialized view (대시보드 응답 속도 향상)
-- - behavioral_summary JSONB 의 GIN 인덱스 (특정 행동 패턴 검색)
-- ---------------------------------------------------------------------
