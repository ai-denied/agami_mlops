"""Pydantic 요청/응답 스키마.

보안 원칙:
  - ChallengeResponse: correct_label, security_grade, attack_proxy_score 일절 미포함
  - AttemptResponse:   is_correct, retry_allowed 만 반환 (점수·정답 미노출)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── 요청 ─────────────────────────────────────────────────────────────────────

class ChallengeRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="클라이언트가 생성하는 불투명 세션 ID",
    )


class AttemptRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=128)
    challenge_id: str = Field(..., description="challenge 응답에서 받은 UUID")
    selected_label: str = Field(..., description="사용자가 선택한 감정 레이블")
    solve_time_ms: int = Field(..., ge=0, le=600_000)
    retry_count: int = Field(0, ge=0, le=10)
    # 선택: 해시된 식별자 (원본 IP/UA 저장 금지)
    user_agent_hash: Optional[str] = Field(None, max_length=64)
    ip_hash: Optional[str] = Field(None, max_length=64)


# ── 응답 ─────────────────────────────────────────────────────────────────────

class ChallengeResponse(BaseModel):
    challenge_id: str
    image_url: str
    choices: list[str]    # 4지선다, 셔플된 순서
    expires_at: datetime  # ISO 8601 UTC
    # ⛔ 미포함: correct_label, final_emotion, security_grade, attacker_proxy_score


class AttemptResponse(BaseModel):
    is_correct: bool
    retry_allowed: bool
    # ⛔ 미포함: points, correct_label, score, 내부 보안 지표


class HealthResponse(BaseModel):
    status: str            # "ok" | "degraded" | "unavailable"
    pool_loaded: bool
    problem_count: int
    version: str
    pool_loaded_at: Optional[str] = None
