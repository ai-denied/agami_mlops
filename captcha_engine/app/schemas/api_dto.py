"""
API DTO (Data Transfer Objects)
================================
WBS #43: HTTP 요청/응답 Pydantic 모델.

도메인 모델(challenge_types.py)과 분리. HTTP 레이어 전용 스키마.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.captcha.challenge_types import ChallengeKind, Difficulty

__all__ = [
    "IssueChallengeRequest",
    "TrajectorySummary",
    "BehavioralData",
    "FlashlightSubmission",
    "SubmitAnswerRequest",
    "SubmitAnswerResponse",
    "FlashlightSubmitResponse",
    "SiteVerifyResponse",
]


# ---------------------------------------------------------------------------
# POST /v1/challenges
# ---------------------------------------------------------------------------

class IssueChallengeRequest(BaseModel):
    kind: ChallengeKind = ChallengeKind.FLASHLIGHT
    difficulty: Difficulty | None = None  # None → tenant 기본값 사용


# ---------------------------------------------------------------------------
# POST /v1/challenges/{cid}/answer
# ---------------------------------------------------------------------------

class TrajectorySummary(BaseModel):
    """프론트엔드가 집계한 마우스 궤적 요약. raw 좌표열은 절대 보내지 않음."""
    total_distance: float | None = None
    direction_changes: int | None = None
    avg_speed: float | None = None


class BehavioralData(BaseModel):
    trajectory_summary: TrajectorySummary | None = None
    time_taken_ms: int | None = Field(None, ge=0, le=600_000)


class FlashlightSubmission(BaseModel):
    """손전등 캡챠 1챌린지(3장 묶음)의 1장 결과."""
    index: int = Field(..., ge=0, le=2)
    click_x: float = Field(..., ge=0.0, le=1.0)
    click_y: float = Field(..., ge=0.0, le=1.0)
    trajectory: list[dict] = Field(
        default_factory=list,
        description="이 그림에서의 raw 마우스 궤적 [{x,y,t}, ...]. 길이<2면 모델 평가 스킵.",
    )


class SubmitAnswerRequest(BaseModel):
    """
    캡챠 종류별로 사용 필드가 다름. 서버에서 challenge.kind 로 분기 후 검증.
    - flashlight:        flashlight_submissions 필수 (길이 3)
    - face_mission:      completed_instructions 필수
    - context_inference: submitted_answers 필수 (N문제 시퀀스 정답 리스트)
    """
    # 손전등용 — 1챌린지=3장 묶음 결과. 길이 3, index 셋트 {0,1,2}.
    flashlight_submissions: list[FlashlightSubmission] | None = Field(
        default=None,
        description="손전등 캡챠 1챌린지(3장 묶음) 결과. 길이 3 필수.",
    )
    # [DEPRECATED] 단일 챌린지 시절 필드. 새 클라이언트는 flashlight_submissions 사용.
    click_x: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="[DEPRECATED] 단일 챌린지 시절 필드. flashlight_submissions 사용.",
    )
    click_y: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="[DEPRECATED] 단일 챌린지 시절 필드. flashlight_submissions 사용.",
    )
    # [DEPRECATED] session_id 기반 누적 정책 폐기됨.
    session_id: str | None = Field(
        default=None,
        description="[DEPRECATED] session_id 기반 누적 정책 폐기됨. 사용 안 함.",
    )
    # [DEPRECATED] flashlight_submissions[i].trajectory 로 이동.
    trajectory: list[dict] | None = Field(
        default=None,
        description="[DEPRECATED] flashlight_submissions[i].trajectory 로 이동.",
    )
    # 안면 미션용 — 사용자가 수행 완료했다고 보고한 지시 타입 순서
    completed_instructions: list[str] | None = None
    # 감정 맥락 추론용 — N문제 시퀀스에 대한 정답 (출제 index 순서대로)
    submitted_answers: list[str] | None = Field(
        default=None,
        description="context_inference 캡챠 N문제 시퀀스에 대한 정답 리스트.",
    )
    # [DEPRECATED] 단일 문제 시절 필드. 신규 클라이언트는 submitted_answers 사용.
    # 하위호환 유지를 위해 필드만 남겨두고 실제 검증에는 사용하지 않음.
    selected_emotion: str | None = Field(
        default=None,
        description="[DEPRECATED] 단일 문제 시절 필드. submitted_answers 사용 권장.",
    )
    # 공통 행동 분석 (손전등은 마우스 궤적 / 안면은 향후 MediaPipe summary)
    behavioral_data: BehavioralData | None = None
    face_behavioral_data: dict | None = Field(
        default=None,
        description="MediaPipe summary 등 안면 캡챠 전용 추가 필드. 현재는 자유 dict.",
    )


class SubmitAnswerResponse(BaseModel):
    captcha_token: str
    expires_in: int = Field(..., description="토큰 유효기간 (초)")


class FlashlightSubmitResponse(BaseModel):
    """손전등 캡챠 전용 응답 (1챌린지 = 3장 묶음 평가).

    decision:
      - 'allow'  → 좌표 2/3 이상 AND 모델 high_risk 0회. captcha_token 발급.
      - 'block'  → 좌표 부족 OR 모델 high_risk 1회 이상. 토큰 미발급.
    """
    decision: Literal["allow", "block"]
    captcha_token: str | None = None
    expires_in: int | None = None
    # 관측/디버그용 — 3장 각각의 결과
    coord_hits: list[bool] | None = None
    scores: list[float] | None = None
    risk_bands: list[str] | None = None


# ---------------------------------------------------------------------------
# POST /v1/siteverify (form-encoded 입력은 deps 에서 처리, 응답만 여기)
# ---------------------------------------------------------------------------

class SiteVerifyResponse(BaseModel):
    """
    reCAPTCHA / hCaptcha 호환 응답 형태.
    실패 시에도 HTTP 200, success=false 로 통신.
    """
    success: bool
    verdict: Literal["human", "bot", "uncertain"] | None = None
    confidence: float | None = None
    challenge_ts: datetime | None = None
    hostname: str | None = None
    error_codes: list[str] = Field(default_factory=list)
