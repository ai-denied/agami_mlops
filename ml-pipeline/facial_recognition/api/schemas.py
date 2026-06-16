"""
얼굴 활성도(liveness) 추론 API Pydantic 스키마.

입력 피처 순서 (20개, selected_features):
  ear, mar, smile_w,
  nose_x, nose_y, cx, cy,
  roll, yaw, pitch,
  nose_dx, nose_dy, center_dx, center_dy, nose_speed,
  ear_velocity, mar_velocity, yaw_velocity, pitch_velocity, roll_velocity
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SEQ_LEN    = 16
N_FEATURES = 20


# ── 추론 요청/응답 ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    단일 얼굴 시퀀스 추론 요청.

    x_seq: 16프레임 × 20피처 행렬 (raw 값, 스케일러는 ONNX 내장)
    seq_length: 실제 유효 프레임 수. 미제공 시 16으로 간주.
    """
    x_seq: Annotated[
        List[Annotated[List[float], Field(min_length=N_FEATURES, max_length=N_FEATURES)]],
        Field(min_length=SEQ_LEN, max_length=SEQ_LEN),
    ]
    seq_length: Optional[int] = Field(default=None, ge=1, le=SEQ_LEN)

    @model_validator(mode="after")
    def _set_seq_length(self) -> "PredictRequest":
        if self.seq_length is None:
            self.seq_length = SEQ_LEN
        return self


class PredictResponse(BaseModel):
    spoof_score: float
    risk_band: Literal["real_safe", "suspicious", "spoof_detected"]
    is_spoof: bool
    low_spoof_threshold: float
    high_spoof_threshold: float
    seq_length: int


# ── 3회 라운드 판정 ────────────────────────────────────────────────────────────

class MissionRoundInput(BaseModel):
    """CAPTCHA 1회 라운드 결과."""
    round_id: Annotated[int, Field(ge=1, le=3)]
    mission_type: Literal["face", "hand"]
    spoof_score: Annotated[float, Field(ge=0.0, le=1.0)]
    mission_pass: bool
    face_detected: bool = True
    timeout: bool = False
    mission_name: str = ""
    hand_detected: bool = False
    detail: str = ""


class DecideRequest(BaseModel):
    """3회 라운드 결과를 받아 CAPTCHA 최종 판정을 요청한다."""
    rounds: Annotated[List[MissionRoundInput], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def _check_round_ids(self) -> "DecideRequest":
        ids = sorted(r.round_id for r in self.rounds)
        if ids != [1, 2, 3]:
            raise ValueError("rounds의 round_id는 1, 2, 3이어야 합니다.")
        return self


class DecideResponse(BaseModel):
    decision: Literal["PASS", "RETRY", "FAIL"]
    reason: str
    total_risk: float
    avg_spoof_score: float
    failed_mission_count: int
    failed_face_count: int
    timeout_count: int


# ── 헬스체크 ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str           # "ok" | "model_not_loaded"
    model_loaded: bool
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
