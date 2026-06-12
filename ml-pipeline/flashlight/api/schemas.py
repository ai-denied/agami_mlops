#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flashlight 추론 API Pydantic 요청/응답 스키마
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class TrajectoryPoint(BaseModel):
    x: float
    y: float
    t: float  # ms timestamp (e.g. Date.now())


class PredictRequest(BaseModel):
    trajectory: Annotated[List[TrajectoryPoint], Field(min_length=1)]
    coordinate_mode: Literal["normalized", "pixel"] = "normalized"
    canvas_width: Optional[float] = None
    canvas_height: Optional[float] = None

    @model_validator(mode="after")
    def _check_canvas(self) -> "PredictRequest":
        if self.coordinate_mode == "normalized" and (
            self.canvas_width is None or self.canvas_height is None
        ):
            raise ValueError(
                "coordinate_mode='normalized' 사용 시 canvas_width, canvas_height가 필요합니다."
            )
        return self


class PredictResponse(BaseModel):
    bot_risk_score: float
    risk_band: str  # "low_risk" | "suspicious" | "high_risk"
    low_risk_threshold: float
    high_risk_threshold: float
    seq_len: int  # 실제 처리된 trajectory 포인트 수


class DecideRequest(BaseModel):
    scores: Annotated[List[float], Field(min_length=1, max_length=10)]


class DecideResponse(BaseModel):
    final_decision: str  # "allow" | "block"
    is_bot: bool
    scores: List[float]
    total_score: float
    avg_score: float
    max_score: float
    min_score: float
    suspicious_count: int
    high_risk_count: int
    low_risk_threshold: float
    high_risk_threshold: float
    block_total_score: float
    internal_decision: str  # "allow" | "challenge_again" | "block"


class HealthResponse(BaseModel):
    status: str  # "ok" | "model_not_loaded"
    model_loaded: bool
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
