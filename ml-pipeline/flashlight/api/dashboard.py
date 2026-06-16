#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flashlight Dashboard API

precompute_dashboard.py 가 생성한 dashboard_cache.json 을 읽어
집계 지표를 제공한다.

엔드포인트:
  GET /api/v1/dashboard/summary          — 전체 통계 + 파이 차트 데이터
  GET /api/v1/dashboard/attack_types     — 공격 유형 Top N
  GET /api/v1/dashboard/risk_distribution — risk_band 분포
  GET /api/v1/dashboard/sessions         — 세션 목록 (페이지네이션 + 필터)
  POST /api/v1/dashboard/reload          — 캐시 재로드
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

_CACHE_PATH = Path(
    os.environ.get(
        "DASHBOARD_CACHE_PATH",
        # K8s: model-store PVC 내 위치 (FLASHLIGHT_MODEL_DIR 의 상위)
        # 로컬: /workspace/data/flashlight/dashboard_cache.json 으로 오버라이드
        os.path.join(
            os.path.dirname(os.environ.get("FLASHLIGHT_MODEL_DIR", "/workspace/ml-pipeline/model-store/flashlight/current")),
            "dashboard_cache.json",
        ),
    )
)

_cache: Optional[Dict[str, Any]] = None
_cache_mtime: float = -1.0

BOT_TYPE_DISPLAY: Dict[str, str] = {
    "grid_search": "Grid Search",
    "known_target": "Known Target",
    "other_search": "AI Vision (GPT-Vision)",
    "random_search": "Random Search",
}

RISK_BAND_DISPLAY: Dict[str, str] = {
    "low_risk": "Safe",
    "suspicious": "Suspicious",
    "high_risk": "Critical",
}


def _load_cache() -> Dict[str, Any]:
    global _cache, _cache_mtime
    if not _CACHE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"dashboard_cache.json 없음: {_CACHE_PATH} "
                "— 먼저 precompute_dashboard.py 를 실행하세요."
            ),
        )
    mtime = _CACHE_PATH.stat().st_mtime
    if _cache is None or mtime != _cache_mtime:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
        _cache_mtime = mtime
    return _cache


# ── Response schemas ───────────────────────────────────────────────────────────

class PieSlice(BaseModel):
    label: str
    count: int
    ratio: float


class SummaryResponse(BaseModel):
    generated_at: str
    model_version: str
    total_sessions: int
    human_total: int
    bot_total: int
    human_pass_rate: float
    human_suspicious_rate: float
    human_block_rate: float
    bot_detect_rate: float
    bot_miss_rate: float
    pie_chart: List[PieSlice]


class AttackTypeItem(BaseModel):
    type_key: str
    display_name: str
    count: int
    blocked: int
    block_rate: float


class AttackTypesResponse(BaseModel):
    total_bot_sessions: int
    top_types: List[AttackTypeItem]


class RiskBandItem(BaseModel):
    band: str
    display_name: str
    count: int
    ratio: float


class RiskDistributionResponse(BaseModel):
    total_sessions: int
    bands: List[RiskBandItem]


class SessionItem(BaseModel):
    file: str
    source_type: str
    bot_type: Optional[str]
    label: int
    image_id: Optional[str]
    bot_risk_score: float
    risk_band: str
    risk_band_display: str
    is_blocked: bool


class SessionsResponse(BaseModel):
    total: int
    offset: int
    limit: int
    sessions: List[SessionItem]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/summary",
    response_model=SummaryResponse,
    summary="전체 통계 + 파이 차트 데이터",
)
def get_summary():
    cache = _load_cache()
    s = cache["summary"]
    total = s["total_sessions"]

    sessions = cache["sessions"]
    human_pass = sum(1 for r in sessions if r["source_type"] == "human" and r["risk_band"] == "low_risk")
    human_suspicious = sum(1 for r in sessions if r["source_type"] == "human" and r["risk_band"] == "suspicious")
    human_blocked = sum(1 for r in sessions if r["source_type"] == "human" and r["is_blocked"])
    bot_detected = sum(1 for r in sessions if r["source_type"] == "bot" and r["is_blocked"])
    bot_missed = sum(1 for r in sessions if r["source_type"] == "bot" and not r["is_blocked"])

    pie = [
        PieSlice(label="Human Passed",     count=human_pass,        ratio=round(human_pass / total, 4)),
        PieSlice(label="Human Suspicious", count=human_suspicious,   ratio=round(human_suspicious / total, 4)),
        PieSlice(label="Human Blocked",    count=human_blocked,      ratio=round(human_blocked / total, 4)),
        PieSlice(label="Bot Detected",     count=bot_detected,       ratio=round(bot_detected / total, 4)),
        PieSlice(label="Bot Missed",       count=bot_missed,         ratio=round(bot_missed / total, 4)),
    ]

    return SummaryResponse(
        generated_at=cache["generated_at"],
        model_version=cache["model_version"],
        total_sessions=total,
        human_total=s["human_total"],
        bot_total=s["bot_total"],
        human_pass_rate=s["human_pass_rate"],
        human_suspicious_rate=s["human_suspicious_rate"],
        human_block_rate=s["human_block_rate"],
        bot_detect_rate=s["bot_detect_rate"],
        bot_miss_rate=s["bot_miss_rate"],
        pie_chart=pie,
    )


@router.get(
    "/attack_types",
    response_model=AttackTypesResponse,
    summary="공격 유형 Top N (기본 5)",
)
def get_attack_types(top_n: int = Query(default=5, ge=1, le=20)):
    cache = _load_cache()
    raw_counts: Dict[str, Dict[str, int]] = cache["summary"]["attack_type_counts"]

    items: List[AttackTypeItem] = []
    for key, val in raw_counts.items():
        cnt = val["count"]
        blocked = val["blocked"]
        items.append(
            AttackTypeItem(
                type_key=key,
                display_name=BOT_TYPE_DISPLAY.get(key, key),
                count=cnt,
                blocked=blocked,
                block_rate=round(blocked / cnt, 4) if cnt else 0.0,
            )
        )

    items.sort(key=lambda x: x.count, reverse=True)

    return AttackTypesResponse(
        total_bot_sessions=cache["summary"]["bot_total"],
        top_types=items[:top_n],
    )


@router.get(
    "/risk_distribution",
    response_model=RiskDistributionResponse,
    summary="Safe / Suspicious / Critical 분포",
)
def get_risk_distribution():
    cache = _load_cache()
    counts: Dict[str, int] = cache["summary"]["risk_band_counts"]
    total = cache["summary"]["total_sessions"]

    bands = [
        RiskBandItem(
            band=key,
            display_name=RISK_BAND_DISPLAY.get(key, key),
            count=counts.get(key, 0),
            ratio=round(counts.get(key, 0) / total, 4) if total else 0.0,
        )
        for key in ("low_risk", "suspicious", "high_risk")
    ]

    return RiskDistributionResponse(total_sessions=total, bands=bands)


@router.get(
    "/sessions",
    response_model=SessionsResponse,
    summary="세션 목록 (페이지네이션 + 필터)",
)
def get_sessions(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    source_type: Optional[str] = Query(default=None, description="human | bot"),
    bot_type: Optional[str] = Query(default=None, description="grid_search | known_target | other_search | random_search"),
    risk_band: Optional[str] = Query(default=None, description="low_risk | suspicious | high_risk"),
    is_blocked: Optional[bool] = Query(default=None),
):
    cache = _load_cache()
    sessions = cache["sessions"]

    if source_type is not None:
        sessions = [s for s in sessions if s["source_type"] == source_type]
    if bot_type is not None:
        sessions = [s for s in sessions if s.get("bot_type") == bot_type]
    if risk_band is not None:
        sessions = [s for s in sessions if s["risk_band"] == risk_band]
    if is_blocked is not None:
        sessions = [s for s in sessions if s["is_blocked"] == is_blocked]

    total = len(sessions)
    page = sessions[offset : offset + limit]

    return SessionsResponse(
        total=total,
        offset=offset,
        limit=limit,
        sessions=[
            SessionItem(
                **s,
                risk_band_display=RISK_BAND_DISPLAY.get(s["risk_band"], s["risk_band"]),
            )
            for s in page
        ],
    )


@router.post(
    "/reload",
    summary="dashboard_cache.json 재로드",
)
def reload_cache():
    global _cache
    _cache = None
    _load_cache()
    return {"status": "ok", "message": "캐시 재로드 완료"}
