from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

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


def _get_cache_path() -> Path:
    env_val = os.environ.get("DASHBOARD_CACHE_PATH", "")
    if not env_val:
        raise HTTPException(
            status_code=503,
            detail="DASHBOARD_CACHE_PATH 환경변수가 설정되지 않았습니다.",
        )
    return Path(env_val)


def _load_cache() -> Dict[str, Any]:
    global _cache, _cache_mtime
    cache_path = _get_cache_path()
    if not cache_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"dashboard_cache.json 없음: {cache_path} "
                "— 먼저 precompute_dashboard.py 를 실행하세요."
            ),
        )
    mtime = cache_path.stat().st_mtime
    if _cache is None or mtime != _cache_mtime:
        with open(cache_path, "r", encoding="utf-8") as f:
            _cache = json.load(f)
        _cache_mtime = mtime
    return _cache


def _filter_sessions(sessions: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    if kind == "all":
        return sessions
    return [s for s in sessions if s.get("kind") == kind or s.get("model_type") == kind]


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


class TrafficItem(BaseModel):
    time: str
    success: int
    attack: int


class TrafficResponse(BaseModel):
    traffic: List[TrafficItem]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/summary",
    response_model=SummaryResponse,
    summary="전체 통계 + 파이 차트 데이터",
)
def get_summary(kind: str = Query(default="all")):
    cache = _load_cache()
    sessions = _filter_sessions(cache.get("sessions", []), kind)
    total = len(sessions)

    human_pass = sum(1 for r in sessions if r.get("source_type") == "human" and r.get("risk_band") == "low_risk")
    human_suspicious = sum(1 for r in sessions if r.get("source_type") == "human" and r.get("risk_band") == "suspicious")
    human_blocked = sum(1 for r in sessions if r.get("source_type") == "human" and r.get("is_blocked"))
    bot_detected = sum(1 for r in sessions if r.get("source_type") == "bot" and r.get("is_blocked"))
    bot_missed = sum(1 for r in sessions if r.get("source_type") == "bot" and not r.get("is_blocked"))

    human_total = human_pass + human_suspicious + human_blocked
    bot_total = bot_detected + bot_missed

    pie = [
        PieSlice(label="Human Passed",     count=human_pass,       ratio=human_pass / total if total else 0.0),
        PieSlice(label="Human Suspicious", count=human_suspicious, ratio=human_suspicious / total if total else 0.0),
        PieSlice(label="Human Blocked",    count=human_blocked,    ratio=human_blocked / total if total else 0.0),
        PieSlice(label="Bot Detected",     count=bot_detected,     ratio=bot_detected / total if total else 0.0),
        PieSlice(label="Bot Missed",       count=bot_missed,       ratio=bot_missed / total if total else 0.0),
    ]

    return SummaryResponse(
        generated_at=cache.get("generated_at", ""),
        model_version=cache.get("model_version", ""),
        total_sessions=total,
        human_total=human_total,
        bot_total=bot_total,
        human_pass_rate=human_pass / human_total if human_total else 0.0,
        human_suspicious_rate=human_suspicious / human_total if human_total else 0.0,
        human_block_rate=human_blocked / human_total if human_total else 0.0,
        bot_detect_rate=bot_detected / bot_total if bot_total else 0.0,
        bot_miss_rate=bot_missed / bot_total if bot_total else 0.0,
        pie_chart=pie,
    )


@router.get(
    "/attack_types",
    response_model=AttackTypesResponse,
    summary="공격 유형 Top N (기본 5)",
)
def get_attack_types(top_n: int = Query(default=5, ge=1, le=20), kind: str = Query(default="all")):
    cache = _load_cache()
    sessions = _filter_sessions(cache.get("sessions", []), kind)

    type_counts = {}
    bot_total_count = 0

    for s in sessions:
        if s.get("source_type") == "bot":
            bot_total_count += 1
            b_type = s.get("bot_type", "unknown")
            if b_type not in type_counts:
                type_counts[b_type] = {"count": 0, "blocked": 0}
            type_counts[b_type]["count"] += 1
            if s.get("is_blocked"):
                type_counts[b_type]["blocked"] += 1

    items: List[AttackTypeItem] = []
    for key, val in type_counts.items():
        cnt = val["count"]
        blocked = val["blocked"]
        items.append(
            AttackTypeItem(
                type_key=key,
                display_name=BOT_TYPE_DISPLAY.get(key, key),
                count=cnt,
                blocked=blocked,
                block_rate=blocked / cnt if cnt else 0.0,
            )
        )

    items.sort(key=lambda x: x.count, reverse=True)

    return AttackTypesResponse(
        total_bot_sessions=bot_total_count,
        top_types=items[:top_n],
    )


@router.get(
    "/risk_distribution",
    response_model=RiskDistributionResponse,
    summary="Safe / Suspicious / Critical 분포",
)
def get_risk_distribution(kind: str = Query(default="all")):
    cache = _load_cache()
    sessions = _filter_sessions(cache.get("sessions", []), kind)
    total = len(sessions)

    counts = {"low_risk": 0, "suspicious": 0, "high_risk": 0}
    for s in sessions:
        rb = s.get("risk_band")
        if rb in counts:
            counts[rb] += 1

    bands = [
        RiskBandItem(
            band=key,
            display_name=RISK_BAND_DISPLAY.get(key, key),
            count=counts[key],
            ratio=counts[key] / total if total else 0.0,
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
    kind: str = Query(default="all")
):
    cache = _load_cache()
    sessions = _filter_sessions(cache.get("sessions", []), kind)

    if source_type is not None:
        sessions = [s for s in sessions if s.get("source_type") == source_type]
    if bot_type is not None:
        sessions = [s for s in sessions if s.get("bot_type") == bot_type]
    if risk_band is not None:
        sessions = [s for s in sessions if s.get("risk_band") == risk_band]
    if is_blocked is not None:
        sessions = [s for s in sessions if s.get("is_blocked") == is_blocked]

    total = len(sessions)
    page = sessions[offset : offset + limit]

    return SessionsResponse(
        total=total,
        offset=offset,
        limit=limit,
        sessions=[
            SessionItem(
                **s,
                risk_band_display=RISK_BAND_DISPLAY.get(s.get("risk_band", ""), s.get("risk_band", "")),
            )
            for s in page
        ],
    )


@router.get(
    "/traffic",
    response_model=TrafficResponse,
    summary="시간대별 트래픽 (정상/차단) 통계",
)
def get_traffic(kind: str = Query(default="all")):
    cache = _load_cache()
    sessions = _filter_sessions(cache.get("sessions", []), kind)

    traffic_dict = {}

    for s in sessions:
        filename = s.get("file", "")
        is_blocked = s.get("is_blocked", False)

        match = re.search(r'_(\d{13})\.json', filename)
        if match:
            ts_ms = int(match.group(1))
            dt = datetime.fromtimestamp(ts_ms / 1000.0)
            hour_str = dt.strftime('%H:00')
        else:
            hour_str = "00:00"

        if hour_str not in traffic_dict:
            traffic_dict[hour_str] = {"time": hour_str, "success": 0, "attack": 0}

        if is_blocked:
            traffic_dict[hour_str]["attack"] += 1
        else:
            traffic_dict[hour_str]["success"] += 1

    sorted_traffic = sorted(list(traffic_dict.values()), key=lambda x: x["time"])

    return TrafficResponse(traffic=sorted_traffic)


@router.post(
    "/reload",
    summary="dashboard_cache.json 재로드",
)
def reload_cache():
    global _cache
    _cache = None
    _load_cache()
    return {"status": "ok", "message": "캐시 재로드 완료"}
