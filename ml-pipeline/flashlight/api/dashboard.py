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
        PieSlice(label="Human Passed",     count=human_pass,        ratio=round(human_pass / total, 4) if total else 0),
        PieSlice(label="Human Suspicious", count=human_suspicious,   ratio=round(human_suspicious / total, 4) if total else 0),
        PieSlice(label="Human Blocked",    count=human_blocked,      ratio=round(human_blocked / total, 4) if total else 0),
        PieSlice(label="Bot Detected",     count=bot_detected,       ratio=round(bot_detected / total, 4) if total else 0),
        PieSlice(label="Bot Missed",       count=bot_missed,         ratio=round(bot_missed / total, 4) if total else 0),
    ]

    return SummaryResponse(
        generated_at=cache.get("generated_at", ""),
        model_version=cache.get("model_version", ""),
        total_sessions=total,
        human_total=s.get("human_total", 0),
        bot_total=s.get("bot_total", 0),
        human_pass_rate=s.get("human_pass_rate", 0.0),
        human_suspicious_rate=s.get("human_suspicious_rate", 0.0),
        human_block_rate=s.get("human_block_rate", 0.0),
        bot_detect_rate=s.get("bot_detect_rate", 0.0),
        bot_miss_rate=s.get("bot_miss_rate", 0.0),
        pie_chart=pie,
    )


@router.get(
    "/attack_types",
    response_model=AttackTypesResponse,
    summary="공격 유형 Top N (기본 5)",
)
def get_attack_types(top_n: int = Query(default=5, ge=1, le=20)):
    cache = _load_cache()
    raw_counts: Dict[str, Dict[str, int]] = cache.get("summary", {}).get("attack_type_counts", {})

    items: List[AttackTypeItem] = []
    for key, val in raw_counts.items():
        cnt = val.get("count", 0)
        blocked = val.get("blocked", 0)
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
        total_bot_sessions=cache.get("summary", {}).get("bot_total", 0),
        top_types=items[:top_n],
    )


@router.get(
    "/risk_distribution",
    response_model=RiskDistributionResponse,
    summary="Safe / Suspicious / Critical 분포",
)
def get_risk_distribution():
    cache = _load_cache()
    counts: Dict[str, int] = cache.get("summary", {}).get("risk_band_counts", {})
    total = cache.get("summary", {}).get("total_sessions", 0)

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
    sessions = cache.get("sessions", [])

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
def get_traffic():
    cache = _load_cache()
    sessions = cache.get("sessions", [])

    traffic_dict = {}

    for s in sessions:
        filename = s.get("file", "")
        is_blocked = s.get("is_blocked", False)

        # 파일명에서 13자리 타임스탬프 추출 (예: ..._1777424592586.json)
        match = re.search(r'_(\d{13})\.json', filename)
        if match:
            ts_ms = int(match.group(1))
            # 밀리초를 초 단위로 변환하여 시간대(HH:00) 텍스트 생성
            dt = datetime.fromtimestamp(ts_ms / 1000.0)
            hour_str = dt.strftime('%H:00')
        else:
            hour_str = "00:00"

        if hour_str not in traffic_dict:
            traffic_dict[hour_str] = {"time": hour_str, "success": 0, "attack": 0}

        # JSON에 있는 진짜 True / False 값을 기준으로 카운트
        if is_blocked:
            traffic_dict[hour_str]["attack"] += 1
        else:
            traffic_dict[hour_str]["success"] += 1

    # 시간을 기준으로 오름차순 정렬
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