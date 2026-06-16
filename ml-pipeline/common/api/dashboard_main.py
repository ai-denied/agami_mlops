#!/usr/bin/env python3
"""
CAPTCHA Dashboard API (standalone)

엔드포인트:
  GET /api/v1/dashboard/summary           — 전체 통계 + 파이 차트
  GET /api/v1/dashboard/attack_types      — 공격 유형 Top N
  GET /api/v1/dashboard/risk_distribution — 위험 밴드 분포
  GET /api/v1/dashboard/sessions          — 세션 목록 (페이지네이션)
  GET /api/v1/dashboard/traffic           — 시간대별 트래픽
  POST /api/v1/dashboard/reload           — 캐시 재로드
  GET /health                             — 헬스체크

실행:
  uvicorn common.api.dashboard_main:app --host 0.0.0.0 --port 8082
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from common.api.dashboard import router

app = FastAPI(
    title="CAPTCHA Dashboard API",
    description="멀티 캡챠 모델 대시보드 통계 API",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health", summary="헬스체크")
def health():
    cache_path = os.environ.get("DASHBOARD_CACHE_PATH", "")
    cache_exists = Path(cache_path).exists() if cache_path else False
    return JSONResponse({
        "status": "ok",
        "cache_path": cache_path,
        "cache_ready": cache_exists,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("common.api.dashboard_main:app", host="0.0.0.0", port=8082, reload=False)
