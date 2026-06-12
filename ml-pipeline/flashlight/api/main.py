#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flashlight CAPTCHA 추론 API

엔드포인트:
  POST /api/v1/predict   — trajectory 1회 → bot_risk_score
  POST /api/v1/decide    — 누적 score 목록 → allow/block 판정
  GET  /health           — 헬스체크
  GET  /model/info       — 로드된 모델 메타데이터

실행:
  uvicorn flashlight.api.main:app --host 0.0.0.0 --port 8080
  python -m flashlight.api.main
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from flashlight.api import loader, schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = loader.get_model_dir()
    logger.info(f"모델 로드 시작: {model_dir}")
    try:
        loader.load_detector()
        meta = loader.get_metadata()
        logger.info(f"모델 로드 완료 — version={meta.get('version', 'unknown')}")
    except FileNotFoundError as e:
        logger.warning(f"모델 파일 없음 (추론 불가): {e}")
    yield
    logger.info("서버 종료")


app = FastAPI(
    title="Flashlight CAPTCHA Inference API",
    description="마우스 trajectory 기반 봇 탐지 추론 서비스",
    version="1.0.0",
    lifespan=lifespan,
)


# ── 헬스체크 ──────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=schemas.HealthResponse,
    summary="서버 및 모델 상태 확인",
)
def health():
    if not loader.is_loaded():
        return schemas.HealthResponse(
            status="model_not_loaded",
            model_loaded=False,
            model_version="",
        )
    version = loader.get_metadata().get("version", "unknown")
    return schemas.HealthResponse(status="ok", model_loaded=True, model_version=version)


# ── 모델 정보 ─────────────────────────────────────────────────────────────────

@app.get(
    "/model/info",
    summary="로드된 모델의 메타데이터 반환",
)
def model_info():
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    return loader.get_metadata()


# ── 단일 시도 추론 ─────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/predict",
    response_model=schemas.PredictResponse,
    summary="trajectory 1회 → bot_risk_score",
    responses={
        422: {"model": schemas.ErrorResponse, "description": "입력 유효성 오류"},
        503: {"model": schemas.ErrorResponse, "description": "모델 미로드"},
    },
)
def predict(req: schemas.PredictRequest):
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    detector = loader.get_detector()
    trajectory = [p.model_dump() for p in req.trajectory]

    try:
        result = detector.predict_trajectory(
            trajectory,
            coordinate_mode=req.coordinate_mode,
            canvas_width=req.canvas_width,
            canvas_height=req.canvas_height,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("추론 중 오류 발생")
        raise HTTPException(status_code=500, detail="추론 실패")

    seq_len = len(result.get("feature_sample", {}).get("dynamic_features", []))

    return schemas.PredictResponse(
        bot_risk_score=result["bot_risk_score"],
        risk_band=result["risk_band"],
        low_risk_threshold=result["low_risk_threshold"],
        high_risk_threshold=result["high_risk_threshold"],
        seq_len=seq_len,
    )


# ── 3회 누적 정책 판정 ─────────────────────────────────────────────────────────

@app.post(
    "/api/v1/decide",
    response_model=schemas.DecideResponse,
    summary="누적 score 목록 → allow/block 최종 판정",
    responses={
        503: {"model": schemas.ErrorResponse, "description": "모델 미로드"},
    },
)
def decide(req: schemas.DecideRequest):
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    detector = loader.get_detector()
    result = detector.decide_three_attempts(req.scores)
    return schemas.DecideResponse(**result)


# ── 직접 실행 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("flashlight.api.main:app", host="0.0.0.0", port=8080, reload=False)
