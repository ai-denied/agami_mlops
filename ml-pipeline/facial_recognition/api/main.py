#!/usr/bin/env python3
"""
얼굴 활성도(Face Liveness) 추론 API

엔드포인트:
  POST /api/v1/predict  — x_seq (16×20) → spoof_score
  POST /api/v1/decide   — 3라운드 결과 → CAPTCHA 최종 판정 (PASS/RETRY/FAIL)
  GET  /health          — 헬스체크
  GET  /model/info      — 로드된 모델 메타데이터

실행:
  uvicorn facial_recognition.api.main:app --host 0.0.0.0 --port 8081
  python -m facial_recognition.api.main
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from facial_recognition.api import loader, schemas
from facial_recognition.captcha_decision import MissionRound, decide_three_round_captcha
from facial_recognition.inference.onnx_face_liveness_detector import classify_spoof_risk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
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
    title="Face Liveness Inference API",
    description="얼굴 활성도 GRU 모델 기반 위변조 탐지 추론 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

# dashboard 라우터는 의도적으로 등록하지 않는다 — facial_recognition 쪽에는
# dashboard_cache.json을 만드는 precompute_dashboard.py/CronJob이 아직 없어서
# (flashlight에만 있음) 등록하면 /api/v1/dashboard/* 가 항상 503을 낸다.
# dashboard가 필요해지면 facial_recognition 전용 precompute 구현과 함께 추가한다.


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


# ── 단일 시퀀스 추론 ──────────────────────────────────────────────────────────

@app.post(
    "/api/v1/predict",
    response_model=schemas.PredictResponse,
    summary="얼굴 피처 시퀀스 1회 → spoof_score",
    responses={
        422: {"model": schemas.ErrorResponse, "description": "입력 유효성 오류"},
        503: {"model": schemas.ErrorResponse, "description": "모델 미로드"},
    },
)
def predict(req: schemas.PredictRequest):
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    detector = loader.get_detector()

    try:
        x_seq = np.array(req.x_seq, dtype=np.float32)   # (16, 20)
        result = detector.predict(x_seq, seq_length=req.seq_length)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("추론 중 오류 발생")
        raise HTTPException(status_code=500, detail="추론 실패")

    return schemas.PredictResponse(
        spoof_score=result["spoof_score"],
        risk_band=result["risk_band"],
        is_spoof=result["is_spoof"],
        low_spoof_threshold=result["low_spoof_threshold"],
        high_spoof_threshold=result["high_spoof_threshold"],
        seq_length=req.seq_length,
    )


# ── 3라운드 CAPTCHA 판정 ──────────────────────────────────────────────────────

@app.post(
    "/api/v1/decide",
    response_model=schemas.DecideResponse,
    summary="3라운드 결과 → CAPTCHA 최종 판정 (PASS/RETRY/FAIL)",
    responses={
        422: {"model": schemas.ErrorResponse, "description": "입력 유효성 오류"},
        503: {"model": schemas.ErrorResponse, "description": "모델 미로드"},
    },
)
def decide(req: schemas.DecideRequest):
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    detector = loader.get_detector()

    try:
        rounds = [
            MissionRound(
                round_id=r.round_id,
                mission_type=r.mission_type,
                spoof_score=r.spoof_score,
                # risk_band는 클라이언트가 보낸 값을 신뢰하지 않고, 현재 로드된
                # 모델의 detector.low_thr/high_thr 로 서버에서 항상 재계산한다.
                # /predict 와 /decide 가 서로 다른 threshold를 쓰는 것을 방지한다.
                risk_band=classify_spoof_risk(
                    float(r.spoof_score), detector.low_thr, detector.high_thr
                ),
                mission_pass=r.mission_pass,
                face_detected=r.face_detected,
                timeout=r.timeout,
                mission_name=r.mission_name,
                hand_detected=r.hand_detected,
                detail=r.detail,
            )
            for r in sorted(req.rounds, key=lambda x: x.round_id)
        ]
        result = decide_three_round_captcha(
            rounds, low_thr=detector.low_thr, high_thr=detector.high_thr
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("판정 중 오류 발생")
        raise HTTPException(status_code=500, detail="판정 실패")

    return schemas.DecideResponse(
        decision=result.decision,
        reason=result.reason,
        total_risk=round(result.total_risk, 6),
        avg_spoof_score=round(result.avg_spoof_score, 6),
        failed_mission_count=result.failed_mission_count,
        failed_face_count=result.failed_face_count,
        timeout_count=result.timeout_count,
        spoof_detected_count=result.spoof_detected_count,
        risk_bands=result.risk_bands,
    )


# ── 직접 실행 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("facial_recognition.api.main:app", host="0.0.0.0", port=8081, reload=False)
