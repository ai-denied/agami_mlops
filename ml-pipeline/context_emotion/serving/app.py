"""
Emotion CAPTCHA Serving API.

엔드포인트:
  GET  /health                    풀 로드 상태 확인
  POST /context-emotion/challenge current 풀에서 문제 1개 출제
  POST /context-emotion/attempt   사용자 풀이 제출 및 attempt 로그 기록

실행:
  uvicorn context_emotion.serving.app:app --host 0.0.0.0 --port 8083

환경 변수:
  CAPTCHA_POOL_DIR      (기본: /model-store/context_emotion/current)
  ATTEMPT_LOG_DIR       (기본: /data/context_emotion/attempt_logs)
  IMAGE_BASE_URL        (기본: /static/images)
  IMAGE_BASE_DIR        (기본: /data/context_emotion)  — 정적 파일 서빙용
  CHALLENGE_TTL_SEC     (기본: 300)
  MAX_CHALLENGE_RETRIES (기본: 2)
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from context_emotion.serving import attempt_logger, challenge_sampler, pool_loader
from context_emotion.serving.schemas import (
    AttemptRequest,
    AttemptResponse,
    ChallengeRequest,
    ChallengeResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_IMAGE_BASE_DIR = Path(os.getenv("IMAGE_BASE_DIR", "/data/context_emotion"))


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 풀 초기 로드 + 백그라운드 태스크
    logger.info("CAPTCHA Serving API 시작")
    await pool_loader.reload()

    reload_task  = asyncio.create_task(pool_loader.background_reload_loop())
    cleanup_task = asyncio.create_task(challenge_sampler.cleanup_loop())

    # 정적 파일 서빙 (IMAGE_BASE_DIR 가 존재하는 경우에만)
    if _IMAGE_BASE_DIR.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount(
            "/static/images",
            StaticFiles(directory=str(_IMAGE_BASE_DIR)),
            name="images",
        )
        logger.info("정적 이미지 서빙: %s → /static/images", _IMAGE_BASE_DIR)

    yield

    # Shutdown
    reload_task.cancel()
    cleanup_task.cancel()
    logger.info("CAPTCHA Serving API 종료")


# ── 앱 ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emotion CAPTCHA Serving API",
    description="current captcha_pool.csv 기반 문제 출제 및 attempt 로깅",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── 공통 의존성 ───────────────────────────────────────────────────────────────

def _require_pool() -> pool_loader.PoolState:
    pool = pool_loader.get_pool()
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CAPTCHA 풀이 아직 로드되지 않았습니다. 잠시 후 재시도하세요.",
        )
    return pool


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="서비스 상태 확인",
)
async def health() -> HealthResponse:
    pool = pool_loader.get_pool()
    if pool is None:
        return HealthResponse(
            status="unavailable",
            pool_loaded=False,
            problem_count=0,
            version="",
            pool_loaded_at=None,
        )
    return HealthResponse(
        status="ok",
        pool_loaded=True,
        problem_count=pool.problem_count,
        version=pool.version,
        pool_loaded_at=pool.loaded_at.isoformat(),
    )


@app.post(
    "/context-emotion/challenge",
    response_model=ChallengeResponse,
    summary="CAPTCHA 문제 1개 출제",
    description=(
        "current captcha_pool.csv 에서 문제를 샘플링해 반환한다. "
        "**정답(correct_label), security_grade, attacker_proxy_score 는 응답에 포함되지 않는다.**"
    ),
)
async def get_challenge(req: ChallengeRequest) -> ChallengeResponse:
    pool = _require_pool()

    try:
        session, image_url = challenge_sampler.create_challenge(pool, req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except Exception as e:
        logger.exception("challenge 생성 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="문제 생성 중 오류가 발생했습니다.",
        ) from e

    return ChallengeResponse(
        challenge_id=session.challenge_id,
        image_url=image_url,
        choices=session.choices,
        expires_at=session.expires_at,
    )


@app.post(
    "/context-emotion/attempt",
    response_model=AttemptResponse,
    summary="사용자 풀이 제출",
    description=(
        "사용자의 선택을 채점하고 attempt log에 기록한다. "
        "**점수(points), 정답(correct_label) 은 응답에 포함되지 않는다.**"
    ),
)
async def submit_attempt(req: AttemptRequest) -> AttemptResponse:
    pool = _require_pool()

    # ── challenge session 조회 ────────────────────────────────────────────
    session = challenge_sampler.get_session(req.challenge_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="challenge가 만료되었거나 존재하지 않습니다.",
        )

    # ── session_id 검증 ──────────────────────────────────────────────────
    if session.session_id != req.session_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="session_id가 challenge 발급 시와 일치하지 않습니다.",
        )

    # ── 채점 ─────────────────────────────────────────────────────────────
    try:
        is_correct, points, retry_allowed = attempt_logger.validate_and_score(session, req)
    except Exception as e:
        logger.exception("채점 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="채점 중 오류가 발생했습니다.",
        ) from e

    # ── 세션 무효화 (재제출 방지) ─────────────────────────────────────────
    if is_correct or not retry_allowed:
        challenge_sampler.invalidate(req.challenge_id)

    # ── 로깅 (비동기, 실패해도 응답은 정상 반환) ──────────────────────────
    attempt_logger.log_attempt(
        session=session,
        req=req,
        is_correct=is_correct,
        points=points,
        pool_version=pool.version,
        problem_count=pool.problem_count,
    )

    return AttemptResponse(is_correct=is_correct, retry_allowed=retry_allowed, score=points)
