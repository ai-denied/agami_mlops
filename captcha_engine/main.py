"""
FastAPI Application Entry Point
================================
WBS #43: 앱 부트스트랩 + 공통 미들웨어/에러 핸들러.

실행
----
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# 💡 새로 추가된 부품(Import)입니다.
from pydantic import BaseModel
from typing import List

from app.api.public import router as public_router
from app.cache.redis_client import close_redis, init_redis
from app.captcha.flashlight_image_dataset import IMAGE_DIR, get_dataset
from app.core.config import get_settings
from app.db.session import dispose_engine, init_engine
from fastapi.middleware.cors import CORSMiddleware
import uuid
import random
logger = logging.getLogger("captcha")


# ---------------------------------------------------------------------------
# Lifespan : 앱 기동/종료 시점의 자원 초기화/정리
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    logger.info("captcha-engine starting (env=%s)", settings.app_env)

    init_engine()
    await init_redis()

    # 손전등 이미지 데이터셋 인덱싱 (앱 시작 시 1회. 첫 챌린지 요청 지연 방지).
    entries = get_dataset()
    logger.info("flashlight image dataset preloaded: %d entries", len(entries))

    yield

    logger.info("captcha-engine stopping")
    await close_redis()
    await dispose_engine()


app = FastAPI(
    title="Captcha Engine",
    description="Behavioral CAPTCHA SaaS API",
    version="0.1.0",
    lifespan=lifespan,
)
# CORS 허용 origin 목록은 Settings.cors_origins (콤마 구분 문자열) 에서 읽어온다.
# 로컬: "http://localhost:5173" / 운영: ConfigMap 에서 "http://210.109.53.140,..." 주입.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Middleware : request_id 부여 (디버깅/고객지원의 기본)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ---------------------------------------------------------------------------
# Exception Handlers : 표준 에러 응답 포맷 (API_SPEC.md 와 일치)
# ---------------------------------------------------------------------------

def _wrap_error(detail, request: Request, status: int) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    if isinstance(detail, dict):
        body = {"error": {**detail, "request_id": rid}}
    else:
        body = {"error": {"code": "http_error", "message": str(detail), "request_id": rid}}
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return _wrap_error(exc.detail, request, exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _wrap_error(
        {"code": "validation_error", "message": "Invalid request payload.", "errors": exc.errors()},
        request,
        422,
    )


# ---------------------------------------------------------------------------
# 가짜 심사위원 (Mock API) 데이터 규격 정의
# ---------------------------------------------------------------------------
class TrajectoryPoint(BaseModel):
    t: int  # 시간
    x: int  # X 좌표
    y: int  # Y 좌표

class FlashlightVerifyRequest(BaseModel):
    captcha_id: str # 👈 새로 추가된 바코드(토큰)
    trajectory: List[TrajectoryPoint]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """K8s liveness/readiness probe 용."""
    return {"status": "ok"}

# 💡 새로 추가된 라우터: 무작위 시험지를 발급해줍니다.
@app.get("/api/challenge")
async def get_challenge():
    # 1. 고유한 바코드(UUID) 생성
    captcha_id = str(uuid.uuid4())
    
    # 2. 무작위 이미지 URL 생성 (seed 값을 랜덤으로 주어 매번 다른 이미지가 나오게 함)
    random_seed = random.randint(1, 1000)
    image_url = f"https://picsum.photos/seed/{random_seed}/500/350"
    
    # (원래는 여기서 redis에 captcha_id를 저장하여 나중에 검증해야 하지만, 
    # 일단 통신 흐름을 먼저 뚫기 위해 바로 return 합니다!)
    
    return {
        "captcha_id": captcha_id,
        "image_url": image_url
    }

# 💡 기존 검증 라우터 업데이트: 바코드(captcha_id)를 함께 받도록 수정
@app.post("/api/verify/flashlight")
async def verify_flashlight(request: FlashlightVerifyRequest):
    captcha_id = request.captcha_id
    trajectory_data = request.trajectory
    
    print(f"📦 [문제번호: {captcha_id}] 정답지 도착! 데이터 길이: {len(trajectory_data)}")
    
    data_length = len(trajectory_data)
    
    if data_length < 50:
        return {
            "status": "success",
            "result": "bot",
            "score": 0.12,
            "message": "비정상적인 궤적이 감지되었습니다."
        }
    
    return {
        "status": "success",
        "result": "human",
        "score": 0.98,
        "message": "검증에 성공했습니다."
    }

# 손전등 캡챠 배경 이미지 정적 서빙. URL: /static/captcha_images/captcha_XXXX.jpg
app.mount(
    "/static/captcha_images",
    StaticFiles(directory=str(IMAGE_DIR)),
    name="captcha_images",
)

app.include_router(public_router)