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

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

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
# CORS:
#   widget 을 임의의 부모 도메인 iframe 에서 임베드 가능하게 하려면 allow_origins=["*"] 가
#   가장 단순. credentials (쿠키) 는 캡챠가 사용하지 않으므로 False 로 고정해도 무방.
#   * + credentials 조합은 Starlette/CORS 사양상 거부되므로 두 값이 동시에 변경되어야 함.
#   추후 부모 도메인을 알면 cors_origins 환경변수로 좁힐 것.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
# widget 통합 빌드의 프론트가 `${VITE_API_URL=/captcha}${image_url}` = /captcha/static/...
# 로 요청하기 때문에 동일 디렉토리를 /captcha/static/captcha_images 로도 노출.
# (기존 /api 는 agami-ingress 의 카카오 로그인 백엔드와 충돌하므로 /captcha 로 분리.)
# 두 경로 모두 같은 디렉토리를 가리키므로 디스크 중복 없음.
app.mount(
    "/captcha/static/captcha_images",
    StaticFiles(directory=str(IMAGE_DIR)),
    name="captcha_prefixed_images",
)


# ---------------------------------------------------------------------------
# /widget — iframe 임베드용 SPA (Vite 빌드 산출물)
#   - WIDGET_BUILD=1 vite build 의 결과를 captcha_engine/static/widget 에 둠.
#   - StaticFiles 의 html=True 는 디렉토리 → index.html 만 처리하고 클라이언트
#     라우팅(/widget/embed 등) 새로고침 시 404. 서브클래스로 404 → index.html 로
#     떨어뜨려 React Router 가 받게 한다.
#   - 디렉토리가 아직 없는 환경(로컬 dev, frontend 빌드 안 함)에선 마운트를 건너뜀.
# ---------------------------------------------------------------------------

WIDGET_DIR = Path(__file__).resolve().parent / "static" / "widget"


class SPAStaticFiles(StaticFiles):
    """디렉토리에 없는 path 요청을 index.html 로 fallback. React Router 호환."""

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and self.html:
                return FileResponse(Path(self.directory) / "index.html")
            raise


if WIDGET_DIR.exists():
    app.mount(
        "/widget",
        SPAStaticFiles(directory=str(WIDGET_DIR), html=True),
        name="widget",
    )
    logger.info("widget mounted at /widget from %s", WIDGET_DIR)
else:
    logger.info(
        "widget dist not found at %s — skipping /widget mount (run "
        "`WIDGET_BUILD=1 npm run build` in captcha-frontend and copy dist here)",
        WIDGET_DIR,
    )

app.include_router(public_router)
# 동일 router 를 /captcha prefix 로 한 번 더 노출 — widget 통합 빌드의 frontend 가
# .env.production 의 VITE_API_URL=/captcha 때문에 /captcha/v1/... 로 호출하기 때문.
# router 내부 prefix("/v1") 와 합쳐져 최종 /captcha/v1/* 경로가 활성화된다.
# 외부 기업 백엔드의 /v1/siteverify 직접 호출 경로도 그대로 유지.
# (기존 /api 는 agami-ingress 의 카카오 로그인 백엔드와 충돌하므로 /captcha 로 분리.)
app.include_router(public_router, prefix="/captcha")