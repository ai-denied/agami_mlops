"""
Public API Handlers
====================
WBS #43: 핵심 캡챠 엔드포인트 3개.

1. POST /v1/challenges               — 챌린지 발급 (Browser → 우리 서버)
2. POST /v1/challenges/{cid}/answer  — 정답 제출 (Browser → 우리 서버)
3. POST /v1/siteverify               — 토큰 검증 (Tenant Backend → 우리 서버)

#41(generator), #42(스키마) 의 산출물을 그대로 조립함. 새 도메인 로직은 없음.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import enforce_rate_limit, get_db, get_store
from app.api.policy import FAILURE_WINDOW_SEC, decide_difficulty
from app.cache.challenge_store import ChallengeStore
from app.captcha.challenge_types import (
    ChallengeKind,
    ContextChallengeSpec,
    Difficulty,
    FaceChallengeSpec,
    FlashlightChallengeSpec,
)
from app.captcha.context_generator import generate_context_challenge
from app.captcha.face_generator import generate_face_challenge
from app.captcha.flashlight_generator import generate_flashlight_challenge
from app.captcha.flashlight_model import FlashlightModel
from app.captcha.captcha_logger import schedule_attempt_log
from app.captcha.flashlight_policy import evaluate_flashlight_decision
from app.captcha.mouse_features import extract_features_for_model
from app.captcha.verifier import (
    baseline_verdict,
    check_context_hit,
    check_face_hit,
    check_flashlight_hit,
)
from app.core.config import get_settings
from app.core.security import (
    hash_secret,
    make_captcha_token,
    parse_captcha_token,
)
from app.db.models import ApiKey, Challenge, TenantSettings, Verification
from app.schemas.api_dto import (
    FlashlightSubmitResponse,
    IssueChallengeRequest,
    SiteVerifyResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# 1. POST /v1/challenges  — 챌린지 발급
# ---------------------------------------------------------------------------

@router.post("/challenges")
async def issue_challenge(
    body: IssueChallengeRequest,
    request: Request,
    api_key: ApiKey = Depends(enforce_rate_limit),
    db: AsyncSession = Depends(get_db),
    store: ChallengeStore = Depends(get_store),
) -> FlashlightChallengeSpec | FaceChallengeSpec | ContextChallengeSpec:
    settings = get_settings()

    # 지원 kind: flashlight, face_mission, context_inference. 그 외는 차후 확장.
    if body.kind not in (
        ChallengeKind.FLASHLIGHT,
        ChallengeKind.FACE_MISSION,
        ChallengeKind.CONTEXT_INFERENCE,
    ):
        raise HTTPException(
            status_code=400,
            detail={"code": "kind_not_supported", "message": f"Kind {body.kind} not yet supported."},
        )

    # 난이도: tenant 기본값을 가져오고, 같은 IP 의 최근 실패 횟수에 따라 동적 상향 (#45).
    ts = (
        await db.execute(
            select(TenantSettings).where(TenantSettings.tenant_id == api_key.tenant_id)
        )
    ).scalar_one_or_none()
    tenant_default = Difficulty(ts.default_difficulty if ts else settings.default_difficulty)

    ip = request.client.host if request.client else "unknown"
    fail_count = await store.get_failure_count(ip)

    difficulty = decide_difficulty(body.difficulty, tenant_default, fail_count)

    # kind 별 generator 호출 (모두 순수 함수, (spec, answer) 페어 반환)
    if body.kind == ChallengeKind.FLASHLIGHT:
        spec, answer = generate_flashlight_challenge(difficulty)
        variant_value: str | None = spec.variant.value
    elif body.kind == ChallengeKind.FACE_MISSION:
        spec, answer = generate_face_challenge(difficulty)
        variant_value = None
    else:  # CONTEXT_INFERENCE
        spec, answer = generate_context_challenge(difficulty)
        variant_value = None

    # Hot path: 정답을 Redis 에 저장 (TTL = time_limit + 10s)
    await store.save_answer(answer)

    # Cold path: 발급 사실을 Postgres 에 기록 (대시보드/감사용)
    db.add(Challenge(
        id=spec.challenge_id,
        tenant_id=api_key.tenant_id,
        api_key_id=api_key.id,
        kind=spec.kind.value,
        variant=variant_value,
        difficulty=spec.difficulty.value,
        issued_at=spec.issued_at,
        expires_at=spec.expires_at,
        requester_ip=request.client.host if request.client else None,
        requester_origin=request.headers.get("origin"),
    ))
    await db.commit()

    # spec 만 반환 (정답 정보 없음 — challenge_types.py 가 구조적으로 차단)
    return spec


# ---------------------------------------------------------------------------
# 2. POST /v1/challenges/{cid}/answer  — 정답 제출
# ---------------------------------------------------------------------------

@router.post("/challenges/{challenge_id}/answer")
async def submit_answer(
    challenge_id: str,
    body: SubmitAnswerRequest,
    request: Request,
    api_key: ApiKey = Depends(enforce_rate_limit),
    db: AsyncSession = Depends(get_db),
    store: ChallengeStore = Depends(get_store),
) -> SubmitAnswerResponse | FlashlightSubmitResponse:
    settings = get_settings()

    # 1. Redis 에서 atomic GETDEL — 1회용
    answer = await store.consume_answer(challenge_id)
    if answer is None:
        raise HTTPException(
            status_code=410,
            detail={
                "code": "challenge_expired_or_consumed",
                "message": "This challenge has expired or already been answered.",
            },
        )

    # 2. kind 별 검증 분기. baseline_verdict 는 동일하게 사용 (#44 가 교체).
    #    답안은 위에서 GETDEL 로 이미 Redis 에서 제거됨 → 오답이어도 챌린지는 폐기됨
    #    (flashlight 와 동일한 1회용 정책).
    if answer.kind == ChallengeKind.FLASHLIGHT:
        if not body.flashlight_submissions or len(body.flashlight_submissions) != 3:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "missing_flashlight_submissions",
                    "message": "flashlight_submissions must contain exactly 3 entries.",
                },
            )
        # 1챌린지 = 3장 묶음 일괄 평가. early return.
        return await _flashlight_submit_bundle(
            challenge_id=challenge_id,
            body=body,
            answer=answer,
            request=request,
            api_key=api_key,
            db=db,
            store=store,
            settings=settings,
        )
    elif answer.kind == ChallengeKind.FACE_MISSION:
        # MediaPipe 합류 전까지의 임시 검증: 클라이언트 보고와 expected 가 일치하는지.
        hit = check_face_hit(answer, body.completed_instructions)
    elif answer.kind == ChallengeKind.CONTEXT_INFERENCE:
        if body.submitted_answers is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "missing_submitted_answers",
                    "message": "submitted_answers is required for context_inference challenges.",
                },
            )
        hit = check_context_hit(answer, body.submitted_answers)
    else:
        # 미래 확장 대비
        hit = False

    # 3. verdict 산출. #44 가 들어오면 baseline_verdict 대신 AI 스코어 사용.
    verdict, confidence = baseline_verdict(hit)

    # 4. 검증 결과를 Postgres 에 기록 (대시보드 통계의 원천)
    behavioral_summary = body.behavioral_data.model_dump() if body.behavioral_data else None
    db.add(Verification(
        challenge_id=challenge_id,
        tenant_id=api_key.tenant_id,
        success=hit,
        verdict=verdict,
        confidence=confidence,
        time_taken_ms=body.behavioral_data.time_taken_ms if body.behavioral_data else None,
        behavioral_summary=behavioral_summary,
        requester_ip=request.client.host if request.client else None,
    ))
    await db.commit()

    # 5. 실패 시 토큰 발급 X. 사용자가 다시 시도하도록 422 리턴.
    #    같은 IP 의 누적 실패를 카운트해두면 다음 issue_challenge 가 난이도 상향 (#45).
    if not hit:
        ip = request.client.host if request.client else "unknown"
        await store.incr_failure(ip, FAILURE_WINDOW_SEC)
        raise HTTPException(
            status_code=422,
            detail={"code": "verification_failed", "message": "Captcha verification failed. Please try again."},
        )

    # 6. captcha_token 발급 (HMAC 서명) + Redis 에 페이로드 저장
    token = make_captcha_token(challenge_id, settings.captcha_token_secret)
    await store.save_token(
        challenge_id,
        payload={
            "verdict": verdict,
            "confidence": confidence,
            "hostname": request.headers.get("origin", ""),
            "ts": answer.created_at.isoformat(),
            "ip": request.client.host if request.client else "",
        },
        ttl=120,
    )

    return SubmitAnswerResponse(captcha_token=token, expires_in=120)


# ---------------------------------------------------------------------------
# 2-a. Flashlight Bundle helper — 1챌린지 = 3장 묶음 일괄 평가
# ---------------------------------------------------------------------------

async def _flashlight_submit_bundle(
    *,
    challenge_id: str,
    body: SubmitAnswerRequest,
    answer,
    request: Request,
    api_key: ApiKey,
    db: AsyncSession,
    store: ChallengeStore,
    settings,
) -> FlashlightSubmitResponse:
    """손전등 캡챠 정답 제출 (1챌린지 = 3장 묶음).

    흐름:
      1) submissions 인덱스/길이 검증 (이미 caller에서 길이 3 확인됨)
      2) 인덱스 순서 정렬 + sub_answers와 zip
      3) 각 sub: 좌표 매칭 + 모델 추론
      4) 정책 평가 (좌표 2/3 이상 AND high_risk 0회 → allow)
      5) Verification 테이블 1행 기록 (번들 단위, summary에 3장 결과 nest)
      6) decision별 token 발급 또는 incr_failure
    """
    submissions = body.flashlight_submissions
    indices = {s.index for s in submissions}
    if indices != {0, 1, 2}:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_submission_indices",
                "message": "submissions must have indices {0, 1, 2}.",
            },
        )

    submissions = sorted(submissions, key=lambda s: s.index)
    sub_answers = sorted(answer.sub_answers, key=lambda a: a.index)

    coord_hits: list[bool] = []
    scores: list[float] = []
    risk_bands: list[str] = []

    for sub_answer, submission in zip(sub_answers, submissions):
        coord_hit = check_flashlight_hit(sub_answer, submission.click_x, submission.click_y)
        coord_hits.append(coord_hit)

        score: float = 0.0
        risk_band: str = "low_risk"
        if submission.trajectory and len(submission.trajectory) >= 2:
            try:
                dynamic, static = extract_features_for_model(submission.trajectory)
                model = FlashlightModel.get_instance()
                score = model.predict(dynamic, static)
                risk_band = model.classify(score)
            except ValueError as e:
                logger.info(
                    "flashlight trajectory invalid sub=%d: %s", submission.index, e
                )
            except Exception:
                logger.exception(
                    "flashlight model inference failed sub=%d; treat low_risk",
                    submission.index,
                )
        scores.append(float(score))
        risk_bands.append(risk_band)

    decision = evaluate_flashlight_decision(scores, coord_hits)

    # Verification 1행 기록 (번들 단위)
    behavioral_summary = body.behavioral_data.model_dump() if body.behavioral_data else {}
    behavioral_summary.update({
        "scores": scores,
        "risk_bands": risk_bands,
        "coord_hits": coord_hits,
        "decision": decision,
    })
    verdict = "human" if decision == "allow" else "bot"
    confidence = max(scores) if scores else 0.0
    db.add(Verification(
        challenge_id=challenge_id,
        tenant_id=api_key.tenant_id,
        success=(decision == "allow"),
        verdict=verdict,
        confidence=float(confidence),
        time_taken_ms=body.behavioral_data.time_taken_ms if body.behavioral_data else None,
        behavioral_summary=behavioral_summary,
        requester_ip=request.client.host if request.client else None,
    ))
    await db.commit()

    # --- 로컬 분석용 JSON 로그 (fire-and-forget; 응답 흐름 비차단) -----------------
    # 캡챠 응답에는 절대 영향 없음. captcha_logger 가 GC-safe set 으로 task 관리.
    _IMG_W = 800
    _IMG_H = 600
    sub_logs: list[dict] = []
    for sa, sub, score, risk, hit in zip(
        sub_answers, submissions, scores, risk_bands, coord_hits
    ):
        bbox_x_norm = sa.correct_x - sa.bbox_w / 2
        bbox_y_norm = sa.correct_y - sa.bbox_h / 2
        sub_logs.append({
            "index": sa.index,
            "image_url": sa.image_url,
            "target_object_id": sa.correct_object_id,
            "target_label": sa.target_label,
            # 정답 정규화
            "correct_x_norm": sa.correct_x,
            "correct_y_norm": sa.correct_y,
            "bbox_w_norm": sa.bbox_w,
            "bbox_h_norm": sa.bbox_h,
            # 정답 픽셀 (이미지 800x600 기준)
            "correct_x_px": round(sa.correct_x * _IMG_W),
            "correct_y_px": round(sa.correct_y * _IMG_H),
            "bbox_x_px": round(bbox_x_norm * _IMG_W),
            "bbox_y_px": round(bbox_y_norm * _IMG_H),
            "bbox_w_px": round(sa.bbox_w * _IMG_W),
            "bbox_h_px": round(sa.bbox_h * _IMG_H),
            # 사용자 클릭
            "click_x_norm": sub.click_x,
            "click_y_norm": sub.click_y,
            "click_x_px": round(sub.click_x * _IMG_W),
            "click_y_px": round(sub.click_y * _IMG_H),
            # 매칭 / 모델
            "coord_hit": hit,
            "bot_risk_score": float(score),
            "risk_band": risk,
            # 마우스 궤적 raw (프론트가 보낸 그대로: {x,y,t})
            "trajectory": list(sub.trajectory) if sub.trajectory else [],
            "trajectory_count": len(sub.trajectory) if sub.trajectory else 0,
        })

    log_payload: dict = {
        "challenge_id": challenge_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "difficulty": answer.difficulty,
        "flashlight_radius": answer.flashlight_radius,
        "time_limit_sec": answer.time_limit_sec,
        "canvas_aspect_w": answer.canvas_aspect_w,
        "canvas_aspect_h": answer.canvas_aspect_h,
        "decision": decision,
        "coord_hits_total": sum(coord_hits),
        "coord_hits_required": 2,  # MIN_COORD_HITS
        "any_high_risk": any(s >= 0.60 for s in scores),
        "behavioral_data": (
            body.behavioral_data.model_dump() if body.behavioral_data else None
        ),
        "sub_attempts": sub_logs,
        "user_agent": request.headers.get("user-agent"),
        "requester_ip": request.client.host if request.client else None,
    }
    schedule_attempt_log(log_payload)
    # --- 로그 끝 ------------------------------------------------------------------

    if decision == "allow":
        token = make_captcha_token(challenge_id, settings.captcha_token_secret)
        await store.save_token(
            challenge_id,
            payload={
                "verdict": "human",
                "confidence": float(confidence),
                "hostname": request.headers.get("origin", ""),
                "ts": answer.created_at.isoformat(),
                "ip": request.client.host if request.client else "",
            },
            ttl=120,
        )
        return FlashlightSubmitResponse(
            decision="allow",
            captcha_token=token,
            expires_in=120,
            coord_hits=coord_hits,
            scores=scores,
            risk_bands=risk_bands,
        )

    # block: incr_failure 1회 호출. 기존 시도당 호출에서 챌린지당 1회로 변경 (의도된 정책 변경).
    ip = request.client.host if request.client else "unknown"
    await store.incr_failure(ip, FAILURE_WINDOW_SEC)
    return FlashlightSubmitResponse(
        decision="block",
        coord_hits=coord_hits,
        scores=scores,
        risk_bands=risk_bands,
    )


# ---------------------------------------------------------------------------
# 3. POST /v1/siteverify  — 기업 백엔드의 토큰 검증
# ---------------------------------------------------------------------------

@router.post("/siteverify", response_model=SiteVerifyResponse)
async def siteverify(
    secret: str = Form(...),
    token: str = Form(...),
    remoteip: str | None = Form(None),  # noqa: ARG001 (감사 로그 확장 시 사용)
    db: AsyncSession = Depends(get_db),
    store: ChallengeStore = Depends(get_store),
) -> SiteVerifyResponse:
    """
    응답 규칙: 모든 실패 케이스도 HTTP 200 + success=false 로 반환 (reCAPTCHA 호환).
    """
    settings = get_settings()

    # 1. 토큰 형식/서명 검증 — 위조 토큰을 DB/Redis 까지 흘리지 않음.
    challenge_id = parse_captcha_token(token, settings.captcha_token_secret)
    if challenge_id is None:
        return SiteVerifyResponse(success=False, error_codes=["invalid-input-token"])

    # 2. 제출된 secret 으로 tenant 조회.
    secret_hash_value = hash_secret(secret, settings.api_key_hmac_pepper)
    stmt = select(ApiKey).where(
        ApiKey.secret_hash == secret_hash_value,
        ApiKey.revoked_at.is_(None),
    )
    api_key = (await db.execute(stmt)).scalar_one_or_none()
    if api_key is None:
        return SiteVerifyResponse(success=False, error_codes=["invalid-input-secret"])

    # 3. Redis 에서 atomic GETDEL — 토큰 1회용.
    payload = await store.consume_token(challenge_id)
    if payload is None:
        return SiteVerifyResponse(success=False, error_codes=["timeout-or-duplicate"])

    # 4. (선택) 토큰의 challenge 가 정말 이 tenant 의 것인지 cross-check.
    #    secret_key 인증을 통과한 시점에 사실상 안전하지만, defense-in-depth.
    stmt2 = select(Challenge.tenant_id).where(Challenge.id == challenge_id)
    chal_tenant_id = (await db.execute(stmt2)).scalar_one_or_none()
    if chal_tenant_id is None or chal_tenant_id != api_key.tenant_id:
        return SiteVerifyResponse(success=False, error_codes=["invalid-input-token"])

    # 5. 성공.
    return SiteVerifyResponse(
        success=True,
        verdict=payload["verdict"],
        confidence=payload["confidence"],
        challenge_ts=datetime.fromisoformat(payload["ts"]),
        hostname=payload.get("hostname") or None,
        error_codes=[],
    )
