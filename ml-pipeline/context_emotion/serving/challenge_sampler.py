"""
CAPTCHA 문제 샘플링 및 challenge session 관리.

보안 원칙:
  - challenge_id 는 서버가 생성한 UUID (sample_id 미노출).
  - ChallengeSession 의 final_emotion 은 서버 메모리에만 존재하며
    API 응답에는 절대 포함되지 않는다.
  - choices 는 generate_choices() 가 생성한 4개 레이블을 셔플해서 반환한다.
  - TTL 이 지난 session 은 자동 만료된다.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from context_emotion.captcha_bank.choice_generation import (
    EMOTIONS,
    generate_choices,
)
from context_emotion.serving.pool_loader import PoolState

logger = logging.getLogger(__name__)

_CHALLENGE_TTL_SEC = int(os.getenv("CHALLENGE_TTL_SEC", "300"))    # 5분
_MAX_PER_SESSION   = int(os.getenv("MAX_CHALLENGES_PER_SESSION", "10"))
_IMAGE_BASE_URL    = os.getenv("IMAGE_BASE_URL", "/static/images")
_CLEANUP_INTERVAL  = 60  # seconds


# ── 세션 데이터 ───────────────────────────────────────────────────────────────

@dataclass
class ChallengeSession:
    challenge_id: str       # UUID (클라이언트에 전달)
    sample_id: str          # 실제 풀 sample_id (서버 내부 전용)
    session_id: str         # 클라이언트 session_id
    choices: list[str]      # 제시된 4지선다 (순서 그대로)
    final_emotion: str      # 정답 레이블 (서버 내부 전용, 응답 미포함)
    aux_emotions: list[str] # 부분 점수 레이블 (서버 내부 전용)
    issued_at: float        # time.monotonic()
    expires_at: datetime    # UTC datetime


# ── 모듈 수준 저장소 ─────────────────────────────────────────────────────────

_sessions: dict[str, ChallengeSession] = {}
_session_counts: dict[str, int] = {}     # session_id → 발급 수 (남용 방지)


# ── 공개 함수 ─────────────────────────────────────────────────────────────────

def create_challenge(pool: PoolState, session_id: str) -> tuple[ChallengeSession, str]:
    """풀에서 문항 1개를 샘플링해 ChallengeSession을 생성하고 image_url을 반환한다."""
    if _session_counts.get(session_id, 0) >= _MAX_PER_SESSION:
        raise ValueError(f"세션당 최대 challenge 수 초과: session_id={session_id}")

    # ── 샘플링: security_grade 가중치 ──────────────────────────────────────
    row = _weighted_sample(pool.rows)

    # ── 4지선다 생성 (셔플) ────────────────────────────────────────────────
    # seed는 challenge_id(UUID) 기반 → 요청마다 다른 셔플 순서
    challenge_id = str(uuid.uuid4())
    seed = int(uuid.UUID(challenge_id).int % (2 ** 31))
    choices = generate_choices(row, seed=seed)

    # ── aux_emotions 파싱 ──────────────────────────────────────────────────
    from context_emotion.captcha_bank.choice_generation import parse_aux
    aux = parse_aux(row.get("aux_emotions", "[]"))

    # ── 세션 생성 ─────────────────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc)
    session = ChallengeSession(
        challenge_id=challenge_id,
        sample_id=row["sample_id"],
        session_id=session_id,
        choices=choices,
        final_emotion=row["final_emotion"],
        aux_emotions=aux,
        issued_at=time.monotonic(),
        expires_at=now_utc + timedelta(seconds=_CHALLENGE_TTL_SEC),
    )

    _sessions[challenge_id] = session
    _session_counts[session_id] = _session_counts.get(session_id, 0) + 1

    # ── image_url 구성 ────────────────────────────────────────────────────
    # 절대 경로에서 파일명만 추출해 정적 URL 생성
    import os.path
    image_path = row.get("image_path", "")
    filename = os.path.basename(image_path) if image_path else "unknown.jpg"
    image_url = f"{_IMAGE_BASE_URL}/{filename}"

    return session, image_url


def get_session(challenge_id: str) -> Optional[ChallengeSession]:
    """challenge_id에 해당하는 유효한 session을 반환한다. 만료 시 None."""
    session = _sessions.get(challenge_id)
    if session is None:
        return None
    if time.monotonic() > session.issued_at + _CHALLENGE_TTL_SEC:
        _sessions.pop(challenge_id, None)
        return None
    return session


def invalidate(challenge_id: str) -> None:
    """attempt 처리 완료 후 session을 즉시 무효화한다."""
    _sessions.pop(challenge_id, None)


# ── 가중치 샘플링 ─────────────────────────────────────────────────────────────

_GRADE_WEIGHTS = {"S": 4, "A": 3, "B": 2, "C": 1, "": 1}


def _weighted_sample(rows: list[dict]) -> dict:
    """security_grade 기반 가중 샘플링. 없으면 균등 샘플링."""
    weights = [
        _GRADE_WEIGHTS.get(r.get("security_grade", ""), 1)
        for r in rows
    ]
    # Python 3.6+ random.choices는 weights 합이 0이면 오류 — 방어
    if all(w == 1 for w in weights):
        return random.choice(rows)
    return random.choices(rows, weights=weights, k=1)[0]


# ── 만료 세션 정리 ────────────────────────────────────────────────────────────

async def cleanup_loop() -> None:
    """만료된 challenge session을 주기적으로 삭제한다."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL)
        _cleanup_expired()


def _cleanup_expired() -> None:
    cutoff = time.monotonic() - _CHALLENGE_TTL_SEC
    expired = [cid for cid, s in _sessions.items() if s.issued_at < cutoff]
    for cid in expired:
        _sessions.pop(cid, None)
    if expired:
        logger.debug("만료 session 정리: %d개", len(expired))
