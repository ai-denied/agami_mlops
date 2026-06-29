"""
attempt 검증·채점·로깅.

채점:
  - primary match (final_emotion): 1.0점
  - aux match (aux_emotions):      0.5점  — 부분 점수, is_correct=False
  - wrong:                         0.0점

로그 저장:
  /data/context_emotion/attempt_logs/attempts_YYYYMMDD.jsonl

개인정보 보호:
  - final_emotion(정답), image_path, choices 는 로그에 저장하지 않는다.
  - sample_id 는 feedback MLOps 상관관계 분석에 필요하므로 저장한다.
    (공개 식별자가 아닌 내부 ID)
  - 원본 IP, 원본 user-agent 저장 금지.
    필요 시 sha256[:16] 해시만 저장한다.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from context_emotion.serving.challenge_sampler import ChallengeSession
from context_emotion.serving.schemas import AttemptRequest

logger = logging.getLogger(__name__)

_ATTEMPT_LOG_DIR = Path(os.getenv("ATTEMPT_LOG_DIR", "/data/context_emotion/attempt_logs"))
_MAX_RETRIES = int(os.getenv("MAX_CHALLENGE_RETRIES", "2"))
_MIN_SOLVE_TIME_MS = int(os.getenv("MIN_SOLVE_TIME_MS", "800"))
_write_lock = threading.Lock()


# ── 채점 ─────────────────────────────────────────────────────────────────────

def validate_and_score(
    session: ChallengeSession,
    req: AttemptRequest,
) -> tuple[bool, float, bool]:
    """(is_correct, points, retry_allowed) 반환.

    is_correct: selected_label == final_emotion (정확 일치)
    points:     0.0 / 0.5 / 1.0 (피드백 MLOps 입력용, API 응답 미포함)
    retry_allowed: 오답이고 retry_count < MAX_RETRIES 일 때만 True
    """
    selected = req.selected_label

    # 제시된 선택지 외 레이블 거부 (injection 방어)
    if selected not in session.choices:
        return False, 0.0, req.retry_count < _MAX_RETRIES

    # 비정상적으로 빠른 응답 — 봇 탐지, 클라이언트에 이유 미노출
    if req.solve_time_ms < _MIN_SOLVE_TIME_MS:
        return False, 0.0, req.retry_count < _MAX_RETRIES

    is_correct = selected == session.final_emotion
    if is_correct:
        points = 1.0
    elif selected in session.aux_emotions:
        points = 0.5
    else:
        points = 0.0

    retry_allowed = (not is_correct) and req.retry_count < _MAX_RETRIES
    return is_correct, points, retry_allowed


# ── 로깅 ─────────────────────────────────────────────────────────────────────

def log_attempt(
    session: ChallengeSession,
    req: AttemptRequest,
    is_correct: bool,
    points: float,
    pool_version: str,
    problem_count: int,
) -> None:
    """attempt를 JSONL에 append한다. 실패 시 경고만 출력하고 서빙은 중단하지 않는다."""
    try:
        _append_log(_build_record(session, req, is_correct, points, pool_version, problem_count))
    except Exception as e:
        logger.warning("attempt 로그 저장 실패: %s", e)


def _build_record(
    session: ChallengeSession,
    req: AttemptRequest,
    is_correct: bool,
    points: float,
    pool_version: str,
    problem_count: int,
) -> dict:
    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "challenge_id":   session.challenge_id,   # UUID (sample_id 아님)
        "sample_id":      session.sample_id,       # 피드백 MLOps 상관관계용 내부 ID
        "session_id_pfx": req.session_id[:8],      # 세션 식별 접두사 (전체 저장 금지)
        "selected_label": req.selected_label,
        "is_correct":     is_correct,
        "points":         points,                  # 0.0 / 0.5 / 1.0
        "solve_time_ms":  req.solve_time_ms,
        "retry_count":    req.retry_count,
        "pool_version":   pool_version,
        "pool_size":      problem_count,
        # 선택적 해시 (원본 IP/UA 저장 금지)
        "user_agent_hash": req.user_agent_hash,
        "ip_hash":         req.ip_hash,
        # ⛔ 미저장: final_emotion, choices, image_path, security_grade
    }


def _append_log(record: dict) -> None:
    _ATTEMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = _ATTEMPT_LOG_DIR / f"attempts_{today}.jsonl"

    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _write_lock:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
