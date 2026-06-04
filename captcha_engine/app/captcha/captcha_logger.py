"""
손전등 캡챠 시도 로그 (디스크 JSON)
=====================================
로컬 분석 전용. 사람이 풀어도 실패하는 케이스의 raw 데이터를 모아 패턴 분석.
운영 인프라 변경 0: 단순 파일 I/O. 환경변수로 비활성화 가능.

파일 이름: captcha_logs/sess_<unix_ms>_<challenge_id_12자>.json
저장 실패는 fire-and-forget — 캡챠 응답 흐름을 절대 막지 않는다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 환경변수로 활성/비활성 토글 (기본 활성)
LOG_ENABLED: bool = os.getenv("CAPTCHA_LOG_ENABLED", "1") == "1"

# 저장 경로: 프로젝트 루트(captcha_engine/)의 captcha_logs/
# app/captcha/captcha_logger.py 기준 parents[2] = captcha_engine/
LOG_DIR: Path = Path(__file__).resolve().parents[2] / "captcha_logs"

# MLOps 학습 데이터(sess_*.json 호환) 저장 위치. 기존 통합 로그(LOG_DIR 루트)와 분리.
# schedule_mlops_logs 가 sub 별로 파일을 떨어뜨림 — 1 challenge 당 최대 3 파일.
MLOPS_DIR: Path = LOG_DIR / "mlops"

# create_task가 만든 코루틴을 GC로부터 보호하기 위한 set.
# asyncio docs: "Save a reference to the result of this function, to avoid a
# task disappearing mid-execution."
_pending_tasks: set[asyncio.Task[Any]] = set()


def _ensure_log_dir() -> bool:
    """디렉토리 없으면 생성. 권한 에러 시 False 반환 → 로깅 스킵."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.warning("captcha_logs 디렉토리 생성 실패: %s. 로깅 비활성.", e)
        return False


def _safe_write(path: Path, data: dict) -> None:
    """동기 파일 쓰기. 실패해도 캡챠 응답에는 영향 없게 try/except."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("captcha log 저장 실패 %s: %s", path, e)
    except (TypeError, ValueError) as e:
        # JSON 직렬화 실패 (numpy 타입 등). payload 만들 때 float() 캐스팅 잘 됐는지 확인.
        logger.warning("captcha log 직렬화 실패 %s: %s", path, e)


async def _save_attempt_log(payload: dict) -> None:
    """asyncio.to_thread 로 I/O 를 별도 스레드에서 실행 (이벤트 루프 비차단)."""
    if not _ensure_log_dir():
        return
    ts_ms = int(time.time() * 1000)
    cid_short = (payload.get("challenge_id") or "unknown")[:12]
    filename = f"sess_{ts_ms}_{cid_short}.json"
    path = LOG_DIR / filename
    await asyncio.to_thread(_safe_write, path, payload)


def schedule_attempt_log(payload: dict) -> None:
    """
    캡챠 시도 로그를 비동기로 큐잉. 호출자는 await 하지 않는다.

    asyncio.create_task 의 결과를 모듈 레벨 set 에 보관하고 done callback 에서
    제거 — Task GC 회피를 위한 표준 패턴.

    LOG_ENABLED=0 이면 no-op.
    """
    if not LOG_ENABLED:
        return
    try:
        task = asyncio.create_task(_save_attempt_log(payload))
    except RuntimeError:
        # 실행 중인 이벤트 루프가 없으면 (테스트 등) 동기로 떨어뜨림.
        # 호출 컨텍스트가 sync 인 경우만 발생. 실패해도 무시.
        if _ensure_log_dir():
            ts_ms = int(time.time() * 1000)
            cid_short = (payload.get("challenge_id") or "unknown")[:12]
            _safe_write(LOG_DIR / f"sess_{ts_ms}_{cid_short}.json", payload)
        return
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


def _ensure_mlops_dir() -> bool:
    """MLOps 디렉토리 보장. 권한 에러 시 False → 로깅 스킵."""
    try:
        MLOPS_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.warning("captcha_logs/mlops 디렉토리 생성 실패: %s. MLOps 로깅 비활성.", e)
        return False


async def _save_mlops_sessions(sessions: list[dict]) -> None:
    """학습 포맷 세션 리스트를 sub 별 파일로 저장. 각 파일 = session_id.json."""
    if not sessions or not _ensure_mlops_dir():
        return
    for s in sessions:
        sid = s.get("session_id") or "unknown"
        await asyncio.to_thread(_safe_write, MLOPS_DIR / f"{sid}.json", s)


def schedule_mlops_logs(sessions: list[dict]) -> None:
    """
    MLOps 학습 데이터(sess_*.json 호환) 를 비동기로 큐잉. 호출자는 await 하지 않는다.

    schedule_attempt_log 와 동일한 fire-and-forget 패턴 (GC-safe set, sync fallback).
    LOG_ENABLED 환경변수를 공유 — 통합 로그가 비활성이면 MLOps 도 비활성.
    sessions 빈 리스트면 no-op (예: 구버전 클라이언트로 canvas 정보 미수신).
    """
    if not LOG_ENABLED or not sessions:
        return
    try:
        task = asyncio.create_task(_save_mlops_sessions(sessions))
    except RuntimeError:
        # 실행 중인 이벤트 루프가 없으면 (테스트 등) 동기로 떨어뜨림.
        if _ensure_mlops_dir():
            for s in sessions:
                sid = s.get("session_id") or "unknown"
                _safe_write(MLOPS_DIR / f"{sid}.json", s)
        return
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


__all__ = [
    "schedule_attempt_log",
    "schedule_mlops_logs",
    "LOG_DIR",
    "MLOPS_DIR",
    "LOG_ENABLED",
]
