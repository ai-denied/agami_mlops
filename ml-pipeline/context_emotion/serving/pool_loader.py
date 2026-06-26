"""
current/ CAPTCHA 풀 로더.

참조 경로:
  $CAPTCHA_POOL_DIR (기본: /model-store/context_emotion/current)

보안 원칙:
  - current/ 만 읽는다.
  - candidates/ 와 archive/ 는 이 모듈에서 절대 참조하지 않는다.
  - 모델 메타데이터(security_grade, attack_hardness)는 PoolState 내부에만 보관하며
    API 응답에 노출되지 않는다.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── 설정 ─────────────────────────────────────────────────────────────────────

_POOL_DIR = Path(os.getenv("CAPTCHA_POOL_DIR", "/model-store/context_emotion/current"))
_RELOAD_INTERVAL = int(os.getenv("POOL_RELOAD_INTERVAL_SEC", "300"))  # 5분

# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class PoolState:
    rows: list[dict]           # 전체 문항 (내부용 — 직렬화 금지)
    version: str
    problem_count: int
    loaded_at: datetime
    metadata: dict = field(default_factory=dict)
    _index: dict[str, dict] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {r["sample_id"]: r for r in self.rows if "sample_id" in r}

    def get_by_sample_id(self, sample_id: str) -> Optional[dict]:
        return self._index.get(sample_id)


# ── 모듈 수준 상태 ────────────────────────────────────────────────────────────

_state: Optional[PoolState] = None
_state_lock = asyncio.Lock()


def get_pool() -> Optional[PoolState]:
    """현재 캐시된 PoolState를 반환한다. 미로드 시 None."""
    return _state


def is_loaded() -> bool:
    return _state is not None


# ── 로드 ─────────────────────────────────────────────────────────────────────

async def reload() -> Optional[PoolState]:
    """current/ 에서 풀을 (재)로드한다. 실패 시 기존 상태 유지."""
    try:
        state = _load_from_disk(_POOL_DIR)
    except Exception as e:
        logger.error("풀 로드 실패 — 기존 상태 유지: %s", e)
        return _state

    await _set_state(state)
    logger.info("풀 로드 완료: version=%s, problem_count=%d", state.version, state.problem_count)
    return state


async def _set_state(new: PoolState) -> None:
    global _state
    async with _state_lock:
        _state = new


def _load_from_disk(pool_dir: Path) -> PoolState:
    """동기 디스크 읽기 (asyncio event loop 바깥에서도 호출 가능)."""
    pool_csv  = pool_dir / "captcha_pool.csv"
    meta_json = pool_dir / "metadata.json"

    if not pool_csv.exists():
        raise FileNotFoundError(f"captcha_pool.csv 없음: {pool_csv}")

    # metadata.json (없으면 빈 dict)
    metadata: dict = {}
    if meta_json.exists():
        with meta_json.open(encoding="utf-8") as f:
            metadata = json.load(f)

    version = metadata.get("version", "unknown")

    # CSV 로드 — 내부 필드 전체 보존, API 응답 시 필터링
    with pool_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # final_emotion 없는 행 제거 (불완전 데이터 방어)
    from context_emotion.captcha_bank.choice_generation import EMOTIONS
    valid_rows = [r for r in rows if r.get("final_emotion", "") in EMOTIONS]

    if not valid_rows:
        raise ValueError(f"유효 문항 없음 (total={len(rows)})")

    return PoolState(
        rows=valid_rows,
        version=version,
        problem_count=len(valid_rows),
        loaded_at=datetime.now(timezone.utc),
        metadata=metadata,
    )


# ── 백그라운드 자동 재로드 ────────────────────────────────────────────────────

async def background_reload_loop() -> None:
    """_RELOAD_INTERVAL 초마다 current/ 재로드 (version / problem_count 변경 감지)."""
    while True:
        await asyncio.sleep(_RELOAD_INTERVAL)
        try:
            new_state = _load_from_disk(_POOL_DIR)
            old = _state
            if (
                old is None
                or new_state.version != old.version
                or new_state.problem_count != old.problem_count
            ):
                await _set_state(new_state)
                logger.info(
                    "풀 갱신: %s → %s (%d→%d 문항)",
                    old.version if old else "none",
                    new_state.version,
                    old.problem_count if old else 0,
                    new_state.problem_count,
                )
        except Exception as e:
            logger.warning("백그라운드 재로드 실패: %s", e)
