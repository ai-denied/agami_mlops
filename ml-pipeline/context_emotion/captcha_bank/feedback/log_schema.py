"""
attempt log 필드 정의 및 집계·품질 컬럼 상수.

attempt_logger.py 가 저장하는 JSONL 레코드와 1:1 대응한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── attempt log 원본 필드 ─────────────────────────────────────────────────────
LOG_FIELDS = [
    "timestamp",        # ISO 8601 UTC
    "challenge_id",     # UUID
    "sample_id",        # 내부 풀 ID (피드백 상관관계 핵심 키)
    "session_id_pfx",   # 세션 앞 8자리
    "selected_label",   # 사용자가 선택한 감정 레이블
    "is_correct",       # bool
    "points",           # 0.0 / 0.5 / 1.0
    "solve_time_ms",    # int
    "retry_count",      # int (>0 이면 재시도)
    "pool_version",     # 문제가 속한 풀 버전
    "pool_size",        # 당시 풀 크기
    "user_agent_hash",  # optional — sha256[:16]
    "ip_hash",          # optional — sha256[:16]
]

# ── sample_id별 집계 결과 컬럼 ────────────────────────────────────────────────
AGG_COLUMNS = [
    "sample_id",
    "attempt_count",            # 총 시도 수
    "correct_count",            # 정답 수 (is_correct=True)
    "aux_count",                # 부분 점수 수 (points=0.5)
    "human_pass_rate",          # correct_count / attempt_count
    "avg_solve_time_ms",        # 평균 풀이 시간
    "median_solve_time_ms",     # 중앙값 풀이 시간
    "retry_rate",               # retry_count > 0 인 비율
    "suspicious_rate",          # SUSPICIOUS_SOLVE_TIME_MS 미만 시도 비율
    "unique_sessions",          # 고유 session_id_pfx 수 (다양성)
    "pool_versions_seen",       # 집계 기간 내 등장한 pool_version 수
]

# ── 품질 점수 컬럼 ────────────────────────────────────────────────────────────
QUALITY_COLUMNS = [
    "sample_id",
    "quality_label",    # robust / normal / ambiguous / ux_poor / attack_exposed / confusing / insufficient_data
    "status",           # ACTIVE / REVIEW / RETIRED
    "confidence",       # 0.0~1.0 (attempt_count 기반)
    "human_pass_rate",
    "suspicious_rate",
    "avg_solve_time_ms",
    "retry_rate",
    "attempt_count",
    "score_note",       # 진단 메모
]

# ── 의심 attempt 판별 임계값 ──────────────────────────────────────────────────
# 0.8초 미만 풀이 → 이미지를 보지 않고 무작위 클릭 또는 봇 의심
SUSPICIOUS_SOLVE_TIME_MS: int = 800


@dataclass
class ProblemStats:
    """sample_id별 실시간 누적 집계 컨테이너."""
    sample_id: str
    attempt_count: int = 0
    correct_count: int = 0
    aux_count: int = 0
    solve_times: list[int] = field(default_factory=list)
    retry_counts: list[int] = field(default_factory=list)
    suspicious_count: int = 0
    session_pfxs: set[str] = field(default_factory=set)
    pool_versions: set[str] = field(default_factory=set)
