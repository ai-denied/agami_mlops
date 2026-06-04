"""
MLOps 학습 데이터 포맷 변환 (Phase 2a)
========================================
손전등 캡챠 통합 로그(log_payload)를 팀원 학습 데이터(sess_*.json) 호환
dict 리스트로 변환한다. 1 챌린지(3 sub) → 최대 3 세션.

I/O / 부수효과 없음 — 순수 변환만. 호출처는 변환 결과를
captcha_logger.schedule_mlops_logs 로 저장.

변환 규칙 (팀원 sess_*.json 사양 일치):
- 좌표계: 800x600 픽셀 정수
- mouse_logs.t = trajectory.t + 50ms (학습 데이터 시작점 보정, 팀원 확인)
- mouse_logs.x/y = round(trajectory.x/y * 800/canvas_width 등) — 캔버스 → 800x600 환산
- click.x/y: 이미 800x600 픽셀로 환산된 click_x_px/click_y_px 그대로
- click.t = trajectory 마지막 t + 50; trajectory 비면 None
- bbox: 이미 픽셀로 환산된 bbox_*_px 그대로 (좌상단 기준)
- duration = (trajectory 마지막 t - 첫 t); trajectory ≤ 1 이면 0
- result = "success" if coord_hit else "fail"
- image_id = image_url 파일명에서 확장자 제거
- target_object = target_label
- session_id = f"sess_{unix_ms}_{cid[:8]}_{sub_index}" — sub 별 유니크

canvas_width/canvas_height 가 None 인 sub 는 결과에서 제외
(정확한 좌표 환산 불가 → 학습 데이터 오염 방지).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

__all__ = ["to_training_sessions", "TRAINING_IMG_W", "TRAINING_IMG_H", "T_OFFSET_MS"]


# 학습 데이터 표준 캔버스 — 절대 변경 금지 (팀원 데이터셋 sess_*.json 기준)
TRAINING_IMG_W: int = 800
TRAINING_IMG_H: int = 600

# 캡챠 시작 시점과 학습 데이터 t=0 의 오프셋 (팀원 사양 확인됨)
T_OFFSET_MS: int = 50


def to_training_sessions(log_payload: dict[str, Any], unix_ms: int) -> list[dict[str, Any]]:
    """log_payload.sub_attempts 를 학습 포맷 dict 리스트로 변환.

    Args:
        log_payload: _flashlight_submit_bundle 이 만든 통합 로그 dict.
            sub_attempts[i] 안에 click_*_px, bbox_*_px, target_label, image_url,
            coord_hit, trajectory, canvas_width, canvas_height 가 있어야 함.
        unix_ms: 챌린지 단위 timestamp (모든 sub 가 공유). session_id 생성용.

    Returns:
        학습 세션 dict 리스트. 최대 len(sub_attempts) 개. canvas_width/height 가
        None 또는 falsy 인 sub 는 제외되므로 빈 리스트일 수 있음.
    """
    cid = str(log_payload.get("challenge_id") or "unknown")
    cid_short = cid[:8]

    sessions: list[dict[str, Any]] = []

    for sub in log_payload.get("sub_attempts", []) or []:
        cw = sub.get("canvas_width")
        ch = sub.get("canvas_height")
        # canvas 정보 없으면 좌표 환산 불가 → SKIP. 학습 데이터 오염 방지.
        if not cw or not ch:
            continue
        if cw <= 0 or ch <= 0:
            continue

        trajectory = sub.get("trajectory") or []
        scale_x = TRAINING_IMG_W / cw
        scale_y = TRAINING_IMG_H / ch

        mouse_logs = [
            {
                "t": int(p["t"]) + T_OFFSET_MS,
                "x": round(p["x"] * scale_x),
                "y": round(p["y"] * scale_y),
            }
            for p in trajectory
            if isinstance(p, dict) and "t" in p and "x" in p and "y" in p
        ]

        # duration: trajectory 양 끝 차이. 1개 이하면 의미 없음 → 0.
        if len(trajectory) >= 2:
            duration = int(trajectory[-1]["t"] - trajectory[0]["t"])
        else:
            duration = 0

        # click.t: trajectory 마지막 t + 50ms. trajectory 비면 None (학습 데이터 의미 없음).
        if trajectory:
            click_t: int | None = int(trajectory[-1]["t"]) + T_OFFSET_MS
        else:
            click_t = None

        # image_id: 파일명에서 확장자 제거. PurePosixPath 는 URL 경로 형식에 안전.
        image_url = str(sub.get("image_url") or "")
        image_id = PurePosixPath(image_url).stem or "unknown"

        idx = int(sub.get("index", 0))

        sessions.append({
            "session_id": f"sess_{unix_ms}_{cid_short}_{idx}",
            "image_id": image_id,
            "target_object": sub.get("target_label") or "",
            "result": "success" if sub.get("coord_hit") else "fail",
            "duration": duration,
            "click": {
                "x": sub.get("click_x_px"),
                "y": sub.get("click_y_px"),
                "t": click_t,
            },
            "bbox": {
                "x": sub.get("bbox_x_px"),
                "y": sub.get("bbox_y_px"),
                "width": sub.get("bbox_w_px"),
                "height": sub.get("bbox_h_px"),
            },
            "mouse_logs": mouse_logs,
        })

    return sessions
