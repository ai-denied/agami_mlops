"""
3라운드 CAPTCHA 판정 로직.

얼굴 모델의 spoof_score 와 미션(face/hand) 성공 여부를 조합하여
PASS / RETRY / FAIL 을 결정한다.

risk_band 사용 원칙
--------------------
spoof_score(연속값)를 risk에 직접 누적하지 않는다. R_live_clip 실환경 데이터에서
FRR(실사용자 오탐) 95%가 확인되었기 때문에, raw score의 작은 흔들림이 곧바로 risk
총합에 반영되면 실사용자가 face score만으로 차단될 수 있다.

대신 `OnnxFaceLivenessDetector`(api/predict)가 분류하는 risk_band
(real_safe / suspicious / spoof_detected)를 캡차 판정의 유일한 face-risk 입력으로
쓰고, band → risk 가중치를 고정값으로 매핑한다. 동일한 threshold가 /predict 와
/decide 양쪽에서 쓰이도록 호출자가 detector의 low_thr/high_thr 를 그대로 전달해야
한다 (api/main.py 참고).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from facial_recognition.inference.onnx_face_liveness_detector import classify_spoof_risk

Decision    = Literal["PASS", "RETRY", "FAIL"]
MissionType = Literal["face", "hand"]
RiskBand    = Literal["real_safe", "suspicious", "spoof_detected"]

# 모델 재학습 없이 운영에서 쓸 안전한 기본 threshold.
# low_thr=0.21 은 학습 시 best_f1 으로 선택된 값(metadata.json) 그대로 유지하고,
# high_thr 는 기존 fallback(low_thr*1.3=0.273, 근거 없는 임의값) 대신 더 보수적인
# 값을 명시적으로 사용한다. detector가 로드되어 있으면 항상 detector의 값을
# 우선 사용하고, 이 값은 detector 없이 모듈을 단독 호출/테스트할 때의 fallback이다.
DEFAULT_LOW_THR  = 0.21
DEFAULT_HIGH_THR = 0.55

# risk_band → 고정 risk 가중치. spoof_score 연속값 대신 이 값을 사용한다.
BAND_RISK_WEIGHT: dict[RiskBand, float] = {
    "real_safe":      0.0,
    "suspicious":      0.5,
    "spoof_detected":  1.0,
}

# FAIL 처리 전 요구되는 최소 spoof_detected 라운드 수.
# face score 1회만으로는 FAIL 시키지 않고, 최소 2/3 라운드에서 spoof_detected가
# 반복 확인되어야 face-risk 기반 FAIL을 허용한다.
MIN_SPOOF_DETECTED_FOR_FAIL = 2


@dataclass
class MissionRound:
    round_id:      int
    mission_type:  MissionType
    spoof_score:   float
    mission_pass:  bool
    face_detected: bool = True
    timeout:       bool = False
    mission_name:  str  = ""
    hand_detected: bool = False
    detail:        str  = ""
    risk_band:     Optional[RiskBand] = None


@dataclass
class CaptchaDecisionResult:
    decision:              Decision
    reason:                str
    total_risk:            float
    avg_spoof_score:       float
    failed_mission_count:  int
    failed_face_count:     int
    timeout_count:         int
    spoof_detected_count:  int
    risk_bands:            list[str]
    rounds:                list[MissionRound]


def resolve_risk_band(
    round_result: MissionRound,
    low_thr:  float = DEFAULT_LOW_THR,
    high_thr: float = DEFAULT_HIGH_THR,
) -> RiskBand:
    """round_result.risk_band가 있으면 그대로 쓰고, 없으면 score로부터 계산한다."""
    if round_result.risk_band is not None:
        return round_result.risk_band
    return classify_spoof_risk(float(round_result.spoof_score), low_thr, high_thr)


def calculate_round_risk(
    round_result: MissionRound,
    low_thr:               float = DEFAULT_LOW_THR,
    high_thr:               float = DEFAULT_HIGH_THR,
    mission_fail_penalty:  float = 0.70,
    face_missing_penalty:  float = 1.00,
    timeout_penalty:       float = 0.80,
) -> float:
    band = resolve_risk_band(round_result, low_thr, high_thr)
    risk = BAND_RISK_WEIGHT[band]
    if not round_result.mission_pass:
        risk += mission_fail_penalty
    if not round_result.face_detected:
        risk += face_missing_penalty
    if round_result.timeout:
        risk += timeout_penalty
    return float(risk)


def decide_three_round_captcha(
    rounds: list[MissionRound],
    low_thr:                     float = DEFAULT_LOW_THR,
    high_thr:                    float = DEFAULT_HIGH_THR,
    pass_threshold:              float = 1.20,
    retry_threshold:             float = 2.00,
    max_failed_missions:         int   = 2,
    max_face_missing:            int   = 2,
    max_timeout:                 int   = 1,
    min_spoof_detected_for_fail: int   = MIN_SPOOF_DETECTED_FOR_FAIL,
) -> CaptchaDecisionResult:
    if len(rounds) != 3:
        raise ValueError("CAPTCHA requires exactly 3 round results.")

    has_face_mission = any(r.mission_type == "face" for r in rounds)
    has_hand_mission = any(r.mission_type == "hand" for r in rounds)

    risk_bands            = [resolve_risk_band(r, low_thr, high_thr) for r in rounds]
    spoof_detected_count  = sum(b == "spoof_detected" for b in risk_bands)
    total_risk            = sum(
        calculate_round_risk(r, low_thr, high_thr) for r in rounds
    )
    spoof_scores          = [max(0.0, min(1.0, float(r.spoof_score))) for r in rounds]
    # face 라운드의 mission_pass는 호출측(엔진)에서 보통 spoof_score<=threshold로
    # 결정된다 (예: captcha_engine_team_handoff 데모). 즉 "face mission 실패"는
    # 사실상 face score 재탕이다. 이를 hand mission 실패와 동일하게 취급해
    # max_failed_missions 게이트에 합산하면, face score만으로도 total_risk 경로를
    # 거치지 않고 곧바로 FAIL에 도달할 수 있다 — corroboration 요구를 무력화한다.
    # 따라서 즉시-FAIL 게이트(max_failed_missions)에는 hand mission 실패만 집계하고,
    # face mission 실패는 risk_band/total_risk 경로(아래)로만 반영한다.
    failed_hand_mission_count = sum(
        (not r.mission_pass) for r in rounds if r.mission_type == "hand"
    )
    failed_mission_count  = sum(not r.mission_pass  for r in rounds)
    failed_face_count     = sum(not r.face_detected for r in rounds)
    timeout_count         = sum(r.timeout            for r in rounds)
    avg_spoof_score       = float(sum(spoof_scores) / len(spoof_scores))

    def _result(decision: Decision, reason: str) -> CaptchaDecisionResult:
        return CaptchaDecisionResult(
            decision=decision, reason=reason,
            total_risk=float(total_risk),
            avg_spoof_score=float(avg_spoof_score),
            failed_mission_count=int(failed_mission_count),
            failed_face_count=int(failed_face_count),
            timeout_count=int(timeout_count),
            spoof_detected_count=int(spoof_detected_count),
            risk_bands=list(risk_bands),
            rounds=rounds,
        )

    # 아래 4개의 FAIL 경로는 모두 face score가 아닌 별도 신호(미션/얼굴 미검출/타임아웃)로
    # 근거가 보강된 경우에만 동작한다 — face score 단독으로는 도달하지 않는다.
    if not has_face_mission or not has_hand_mission:
        return _result("FAIL", "Face and hand missions must both be included.")
    if failed_hand_mission_count >= max_failed_missions:
        return _result("FAIL", "Too many hand mission failures.")
    if failed_face_count >= max_face_missing:
        return _result("FAIL", "Too many face missing rounds.")
    if timeout_count > max_timeout:
        return _result("FAIL", "Too many timeout rounds.")
    if total_risk < pass_threshold:
        return _result("PASS", "Total risk is low and required missions passed.")
    if total_risk < retry_threshold:
        return _result("RETRY", "Total risk is ambiguous.")

    # 여기서부터는 total_risk >= retry_threshold — 일반적으로는 FAIL 대상이지만,
    # face score가 그 원인의 전부이거나 대부분일 수 있다. R_live_clip 실환경에서
    # FRR 95%가 확인된 모델이므로, spoof_detected가 반복 확인되지 않으면(<2/3 라운드)
    # face score만으로 사용자를 차단하지 않고 RETRY로 한 번 더 기회를 준다.
    if spoof_detected_count < min_spoof_detected_for_fail:
        return _result(
            "RETRY",
            "Total risk is high but face evidence alone is not corroborated "
            "(spoof_detected in fewer than "
            f"{min_spoof_detected_for_fail}/3 rounds) — retrying instead of failing.",
        )
    return _result(
        "FAIL",
        f"Total risk is high with corroborated spoof evidence "
        f"({spoof_detected_count}/3 rounds spoof_detected).",
    )
