"""
3라운드 CAPTCHA 판정 로직.

라운드 구조: 1라운드 = 얼굴 미션 + 손 미션 한 쌍.
  - spoof_score / face_detected : face-liveness API 결과 (우리 모델)
  - mission_pass / hand_detected: 손 미션 mediapipe 결과 (위젯팀)

3쌍(총 3라운드) 결과를 조합하여 PASS / RETRY / FAIL을 결정한다.

risk_band 사용 원칙
--------------------
spoof_score(연속값)를 risk에 직접 누적하지 않는다. R_live_clip 실환경 데이터에서
FRR(실사용자 오탐) 95%가 확인되었기 때문에, raw score의 작은 흔들림이 곧바로 risk
총합에 반영되면 실사용자가 face score만으로 차단될 수 있다.

대신 `OnnxFaceLivenessDetector`(api/predict)가 분류하는 risk_band
(real_safe / suspicious / spoof_detected)를 캡차 판정의 유일한 face-risk 입력으로
쓰고, band → risk 가중치를 고정값으로 매핑한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from facial_recognition.inference.onnx_face_liveness_detector import classify_spoof_risk

Decision = Literal["PASS", "RETRY", "FAIL"]
RiskBand = Literal["real_safe", "suspicious", "spoof_detected"]

# 모델 재학습 없이 운영에서 쓸 안전한 기본 threshold.
DEFAULT_LOW_THR  = 0.21
DEFAULT_HIGH_THR = 0.55

# risk_band → 고정 risk 가중치. spoof_score 연속값 대신 이 값을 사용한다.
BAND_RISK_WEIGHT: dict[RiskBand, float] = {
    "real_safe":     0.0,
    "suspicious":    0.5,
    "spoof_detected": 1.0,
}

# FAIL 처리 전 요구되는 최소 spoof_detected 라운드 수.
MIN_SPOOF_DETECTED_FOR_FAIL = 2


@dataclass
class MissionRound:
    """1라운드 = 얼굴 미션 + 손 미션 한 쌍의 결과.

    face-liveness API:
      spoof_score   — 위변조 점수 (0~1). 손 라운드에서 face 서비스 없으면 0.0 권장.
      face_detected — 얼굴 검출 여부. 미검출 시 face_missing_penalty 부과.

    mediapipe (위젯팀):
      mission_pass  — 손 미션 성공 여부.
      hand_detected — 손 검출 여부.
    """
    round_id:      int
    spoof_score:   float
    mission_pass:  bool
    face_detected: bool = True
    hand_detected: bool = False
    timeout:       bool = False
    mission_name:  str  = ""
    detail:        str  = ""
    risk_band:     Optional[RiskBand] = None


@dataclass
class CaptchaDecisionResult:
    decision:             Decision
    reason:               str
    total_risk:           float
    avg_spoof_score:      float
    failed_mission_count: int
    failed_face_count:    int
    timeout_count:        int
    spoof_detected_count: int
    risk_bands:           list[str]
    rounds:               list[MissionRound]


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
    round_result:         MissionRound,
    low_thr:              float = DEFAULT_LOW_THR,
    high_thr:             float = DEFAULT_HIGH_THR,
    mission_fail_penalty: float = 0.70,
    face_missing_penalty: float = 1.00,
    timeout_penalty:      float = 0.80,
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
    """3라운드(얼굴+손 쌍 × 3) 결과를 받아 PASS / RETRY / FAIL을 반환한다."""
    if len(rounds) != 3:
        raise ValueError("CAPTCHA requires exactly 3 round results.")

    risk_bands           = [resolve_risk_band(r, low_thr, high_thr) for r in rounds]
    spoof_detected_count = sum(b == "spoof_detected" for b in risk_bands)
    total_risk           = sum(calculate_round_risk(r, low_thr, high_thr) for r in rounds)
    spoof_scores         = [max(0.0, min(1.0, float(r.spoof_score))) for r in rounds]
    failed_mission_count = sum(not r.mission_pass  for r in rounds)
    failed_face_count    = sum(not r.face_detected for r in rounds)
    timeout_count        = sum(r.timeout           for r in rounds)
    avg_spoof_score      = float(sum(spoof_scores) / len(spoof_scores))

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

    # 손 미션 실패는 face score와 독립적인 신호이므로 즉시-FAIL 게이트에 포함.
    if failed_mission_count >= max_failed_missions:
        return _result("FAIL", "Too many hand mission failures.")
    if failed_face_count >= max_face_missing:
        return _result("FAIL", "Too many face missing rounds.")
    if timeout_count > max_timeout:
        return _result("FAIL", "Too many timeout rounds.")
    if total_risk < pass_threshold:
        return _result("PASS", "Total risk is low and all missions passed.")
    if total_risk < retry_threshold:
        return _result("RETRY", "Total risk is ambiguous.")

    # total_risk >= retry_threshold 이지만 face evidence가 충분히 반복되지 않으면
    # face score 단독 차단을 피해 RETRY를 한 번 더 준다 (FRR 95% 모델 보정).
    if spoof_detected_count < min_spoof_detected_for_fail:
        return _result(
            "RETRY",
            "Total risk is high but face evidence alone is not corroborated "
            f"({spoof_detected_count}/3 rounds spoof_detected) — retrying instead of failing.",
        )
    return _result(
        "FAIL",
        f"Total risk is high with corroborated spoof evidence "
        f"({spoof_detected_count}/3 rounds spoof_detected).",
    )
