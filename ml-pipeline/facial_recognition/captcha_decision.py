"""
3라운드 CAPTCHA 판정 로직.

얼굴 모델의 spoof_score 와 미션(face/hand) 성공 여부를 조합하여
PASS / RETRY / FAIL 을 결정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Decision    = Literal["PASS", "RETRY", "FAIL"]
MissionType = Literal["face", "hand"]


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


@dataclass
class CaptchaDecisionResult:
    decision:             Decision
    reason:               str
    total_risk:           float
    avg_spoof_score:      float
    failed_mission_count: int
    failed_face_count:    int
    timeout_count:        int
    rounds:               list[MissionRound]


def calculate_round_risk(
    round_result: MissionRound,
    mission_fail_penalty:  float = 0.70,
    face_missing_penalty:  float = 1.00,
    timeout_penalty:       float = 0.80,
) -> float:
    risk = max(0.0, min(1.0, float(round_result.spoof_score)))
    if not round_result.mission_pass:
        risk += mission_fail_penalty
    if not round_result.face_detected:
        risk += face_missing_penalty
    if round_result.timeout:
        risk += timeout_penalty
    return float(risk)


def decide_three_round_captcha(
    rounds: list[MissionRound],
    pass_threshold:       float = 1.20,
    retry_threshold:      float = 2.00,
    max_failed_missions:  int   = 2,
    max_face_missing:     int   = 2,
    max_timeout:          int   = 1,
) -> CaptchaDecisionResult:
    if len(rounds) != 3:
        raise ValueError("CAPTCHA requires exactly 3 round results.")

    has_face_mission = any(r.mission_type == "face" for r in rounds)
    has_hand_mission = any(r.mission_type == "hand" for r in rounds)

    total_risk           = sum(calculate_round_risk(r) for r in rounds)
    spoof_scores         = [max(0.0, min(1.0, float(r.spoof_score))) for r in rounds]
    failed_mission_count = sum(not r.mission_pass  for r in rounds)
    failed_face_count    = sum(not r.face_detected for r in rounds)
    timeout_count        = sum(r.timeout            for r in rounds)
    avg_spoof_score      = float(sum(spoof_scores) / len(spoof_scores))

    def _result(decision: Decision, reason: str) -> CaptchaDecisionResult:
        return CaptchaDecisionResult(
            decision=decision, reason=reason,
            total_risk=float(total_risk),
            avg_spoof_score=float(avg_spoof_score),
            failed_mission_count=int(failed_mission_count),
            failed_face_count=int(failed_face_count),
            timeout_count=int(timeout_count),
            rounds=rounds,
        )

    if not has_face_mission or not has_hand_mission:
        return _result("FAIL", "Face and hand missions must both be included.")
    if failed_mission_count >= max_failed_missions:
        return _result("FAIL", "Too many mission failures.")
    if failed_face_count >= max_face_missing:
        return _result("FAIL", "Too many face missing rounds.")
    if timeout_count > max_timeout:
        return _result("FAIL", "Too many timeout rounds.")
    if total_risk < pass_threshold:
        return _result("PASS", "Total risk is low and required missions passed.")
    if total_risk < retry_threshold:
        return _result("RETRY", "Total risk is ambiguous.")
    return _result("FAIL", "Total risk is high.")
