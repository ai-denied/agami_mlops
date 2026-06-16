from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.captcha_decision import MissionRound, decide_three_round_captcha


def main() -> None:
    pass_rounds = [
        MissionRound(1, "face", spoof_score=0.18, mission_pass=True, face_detected=True),
        MissionRound(2, "hand", spoof_score=0.20, mission_pass=True, face_detected=True, hand_detected=True),
        MissionRound(3, "hand", spoof_score=0.22, mission_pass=True, face_detected=True, hand_detected=True),
    ]
    retry_rounds = [
        MissionRound(1, "face", spoof_score=0.35, mission_pass=True, face_detected=True),
        MissionRound(2, "hand", spoof_score=0.40, mission_pass=True, face_detected=True, hand_detected=True),
        MissionRound(3, "hand", spoof_score=0.45, mission_pass=True, face_detected=True, hand_detected=True),
    ]
    fail_rounds = [
        MissionRound(1, "face", spoof_score=0.70, mission_pass=False, face_detected=True),
        MissionRound(2, "hand", spoof_score=0.65, mission_pass=False, face_detected=True, hand_detected=True),
        MissionRound(3, "hand", spoof_score=0.55, mission_pass=True, face_detected=True, hand_detected=True),
    ]

    cases = [
        ("pass", pass_rounds, "PASS"),
        ("retry", retry_rounds, "RETRY"),
        ("fail", fail_rounds, "FAIL"),
    ]

    for name, rounds, expected in cases:
        result = decide_three_round_captcha(rounds)
        print(name, result.decision, f"risk={result.total_risk:.3f}", result.reason)
        assert result.decision == expected

    print("dummy captcha decision test passed")


if __name__ == "__main__":
    main()
