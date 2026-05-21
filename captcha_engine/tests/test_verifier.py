"""
captcha.verifier 단위 테스트
============================
Pydantic v2 (FlashlightSubAnswer) 의존. 로컬에서 pip install 후 실행.

1챌린지 = 3장 묶음 rework 이후 verifier는 FlashlightSubAnswer 단위로 호출됨.
"""

from __future__ import annotations

import math
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.captcha.challenge_types import FlashlightSubAnswer
from app.captcha.verifier import baseline_verdict, check_flashlight_hit


def make_answer(x: float = 0.5, y: float = 0.5, tol: float = 0.05) -> FlashlightSubAnswer:
    return FlashlightSubAnswer(
        index=0,
        correct_object_id="key",
        correct_x=x,
        correct_y=y,
        tolerance=tol,
    )


def run_tests() -> bool:
    fails: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    # 1. 정확히 정답 좌표
    a = make_answer()
    expect(check_flashlight_hit(a, 0.5, 0.5), "정확한 좌표가 hit 으로 판정 안됨")

    # 2. tolerance 내부
    a = make_answer(x=0.5, y=0.5, tol=0.05)
    # 거리 = sqrt(0.03^2 + 0.04^2) = 0.05 → 경계
    expect(check_flashlight_hit(a, 0.53, 0.54), "tolerance 경계가 miss 로 판정")

    # 3. tolerance 외부
    expect(not check_flashlight_hit(a, 0.6, 0.6), "tolerance 외부가 hit 으로 판정")

    # 4. 모서리 케이스 — 좌표 0
    a = make_answer(x=0.1, y=0.1, tol=0.05)
    expect(check_flashlight_hit(a, 0.1, 0.1), "모서리 정답에서 hit 실패")
    expect(not check_flashlight_hit(a, 0.0, 0.0), "모서리에서 거리 초과인데 hit")

    # 5. 거리 계산 정확성 (math.hypot 으로 검증)
    a = make_answer(x=0.5, y=0.5, tol=0.1)
    for cx, cy in [(0.5, 0.5), (0.6, 0.5), (0.5, 0.4), (0.55, 0.55)]:
        d = math.hypot(cx - 0.5, cy - 0.5)
        result = check_flashlight_hit(a, cx, cy)
        should_hit = d <= 0.1
        expect(result == should_hit, f"({cx},{cy}) 거리 {d:.3f}: 기대 {should_hit}, 실제 {result}")

    # 6. baseline_verdict
    expect(baseline_verdict(True) == ("human", 0.5), "hit 의 verdict 가 ('human', 0.5) 가 아님")
    expect(baseline_verdict(False) == ("bot", 0.5), "miss 의 verdict 가 ('bot', 0.5) 가 아님")

    # --- 결과 ---
    if fails:
        print(f"실패 {len(fails)}건:")
        for f in fails:
            print(f"  - {f}")
        return False
    print("test_verifier PASS — 모든 케이스 통과")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
