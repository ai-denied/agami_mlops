"""
core.security 단위 테스트
=========================
DB/Redis 의존 0. 순수 함수만 검증.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.core.security import (
    hash_secret,
    make_captcha_token,
    parse_captcha_token,
    verify_secret,
)


PEPPER = "test_pepper_value"
TOKEN_SECRET = "test_token_secret"


def _expect(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


def run_tests() -> bool:
    fails: list[str] = []

    # --- 1. 토큰 round-trip ---
    cid = "PA9-JfGO9hXImxCMQRxuhg"
    token = make_captcha_token(cid, TOKEN_SECRET)
    _expect("." in token, "토큰에 점 구분자가 없음", fails)
    parsed = parse_captcha_token(token, TOKEN_SECRET)
    _expect(parsed == cid, f"round-trip 실패: {parsed} != {cid}", fails)

    # --- 2. 위조 방어 ---
    _expect(parse_captcha_token(token, "wrong_secret") is None, "다른 secret 으로 검증이 통과됨", fails)

    parts = token.split(".")
    tampered_payload = parts[0] + "X." + parts[1]
    _expect(parse_captcha_token(tampered_payload, TOKEN_SECRET) is None, "페이로드 변조가 감지되지 않음", fails)

    tampered_sig = parts[0] + "." + parts[1][:-1] + "A"
    _expect(parse_captcha_token(tampered_sig, TOKEN_SECRET) is None, "서명 변조가 감지되지 않음", fails)

    # --- 3. 잘못된 형식 ---
    for bad in ["", "no_dot", ".", "a.b.c", "🙂.🙂"]:
        _expect(parse_captcha_token(bad, TOKEN_SECRET) is None, f"비정상 토큰 통과: {bad!r}", fails)

    # --- 4. secret 해시 round-trip ---
    secret = "sk_live_xyz"
    h = hash_secret(secret, PEPPER)
    _expect(len(h) == 64, f"hex 길이가 64 가 아님: {len(h)}", fails)
    _expect(verify_secret(secret, h, PEPPER), "올바른 secret 검증 실패", fails)
    _expect(not verify_secret("wrong", h, PEPPER), "잘못된 secret 이 통과됨", fails)
    _expect(not verify_secret(secret, h, "wrong_pepper"), "다른 pepper 가 통과됨", fails)

    # --- 5. 결정적 출력 (같은 입력 → 같은 해시) ---
    _expect(hash_secret(secret, PEPPER) == h, "해시 결정성 깨짐", fails)

    # --- 6. token uniqueness over different cids ---
    other_token = make_captcha_token("different_cid", TOKEN_SECRET)
    _expect(other_token != token, "다른 cid 가 같은 토큰을 생성", fails)

    # --- Result ---
    if fails:
        print(f"실패 {len(fails)}건:")
        for f in fails:
            print(f"  - {f}")
        return False
    print("test_security PASS — 모든 케이스 통과")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
