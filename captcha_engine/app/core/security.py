"""
Security primitives
====================
WBS #43: 암호 처리 유틸리티 (순수 함수, 외부 의존 0).

이 모듈의 책임
- secret_key 의 해시 계산 (DB 저장용) + 상수 시간 비교
- captcha_token 의 HMAC 서명 발급 + 검증

설계 노트
- HMAC pepper / token secret 은 환경변수로 주입 (config.py).
- 모든 비교는 hmac.compare_digest 로 timing attack 방어.
- 토큰 포맷: {b64url(challenge_id)}.{b64url(hmac_16bytes)}
  단순 JWT-like. 페이로드는 challenge_id 만. 부가 정보(verdict 등)는 Redis 에서.
"""

from __future__ import annotations

import base64
import hashlib
import hmac


# ---------------------------------------------------------------------------
# Base64url helpers
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


# ---------------------------------------------------------------------------
# Tenant secret_key 해시
# ---------------------------------------------------------------------------

def hash_secret(secret: str, pepper: str) -> str:
    """
    Tenant API secret_key 의 보관용 해시.
    pepper 는 서버 환경변수 (DB compromise 시에도 secret 복원 어렵게).
    HMAC-SHA256 hex 출력 (64자).
    """
    return hmac.new(pepper.encode("utf-8"), secret.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_secret(secret: str, expected_hash: str, pepper: str) -> bool:
    """상수 시간 비교."""
    actual = hash_secret(secret, pepper)
    return hmac.compare_digest(actual, expected_hash)


# ---------------------------------------------------------------------------
# captcha_token (HMAC-signed)
# ---------------------------------------------------------------------------

_SIG_BYTES = 16  # 128-bit MAC. 토큰 길이를 적절히 유지하면서 충돌 저항 충분.


def make_captcha_token(challenge_id: str, secret: str) -> str:
    """
    challenge_id 에 HMAC 서명을 붙여 토큰 발급.
    포맷: {b64url(challenge_id)}.{b64url(hmac[:16])}
    """
    cid_bytes = challenge_id.encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), cid_bytes, hashlib.sha256).digest()[:_SIG_BYTES]
    return f"{_b64url(cid_bytes)}.{_b64url(sig)}"


def parse_captcha_token(token: str, secret: str) -> str | None:
    """
    토큰 검증 + challenge_id 추출.
    - 형식 불일치 / 서명 불일치 → None
    - 정상 → challenge_id 문자열
    """
    if not isinstance(token, str) or "." not in token:
        return None
    try:
        cid_b64, sig_b64 = token.split(".", 1)
        cid_bytes = _b64url_decode(cid_b64)
        provided_sig = _b64url_decode(sig_b64)
    except Exception:
        return None
    expected_sig = hmac.new(secret.encode("utf-8"), cid_bytes, hashlib.sha256).digest()[:_SIG_BYTES]
    if not hmac.compare_digest(provided_sig, expected_sig):
        return None
    try:
        return cid_bytes.decode("ascii")
    except UnicodeDecodeError:
        return None
