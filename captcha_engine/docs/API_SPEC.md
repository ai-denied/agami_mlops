# API Endpoint Specification

WBS #42 산출물 중 하나. #43(검증 API 구현), 프론트엔드(46), 가이드 문서(2.2.3) 가
모두 이 명세를 참조함.

## 인증 방식 요약

| 영역 | 인증 | 헤더 |
|---|---|---|
| Public API | API key (client_key) | `X-Captcha-Client-Key: {client_key}` |
| Public API (서버 검증 호출) | API key + secret | `X-Captcha-Client-Key`, `X-Captcha-Secret-Key` |
| Admin API | Firebase ID Token | `Authorization: Bearer {firebase_id_token}` |

## Public API (기업 고객의 웹사이트가 호출)

### 1. POST /v1/challenges — 챌린지 발급

**누가 호출:** 기업 고객 웹사이트의 프론트엔드 (브라우저)
**언제:** 사용자가 캡챠 위젯을 트리거할 때

```http
POST /v1/challenges
X-Captcha-Client-Key: ck_abc123...
Content-Type: application/json
Origin: https://customer-site.com

{
  "kind": "flashlight",
  "difficulty": "medium"  // optional. 미지정 시 tenant_settings.default_difficulty
}
```

**응답 200**
```json
{
  "challenge_id": "PA9-JfGO9hXImxCMQRxuhg",
  "kind": "flashlight",
  "difficulty": "medium",
  "issued_at": "2026-04-28T10:23:45Z",
  "expires_at": "2026-04-28T10:24:40Z",
  "variant": "among_decoys",
  "target_hint": { "object_id": "key", "label": "열쇠", "emoji": "🔑" },
  "decoys": [ { "object_id": "gem", "label": "보석", "emoji": "💎", "x": 0.31, "y": 0.42 } ],
  "flashlight_radius": 0.12,
  "time_limit_sec": 45,
  "hint_after_sec": 18,
  "canvas_aspect_w": 16,
  "canvas_aspect_h": 9
}
```

**에러 401** : invalid client_key
**에러 403** : Origin 헤더가 allowed_origins 에 없음
**에러 429** : rate limit 초과

---

### 2. POST /v1/challenges/{challenge_id}/answer — 정답 제출 (사용자 → 우리 서버)

**누가 호출:** 기업 고객 웹사이트의 프론트엔드
**언제:** 사용자가 손전등으로 정답 위치를 클릭/드래그 종료한 직후

```http
POST /v1/challenges/PA9-JfGO9hXImxCMQRxuhg/answer
X-Captcha-Client-Key: ck_abc123...
Content-Type: application/json

{
  "click_x": 0.62,
  "click_y": 0.31,
  "behavioral_data": {
    "trajectory_summary": { "total_distance": 3.2, "direction_changes": 14, "avg_speed": 0.08 },
    "time_taken_ms": 7320
  }
}
```

**응답 200**
```json
{
  "captcha_token": "ct_eyJ0eW...",   // 기업 고객 backend 가 #3 으로 검증할 토큰
  "expires_in": 120                  // 토큰 유효기간 (초)
}
```

응답에 `human/bot` 판정은 노출하지 않음. 토큰만 줌. 기업 backend 가 토큰을 우리에게
검증 요청해야 진짜 결과 확인 가능 → 토큰 위변조 방지.

**에러 410** : challenge 가 이미 만료/소비됨
**에러 422** : behavioral_data 형식 오류

---

### 3. POST /v1/siteverify — 토큰 서버 검증 (기업 백엔드 → 우리 서버)

**누가 호출:** 기업 고객의 백엔드 서버 (예: 회원가입 API)
**언제:** 프론트엔드에서 받은 captcha_token 의 진위 확인이 필요할 때

reCAPTCHA / hCaptcha 와 동일 패턴.

```http
POST /v1/siteverify
Content-Type: application/x-www-form-urlencoded

secret=sk_xyz789...&token=ct_eyJ0eW...&remoteip=203.0.113.5
```

**응답 200**
```json
{
  "success": true,
  "verdict": "human",
  "confidence": 0.94,
  "challenge_ts": "2026-04-28T10:23:45Z",
  "hostname": "customer-site.com",
  "error_codes": []
}
```

**`success=false` 사유 (`error_codes`)**
- `missing-input-secret`, `invalid-input-secret`
- `missing-input-token`, `invalid-input-token`
- `timeout-or-duplicate` — 토큰이 이미 검증됐거나 만료
- `bot-detected`

## Admin API (기업 고객의 대시보드 사용자가 호출)

모든 엔드포인트는 `Authorization: Bearer {firebase_id_token}` 필요.
Firebase Admin SDK 로 검증 → `tenant_users.firebase_uid` 매칭 → tenant_id 추출.

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/admin/me` | 내 정보 + 소속 tenant |
| GET | `/admin/tenant` | tenant 정보 |
| PATCH | `/admin/tenant` | tenant 이름 등 수정 |
| GET | `/admin/api-keys` | API 키 목록 (secret 은 noop, 값 안 보임) |
| POST | `/admin/api-keys` | 새 API 키 발급 (응답에 secret 1회만 노출) |
| DELETE | `/admin/api-keys/{id}` | revoke |
| GET | `/admin/origins` | 허용 도메인 목록 |
| POST | `/admin/origins` | 도메인 추가 |
| DELETE | `/admin/origins/{id}` | 삭제 |
| GET | `/admin/settings` | tenant 설정 조회 |
| PATCH | `/admin/settings` | 난이도 / 활성 캡챠 종류 / rate limit 변경 |
| GET | `/admin/stats?range=24h` | 대시보드 통계 (인증 수, 성공률 등) |
| GET | `/admin/logs?limit=100&cursor=...` | 인증 로그 페이지네이션 |

## 응답 표준 에러 형식

모든 에러는 다음 구조를 따른다 (FastAPI 의 HTTPException 핸들러로 통일).

```json
{
  "error": {
    "code": "invalid_client_key",
    "message": "사람 친화적 메시지",
    "request_id": "req_abc123"
  }
}
```

`request_id` 는 모든 응답에 포함. 고객지원/디버깅 시 핵심.
