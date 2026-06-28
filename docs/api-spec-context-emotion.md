# Emotion CAPTCHA API 명세서

**버전**: v1.0.0  
**작성일**: 2026-06-27  
**대상**: CAPTCHA 위젯 팀

---

## 개요

감정 이미지 기반 CAPTCHA 문제 출제 및 풀이 채점 API입니다.  
사용자에게 감정 이미지를 보여주고 4지선다 중 올바른 감정을 선택하도록 합니다.

**Base URL**: `http://<api-host>:8083`  
**인터랙티브 문서**: `http://<api-host>:8083/docs` (Swagger UI)

> 배포 도메인/Ingress는 인프라팀에 별도 확인 필요.

---

## 인증

현재 별도 인증 없음. CORS는 모든 출처(`*`) 허용.

---

## 플로우

```
위젯 초기화
    │
    ▼
POST /context-emotion/challenge   ← session_id 전달
    │  challenge_id, image_url, choices, expires_at 수신
    │
    ▼
사용자에게 이미지 + 4지선다 표시
    │
    ▼
POST /context-emotion/attempt     ← challenge_id + selected_label 전달
    │  is_correct, retry_allowed 수신
    │
    ├─ is_correct=true  → CAPTCHA 통과
    └─ is_correct=false, retry_allowed=true  → 재시도 (challenge 재발급)
    └─ is_correct=false, retry_allowed=false → CAPTCHA 실패
```

---

## 엔드포인트

### 1. 서비스 상태 확인

```
GET /health
```

#### 응답

```json
{
  "status": "ok",
  "pool_loaded": true,
  "problem_count": 2122,
  "version": "v1_20260627",
  "pool_loaded_at": "2026-06-27T07:18:31.526600+00:00"
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `status` | string | `"ok"` \| `"unavailable"` |
| `pool_loaded` | boolean | 문제 풀 로드 여부 |
| `problem_count` | integer | 현재 풀 문항 수 |
| `version` | string | 현재 모델 버전 |
| `pool_loaded_at` | string \| null | 풀 로드 시각 (ISO 8601 UTC) |

`pool_loaded: false`이면 503 대신 200을 반환하며 서비스는 아직 준비 중입니다.

---

### 2. CAPTCHA 문제 출제

```
POST /context-emotion/challenge
Content-Type: application/json
```

#### 요청

```json
{
  "session_id": "user-session-abc123"
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `session_id` | string | ✓ | 위젯이 생성하는 세션 식별자 (8~128자) |

> `session_id`는 위젯/클라이언트가 직접 생성합니다. UUID v4 권장.  
> 동일 `session_id`로 최대 **10회**까지 challenge 발급 가능.

#### 응답 `200 OK`

```json
{
  "challenge_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "image_url": "/static/images/emotic/framesdb/framesdb/images/frame_kmzni9cc6gdddr8t.jpg",
  "choices": ["happiness", "anger", "fear", "sadness"],
  "expires_at": "2026-06-27T08:00:00.000000+00:00"
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `challenge_id` | string | 서버 발급 UUID. attempt 제출 시 사용 |
| `image_url` | string | 감정 이미지 상대 경로. Base URL 붙여서 사용 |
| `choices` | string[] | 4지선다 감정 레이블 (매 요청마다 순서 랜덤) |
| `expires_at` | string | challenge 만료 시각 (ISO 8601 UTC, 기본 5분) |

> **이미지 URL**: `image_url`은 같은 API 서버가 서빙합니다. 내부 서브디렉토리 경로가 포함된 형식이므로 파싱 없이 Base URL에 그대로 붙여 사용하세요.  
> 예: `http://<api-host>:8083` + `image_url` 전체

> **선택지 언어**: `choices`는 영어 레이블(`happiness`, `anger` 등 14종 중 4개)을 반환합니다. 한국어 표시가 필요하면 클라이언트에서 매핑하세요 (전체 목록은 별도 전달된 라벨 테이블 참조).

> **보안**: 정답 레이블, 내부 점수, 문제 등급은 응답에 포함되지 않습니다.

#### 오류

| 코드 | 상황 |
|---|---|
| `429 Too Many Requests` | 세션당 challenge 발급 한도 초과 (10회) |
| `503 Service Unavailable` | 풀 로드 전 요청 |

---

### 3. 풀이 제출

```
POST /context-emotion/attempt
Content-Type: application/json
```

#### 요청

```json
{
  "session_id": "user-session-abc123",
  "challenge_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "selected_label": "슬픔",
  "solve_time_ms": 4200,
  "retry_count": 0,
  "user_agent_hash": "a3f1c2d4",
  "ip_hash": "b7e3a1f2"
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `session_id` | string | ✓ | challenge 발급 시 사용한 session_id |
| `challenge_id` | string | ✓ | challenge 응답에서 받은 UUID |
| `selected_label` | string | ✓ | 사용자가 선택한 감정 레이블 |
| `solve_time_ms` | integer | ✓ | 풀이 소요 시간 (밀리초, 0~600000) |
| `retry_count` | integer | | 현재 재시도 횟수 (기본 0) |
| `user_agent_hash` | string | | User-Agent SHA256 앞 8~16자 (선택) |
| `ip_hash` | string | | 클라이언트 IP SHA256 앞 8~16자 (선택) |

> **개인정보**: 원본 IP, 원본 User-Agent를 **절대 전송하지 마세요.**  
> 필요 시 SHA256 해시 앞 16자만 전달하세요.

#### 응답 `200 OK`

```json
{
  "is_correct": false,
  "retry_allowed": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `is_correct` | boolean | 정답 여부 |
| `retry_allowed` | boolean | 재시도 가능 여부 (오답이고 재시도 한도 미초과 시 true) |

> **보안**: 점수, 정답 레이블은 응답에 포함되지 않습니다.

#### 오류

| 코드 | 상황 |
|---|---|
| `403 Forbidden` | session_id가 challenge 발급 시와 불일치 |
| `410 Gone` | challenge 만료 또는 이미 소비됨 |
| `503 Service Unavailable` | 풀 로드 전 요청 |

---

## 재시도 처리

```
attempt 결과에 따른 위젯 동작:

is_correct=true                → CAPTCHA 통과 처리
is_correct=false, retry=true   → 새 challenge 발급 후 재시도
is_correct=false, retry=false  → CAPTCHA 실패 처리 (봇 의심)
```

재시도 시 **새 challenge를 발급**받아야 합니다 (같은 challenge_id 재사용 불가).  
기본 최대 재시도 횟수: **2회** (총 3번 시도 가능).

---

## 위젯 구현 예시 (Pseudocode)

```javascript
const sessionId = crypto.randomUUID();

// 1. 문제 발급
const { challenge_id, image_url, choices, expires_at } =
  await fetch('/context-emotion/challenge', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  }).then(r => r.json());

// 2. 이미지 + 선택지 렌더링
renderCaptcha(BASE_URL + image_url, choices);

// 3. 사용자 선택 후 제출
const { is_correct, retry_allowed } =
  await fetch('/context-emotion/attempt', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      challenge_id,
      selected_label: userSelection,
      solve_time_ms: elapsedMs,
      retry_count: retryCount,
    }),
  }).then(r => r.json());

if (is_correct) {
  onCaptchaPass();
} else if (retry_allowed) {
  retryCount++;
  // 새 challenge 발급 후 재시도
} else {
  onCaptchaFail();
}
```

---

## 주의사항

1. **challenge 만료**: `expires_at` 이후 제출하면 `410 Gone`입니다. 만료 전 재발급 권장.
2. **선택지 순서**: `choices` 배열 순서는 매 요청마다 다릅니다. 서버가 반환한 순서 그대로 표시하세요.
3. **이미지 서빙**: 이미지는 동일 API 서버(`/static/images/`)에서 제공됩니다.
4. **세션 관리**: `session_id`는 클라이언트 세션 당 하나로 유지하세요. 세션당 최대 10개 challenge 발급 가능.
5. **개인정보**: `user_agent_hash`, `ip_hash`는 선택 필드이며 원본값이 아닌 해시만 허용됩니다.
