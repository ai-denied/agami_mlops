# agami Captcha — 임베드 가이드

> 부모 페이지에서 `iframe` 으로 캡챠 위젯을 로드하고, 풀이 결과를 `postMessage` 로 수신하는 표준 통합 방법.
>
> 본 문서의 코드는 모두 `captcha_engine/test-embed.html` 에서 실제 동작이 검증된 패턴을 기반으로 한다.

---

## 1. 임베드 한 줄 요약

```html
<iframe
  src="http://YOUR-HOST/widget/embed?kind=flashlight&difficulty=easy"
  width="900" height="700"
  allow="camera; microphone"
  referrerpolicy="no-referrer-when-downgrade">
</iframe>
```

위 한 줄 + 아래 `postMessage` 수신 스크립트만 부모 페이지에 추가하면 끝이다.

---

## 2. URL 쿼리

| 파라미터 | 값 | 기본값 | 비고 |
|---|---|---|---|
| `kind` | `flashlight` / `face_mission` / `context_inference` | `flashlight` | 미지정/잘못된 값이면 `flashlight` 로 fallback |
| `difficulty` | `easy` / `normal` / `medium` / `hard` | `easy` | `normal` 은 별칭 — 백엔드 enum `medium` 으로 자동 매핑 |

예시:
- `http://HOST/widget/embed?kind=flashlight&difficulty=hard`
- `http://HOST/widget/embed?kind=face_mission`
- `http://HOST/widget/embed?kind=context_inference&difficulty=easy`

---

## 3. 부모 페이지 — postMessage 수신

캡챠가 성공/실패 시 위젯이 부모창으로 `agami-result` 메시지를 **단 한 번** 발신한다.

```html
<script>
  window.addEventListener('message', (e) => {
    // type 검사 — 다른 origin/스크립트의 메시지와 섞이지 않게 first guard.
    if (!e.data || e.data.type !== 'agami-result') return;

    if (e.data.success) {
      console.log('[agami] PASS', {
        challengeId: e.data.challengeId,
        challengeType: e.data.challengeType,
        captchaToken: e.data.captchaToken,
      });

      // captchaToken 을 기업 백엔드 /v1/siteverify 로 전달 (서버-서버 호출):
      //   POST /v1/siteverify
      //   Form: secret=<sk_xxx>&token=<captchaToken>
      //   응답: { success: bool, verdict: 'human'|'bot', confidence: 0..1, ... }
      //
      // 실 사용자 검증은 반드시 서버에서 siteverify 로 확인한다.
      // 클라이언트의 success=true 만 신뢰하면 우회 가능.
    } else {
      console.log('[agami] FAIL — 사용자에게 재시도 안내', {
        challengeId: e.data.challengeId,
        challengeType: e.data.challengeType,
      });
      // 부모 앱에서 직접 iframe 을 reload 하면 새 챌린지로 재시작 가능:
      //   document.getElementById('captcha-frame').src += '';
    }
  });
</script>
```

### 페이로드 스키마

`agami-result` 단일 type — `success` 필드로 성공/실패 분기.

```jsonc
{
  "type":          "agami-result",
  "success":       true,                       // boolean (필수)
  "challengeId":   "UuYlskFBRyp73pda-aYDWA",   // string  (필수, 추적/로그용)
  "challengeType": "flashlight",               // 'flashlight' | 'face_mission' | 'context_inference'
  "captchaToken":  "eyJhbGciOi..."             // success=true 일 때만 채워짐. fail 시 null.
}
```

| 필드 | 타입 | 비고 |
|---|---|---|
| `type` | `'agami-result'` | 고정. 부모는 이 값으로만 위젯 메시지를 식별 |
| `success` | `boolean` | 모든 종료 사유 (정답/오답/타임아웃/네트워크 에러) 를 통합 |
| `challengeId` | `string` | 챌린지 발급 시 백엔드가 부여한 ID. 로그 추적용 |
| `challengeType` | `string` | URL 쿼리의 `kind` 와 동일 |
| `captchaToken` | `string \| null` | HMAC 서명된 1회용 토큰. 서버 siteverify 의 input |

실패 사유는 부모로 전달되지 않는다 (위젯 내부에서 사용자에게만 표시). 부모는 단순히 "통과/실패" 로만 판단하고, 자세한 진단은 서버 로그에서 확인.

---

## 4. 토큰 검증 (서버-서버)

부모 앱 백엔드가 `captchaToken` 을 받으면 반드시 캡챠 엔진의 `siteverify` 로 검증한다.

```bash
curl -X POST http://CAPTCHA-HOST/v1/siteverify \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "secret=sk_xxx&token=eyJhbGciOi..."
```

응답:
```jsonc
{
  "success":      true,
  "verdict":      "human",            // 'human' | 'bot' | 'uncertain'
  "confidence":   0.12,                // 봇 위험도. 낮을수록 사람.
  "challenge_ts": "2026-05-22T01:23:45Z",
  "hostname":     "https://parent.example.com",
  "error_codes":  []
}
```

- `success: false` 또는 `verdict !== 'human'` 이면 부모 앱이 사용자를 차단/재시도.
- `secret` 은 캡챠 엔진 발급용 secret key (sk_xxx). 절대 클라이언트에 노출 X.
- token 은 1회용 (siteverify 호출 즉시 redis 에서 폐기).

---

## 5. 보안 / 환경 메모

| 항목 | 설명 |
|---|---|
| **iframe `allow="camera; microphone"`** | `face_mission` 캡챠는 카메라 권한 필요. iframe 에서 사용자 카메라 접근하려면 부모가 명시적으로 권한 위임 |
| **`file://` 프로토콜 불가** | Chrome/Safari 보안 정책상 `file://` 페이지에서 iframe 카메라 사용 막힘. **반드시 `http(s)://` 서버**로 부모 페이지 띄울 것 (`python3 -m http.server` 도 OK) |
| **postMessage origin 검증** | 위젯은 `'*'` 으로 부모 후보 모두에 발신. 부모는 수신 시 `e.origin` 을 화이트리스트 검사 권장 (예: `if (e.origin !== 'https://captcha.agami.kr') return;`) |
| **CORS** | 캡챠 엔진은 `allow_origins=["*"]` + `allow_credentials=false` 임시 설정. 운영 안정화 시 부모 도메인 화이트리스트로 좁힐 것 |
| **CDN 의존** | `face_mission` 은 `cdn.jsdelivr.net` 의 MediaPipe wasm 다운로드 필요. 사내망 격리 환경이면 추가 작업 |
| **첫 로딩 지연** | 첫 진입 시 widget bundle ~280KB + face_mission 은 MediaPipe wasm/data 수 MB. 1~3초 표시. 두 번째부터 브라우저 캐시 |

---

## 6. 동작 확인 — 가장 빠른 테스트

캡챠 엔진 컨테이너에 `test-embed.html` 이 포함돼 있다 (`captcha_engine/test-embed.html`). HTTP 서버로 띄워서 즉시 테스트 가능:

```bash
cd captcha_engine
python3 -m http.server 8001
# 브라우저: http://localhost:8001/test-embed.html
```

상단 5개 버튼으로 `kind` × `difficulty` 시나리오 전환, 우측 패널에 `postMessage` 수신 페이로드 실시간 표시 — 부모 페이지 작성 전 동작 확인용으로 그대로 활용.

---

## 7. 트러블슈팅 (자주 발생)

| 증상 | 원인 후보 | 대응 |
|---|---|---|
| 콘솔 `origin_not_allowed` 403 | 캡챠 엔진 DB `allowed_origins` 에 부모 도메인 미등록 | 운영팀에 도메인 추가 요청 (`/Users/...captcha_engine/app/db/seed.sql` 패턴) |
| 콘솔 `/captcha/v1/... 404` | 캡챠 엔진 ingress 룰 미설정 또는 prefix mismatch | ingress 에 `/captcha` → captcha-api 룰 추가. 또는 `/v1` 직접 호출 |
| face_mission 비디오 안 뜸 | `cdn.jsdelivr.net` 차단 / `allow="camera"` 누락 / 다른 앱이 카메라 점유 | Network 탭에서 jsdelivr 응답 확인. iframe `allow` 속성 확인 |
| 캡챠는 통과했는데 부모에 메시지 안 옴 | 부모가 `e.data.type` 검사를 잘못함 (`'agami-captcha-success'` 등 추측) | 정확히 `e.data.type === 'agami-result'` 사용 |
| iframe 안 캡챠가 잘림 / 스크롤 발생 | iframe `height` 부족 | 데스크탑은 `height="700"`, 모바일 대응은 `height="auto"` + 부모에서 `Window.postMessage` 로 높이 동기화 (추후 작업) |

---

## 8. 최소 통합 예제 (복붙용)

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>agami captcha demo</title>
</head>
<body>
  <h1>회원가입 — 사람 인증</h1>

  <iframe
    id="agami-frame"
    src="http://YOUR-CAPTCHA-HOST/widget/embed?kind=flashlight&difficulty=easy"
    width="900" height="700"
    style="border:0;"
    allow="camera; microphone"
    referrerpolicy="no-referrer-when-downgrade">
  </iframe>

  <button id="submit-btn" disabled>가입 완료</button>

  <script>
    let captchaToken = null;

    window.addEventListener('message', (e) => {
      if (!e.data || e.data.type !== 'agami-result') return;

      // 운영에선 origin 검증 추가:
      // if (e.origin !== 'https://captcha.agami.kr') return;

      if (e.data.success && e.data.captchaToken) {
        captchaToken = e.data.captchaToken;
        document.getElementById('submit-btn').disabled = false;
      } else {
        alert('캡챠 인증에 실패했습니다. 다시 시도해 주세요.');
        document.getElementById('agami-frame').src += '';  // iframe reload
      }
    });

    document.getElementById('submit-btn').addEventListener('click', async () => {
      // 회원가입 API 호출 시 captchaToken 동봉.
      // 서버는 받은 토큰을 /v1/siteverify 로 검증한 뒤 회원가입 진행.
      await fetch('/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: '...', password: '...', captchaToken }),
      });
    });
  </script>
</body>
</html>
```

---

## 변경 이력 메모

- 캡챠 엔진의 frontend API prefix 는 `/captcha` (예: `/captcha/v1/challenges`). 기존 `/api` 는 카카오 로그인 백엔드와 충돌해서 별도 prefix 로 분리.
- 캡챠 이미지 정적 서빙: `/captcha/static/captcha_images/...` 또는 `/static/captcha_images/...` (양쪽 dual mount).
- siteverify 직접 호출은 prefix 없는 `/v1/siteverify` 도 호환 유지.
