# Captcha Engine — Challenge Generator (WBS #41)

손전등 캡챠의 **챌린지 유형 정의 및 동적 생성 로직** 모듈.

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `challenge_types.py` | Pydantic v2 데이터 모델 (Spec / Answer / Enum) |
| `flashlight_generator.py` | 손전등 캡챠 1회 인스턴스 생성기 |
| `_verify_logic.py` | Pydantic 없이 핵심 알고리즘만 검증하는 스크립트 |

## 의존성

- Python 3.10+
- pydantic >= 2.0

```bash
pip install "pydantic>=2.0"
```

## 빠른 사용 예

```python
from flashlight_generator import generate_flashlight_challenge
from challenge_types import Difficulty

spec, answer = generate_flashlight_challenge(Difficulty.MEDIUM)

# spec 은 클라이언트로 응답 (정답 좌표 없음)
client_payload = spec.model_dump(mode="json")

# answer 는 절대 클라이언트로 보내지 말 것. Redis 등에 저장.
server_payload = answer.model_dump(mode="json")
```

## 핵심 설계 결정

### 1. 클라이언트와 서버의 정보 비대칭
- `FlashlightChallengeSpec` → 클라이언트로 송신. 정답 좌표 미포함.
- `FlashlightChallengeAnswer` → 서버 보관 전용. **클라이언트 직렬화 금지.**

이렇게 분리해야 봇이 응답 패킷을 가로채도 정답을 알 수 없다.

### 2. 좌표는 항상 0~1 비율
캔버스 픽셀 크기와 무관. 반응형 환경(PC/모바일/태블릿)에서 모두 정확히 동작.

### 3. 암호학적 난수 사용
`random` 대신 `secrets.SystemRandom` 사용. 시드 추측 공격 방지.

### 4. 난이도 = 단일 프로필 dict
난이도 추가 시 `DIFFICULTY_PROFILES`에 키만 추가하면 끝. 기획서 2번 이미지의
"45초 제한 / 18초 후 힌트"는 `MEDIUM` 프로필에 정확히 매핑되어 있다.

### 5. 미끼-정답 최소 거리 보장
`min_separation` 제약으로 미끼와 정답이 손전등 반경 안에서 동시에 보이지 않게 함.
사용자가 우연히 정답을 맞히는 false positive 를 차단.

## 다음 작업자에게 넘기는 인터페이스 (WBS #42, #43)

### WBS #42 (백엔드 아키텍처 / DB · 캐시 스키마) 가 받을 것
- `FlashlightChallengeAnswer` Pydantic 모델
- **권장 Redis 키:** `captcha:answer:{challenge_id}`
- **권장 TTL:** `time_limit_sec + 10` (네트워크 지연 마진)
- **저장 형식:** `answer.model_dump_json()` 그대로 직렬화

### WBS #43 (서빙 / 정답·토큰 검증 API) 가 받을 것
- `generate_flashlight_challenge()` 함수: 호출 시 (spec, answer) 페어 반환
- **검증 로직:** 사용자 클릭 좌표 (x, y) 와 `answer.correct_x/y` 의 거리가
  `answer.tolerance` 이내인가?
  ```python
  hit = math.hypot(click_x - answer.correct_x, click_y - answer.correct_y) <= answer.tolerance
  ```
- **HMAC 서명:** `challenge_id` 자체에 추가 서명을 걸려면 #43 에서 `hmac.new(SECRET, challenge_id, sha256)` 으로 처리. 본 모듈은 random 토큰까지만 책임.

### WBS #44 (팀원 AI 모델 연동) 가 받을 것
- spec 에 들어 있는 `time_limit_sec`, `hint_after_sec` 을 사용해 마우스 궤적 수집
  윈도우 결정.
- 손전등 반경(`flashlight_radius`) 정보로 "탐색 영역 대비 실제 이동량" 같은 피처 계산 가능.

## 검증

Pydantic 미설치 환경에서도 알고리즘 검증 가능:

```bash
python3 _verify_logic.py
```

2000회 반복 생성하면서 다음 invariant 를 검사:
- challenge_id 고유성
- 좌표 범위 [0, 1]
- edge_padding 준수
- 난이도별 미끼 개수 일치
- 미끼-정답, 미끼-미끼 최소 거리 준수
- 정답 객체와 미끼 객체 ID 중복 없음
- time_limit 일관성

## 향후 확장 포인트

1. **객체 카탈로그 확장** — `OBJECT_CATALOG` 에 항목 추가만 하면 즉시 반영.
2. **MULTI_TARGET variant** — 같은 종류 객체 여러 개 중 색깔/크기로 구별하는 변형.
   스키마(`FlashlightVariant.MULTI_TARGET`)는 이미 정의됨, 생성 로직만 추가.
3. **Adaptive difficulty** — WBS #45 가 사용자 환경별 난이도 동적 조정을 담당하므로,
   본 모듈은 `Difficulty` 인자만 외부에서 받으면 됨. 결합 없음.
