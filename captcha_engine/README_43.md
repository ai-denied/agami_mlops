# Captcha Engine — Public API (WBS #43)

3개 핵심 엔드포인트 구현 + FastAPI 앱 부트스트랩.

## 신규 파일 한 줄 요약

| 파일 | 역할 |
|---|---|
| `app/main.py` | FastAPI 앱 entry. lifespan, 미들웨어, 에러 핸들러 |
| `app/api/deps.py` | `verify_client_key`, `verify_origin`, DB/Redis 주입 |
| `app/api/public.py` | `/v1/challenges`, `/v1/.../answer`, `/v1/siteverify` |
| `app/core/security.py` | secret 해시, captcha_token HMAC 서명/검증 |
| `app/captcha/verifier.py` | 좌표 거리 비교 (#44 가 교체할 baseline) |
| `app/db/session.py` | SQLAlchemy 비동기 엔진 + 세션 팩토리 |
| `app/cache/redis_client.py` | redis-py 비동기 클라이언트 풀 |
| `app/schemas/api_dto.py` | HTTP 요청/응답 Pydantic 모델 |

## 수정 파일

- `app/cache/challenge_store.py` — `save_token`, `consume_token`, `k_token` 추가
- `app/core/config.py` — `captcha_token_secret` 환경변수 추가
- `requirements.txt` — `python-multipart` 추가 (form-encoded /siteverify 처리용)

## 로컬 실행

### 1. 의존성 설치
```bash
cd captcha_engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. PostgreSQL + Redis 띄우기 (Docker 권장)
```bash
docker run -d --name captcha-pg \
  -e POSTGRES_USER=captcha -e POSTGRES_PASSWORD=captcha -e POSTGRES_DB=captcha \
  -p 5432:5432 postgres:16

docker run -d --name captcha-redis -p 6379:6379 redis:7
```

### 3. 스키마 적용
```bash
psql -h localhost -U captcha -d captcha -f app/db/schema.sql
```

### 4. 환경변수 — `.env` 파일
```
APP_ENV=local
DATABASE_URL=postgresql+asyncpg://captcha:captcha@localhost:5432/captcha
REDIS_URL=redis://localhost:6379/0
API_KEY_HMAC_PEPPER=local-pepper-do-not-use-in-prod
CAPTCHA_TOKEN_SECRET=local-token-secret-do-not-use-in-prod
```

### 5. 시드 데이터 (테스트용 tenant + api_key 한 세트)
```sql
-- psql 에서 실행
INSERT INTO tenants (id, name) VALUES ('11111111-1111-1111-1111-111111111111', 'Test Tenant');
INSERT INTO tenant_settings (tenant_id) VALUES ('11111111-1111-1111-1111-111111111111');
INSERT INTO allowed_origins (tenant_id, origin)
  VALUES ('11111111-1111-1111-1111-111111111111', 'http://localhost:3000');

-- secret_hash 는 hash_secret('sk_test', PEPPER) 의 결과를 넣어야 함.
-- 임시: 파이썬에서 한 번 계산:
--   from app.core.security import hash_secret
--   print(hash_secret('sk_test', 'local-pepper-do-not-use-in-prod'))
INSERT INTO api_keys (tenant_id, name, client_key, secret_hash)
  VALUES ('11111111-1111-1111-1111-111111111111', 'test',
          'ck_test', '여기에_위에서_계산한_해시');
```

### 6. 실행
```bash
uvicorn app.main:app --reload --port 8000
```

### 7. 동작 확인
```bash
# 1) 챌린지 발급
curl -X POST http://localhost:8000/v1/challenges \
  -H "X-Captcha-Client-Key: ck_test" \
  -H "Content-Type: application/json" \
  -H "Origin: http://localhost:3000" \
  -d '{"kind":"flashlight","difficulty":"easy"}'
# → spec(challenge_id, target_hint 등) 반환

# 2) 정답 제출 (실제로는 좌표를 모르니 일부러 fail 시켜서 422 확인)
CID="응답에서_받은_challenge_id"
curl -X POST "http://localhost:8000/v1/challenges/$CID/answer" \
  -H "X-Captcha-Client-Key: ck_test" \
  -H "Content-Type: application/json" \
  -H "Origin: http://localhost:3000" \
  -d '{"click_x":0.0,"click_y":0.0}'
# → 422 verification_failed
```

## 단위 테스트

DB/Redis 없이 실행 가능한 순수 함수 단위 테스트:
```bash
python tests/test_security.py   # HMAC 토큰, secret 해시
python tests/test_verifier.py   # 좌표 거리 비교
python tests/test_generator.py  # #41 의 invariant
```

## 다음 단계 #44 / #45 가 받아갈 인터페이스

### #44 (AI 모델 연동)
`app/captcha/verifier.py` 의 `baseline_verdict(hit)` 를 대체.
시그니처:
```python
def score_with_ai_model(
    hit: bool,
    behavioral_data: BehavioralData | None,
    answer: FlashlightChallengeAnswer,
) -> tuple[Literal["human", "bot", "uncertain"], float]:
    ...
```
`api/public.py:submit_answer` 안의 `baseline_verdict(hit)` 호출 한 줄만 교체하면 됨.

### #45 (Rate Limit + 동적 난이도)
- Rate limit: `verify_client_key` 의존성 뒤에 미들웨어/Depends 로 끼워넣기.
  `ChallengeStore.incr_rate_counter` 가 이미 준비됨.
- 동적 난이도: `issue_challenge` 안의 `body.difficulty or settings.default_difficulty` 부분을
  "최근 N회 실패율을 보고 결정" 로직으로 교체.

## 검증 완료된 것 (이 단계)

- 전체 13개 Python 파일 AST 파싱 PASS
- 프로젝트 내부 import 그래프 정합 PASS
- `tests/test_security.py` 본 환경 실행 PASS (HMAC 토큰 round-trip / 위변조 감지 / secret 해시)

## 검증 미완 (사용자 환경에서 필요)

- FastAPI 앱이 실제로 부팅되어 endpoint 가 동작하는 것
- PostgreSQL + Redis 와의 실제 연동
- `tests/test_verifier.py`, `tests/test_generator.py` 실행 (Pydantic 필요)
