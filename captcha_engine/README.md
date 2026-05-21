# Captcha Engine Backend (WBS #42)

캡챠 엔진의 백엔드 아키텍처 + DB/캐시 스키마 설계 산출물.

## 산출물 구조

```
captcha_backend/
├── README.md                 # 이 파일 (아키텍처 결정 + 핸드오프 명세)
├── API_SPEC.md              # API 엔드포인트 명세 (#43, 프론트, 가이드 문서가 참조)
├── core/
│   └── config.py            # Pydantic Settings (환경변수 기반)
├── db/
│   ├── schema.sql           # PostgreSQL DDL (운영 DB 초기화 시 실행)
│   └── models.py            # SQLAlchemy 2.0 ORM (애플리케이션이 사용)
├── cache/
│   └── challenge_store.py   # Redis 위에 얹은 챌린지 저장소 wrapper
└── schemas/                 # (다음 단계에서 채울 자리)
```

## 아키텍처 한 줄 요약

> **FastAPI 가 가운데, Redis 가 휘발성 1회용 데이터, PostgreSQL 이 영구 데이터.
> 모든 비즈니스 데이터는 `tenant_id` 로 격리.**

## 핵심 결정 1 — Hot Path / Cold Path 분리

41번에서 만든 `FlashlightChallengeAnswer` 는 30~70초만 살면 충분한 휘발성 데이터.
PostgreSQL 에 넣으면 디스크 IO 와 만료 청소 잡이 비효율을 만든다.

| 데이터 | 저장소 | TTL | 키/테이블 |
|---|---|---|---|
| Challenge 정답 | Redis | time_limit_sec + 10s | `captcha:answer:{cid}` |
| Captcha 검증 토큰 | Redis | 120s | `captcha:token:{token}` |
| Rate limit 카운터 | Redis | window | `captcha:rate:ip:{ip}:1m` |
| API key 캐시 | Redis | 300s | `captcha:apikey:{ck}` |
| Tenant / API key 원본 | PostgreSQL | 영구 | `tenants`, `api_keys` |
| 챌린지 발급 로그 | PostgreSQL | 영구 (월별 파티셔닝 권장) | `challenges` |
| 검증 결과 로그 | PostgreSQL | 영구 (월별 파티셔닝 권장) | `verifications` |

## 핵심 결정 2 — 멀티테넌시 (shared schema, tenant_id 분리)

- 모든 비즈니스 테이블에 `tenant_id` 컬럼.
- 모든 쿼리에서 `WHERE tenant_id = :current_tenant` 강제.
- 별도 schema-per-tenant 는 MVP 오버킬.
- (향후) Row-Level Security 정책 도입 가능 — DB 차원의 보호막.

## 핵심 결정 3 — reCAPTCHA 스타일 토큰 검증 흐름

End-user 가 응답을 보고 정답/오답을 직접 알면 봇이 정답을 학습한다.
그래서 3단계 흐름을 둔다:

1. **사용자 → 우리 서버**: 캡챠 제출 → 우리는 `captcha_token` 만 돌려줌 (verdict 미공개)
2. **사용자 → 기업 백엔드**: 회원가입 등 본 액션 + token 동봉
3. **기업 백엔드 → 우리 서버**: `POST /v1/siteverify` 로 토큰 검증 → `verdict, confidence` 받음

이 패턴의 이점:
- 봇이 응답 패킷을 가로채도 정답 여부를 알 수 없음
- 토큰은 1회용 + TTL 짧음 → 재사용 불가
- 기업 백엔드 secret_key 가 우리 서버까지 와야 verdict 가 풀림 → 위변조 차단

상세는 `API_SPEC.md` 참조.

## 핵심 결정 4 — 보안/프라이버시 처리

| 항목 | 결정 |
|---|---|
| `secret_key` 저장 | 절대 평문 X. SHA-256 hash + HMAC pepper 만 DB 에 저장 |
| `client_key` 저장 | 평문. 어차피 프론트에 임베드되어 공개 |
| 사용자 IP | INET 컬럼에 raw 저장. **운영에서 N일 후 해시화 또는 마스킹 권장** (기획서 1.3 보안 민감 사용자) |
| 마우스 raw 궤적 | **저장하지 않음**. 집계 피처(총 이동거리, 방향 전환 횟수 등) 만 `behavioral_summary` JSONB |
| 얼굴 데이터 | 본 단계 범위 외 (#44). 단 raw 영상 영속 저장 금지 원칙은 동일 |
| HMAC pepper | 환경변수 `API_KEY_HMAC_PEPPER`. K8s Secret 으로만. git 커밋 금지 |

## 41번과의 연결고리

41번 산출물(`flashlight_generator.py`) 의 `generate_flashlight_challenge()` 는
이 백엔드의 **`POST /v1/challenges` 핸들러 안에서 호출**된다. 흐름:

```python
# (의사코드 — #43 가 실제 구현)
@router.post("/v1/challenges")
async def issue_challenge(
    body: IssueRequest,
    api_key: ApiKey = Depends(verify_client_key),
    store: ChallengeStore = Depends(get_store),
    db: AsyncSession = Depends(get_db),
):
    # 1. 41번 generator 호출 (순수 함수)
    spec, answer = generate_flashlight_challenge(body.difficulty or api_key.tenant.settings.default_difficulty)

    # 2. 정답을 Redis 에 저장 (Hot path)
    await store.save_answer(answer)

    # 3. 발급 사실을 Postgres 에 기록 (Cold path, audit)
    db.add(Challenge(
        id=spec.challenge_id,
        tenant_id=api_key.tenant_id,
        api_key_id=api_key.id,
        kind=spec.kind.value,
        variant=spec.variant.value,
        difficulty=spec.difficulty.value,
        issued_at=spec.issued_at,
        expires_at=spec.expires_at,
        requester_ip=request.client.host,
        requester_origin=request.headers.get("origin"),
    ))
    await db.commit()

    # 4. spec(정답 좌표 없음)만 클라이언트에 반환
    return spec
```

## 다음 단계 #43 가 받을 인터페이스

`#43 — 문제 서빙 및 정답/토큰 검증 API` 가 본 산출물 위에 구현해야 할 것:

1. **챌린지 발급 핸들러** — 위 의사코드. `ChallengeStore.save_answer` + `Challenge` INSERT
2. **정답 제출 핸들러** — `ChallengeStore.consume_answer` 로 GETDEL → 좌표 거리 비교 →
   `Verification` INSERT → `captcha_token` 발급 (HMAC 서명) → Redis 에 토큰 저장
3. **siteverify 핸들러** — secret_key 검증 + 토큰 검증 + verdict 반환

거리 비교 로직 예시 (이건 #43 영역):
```python
import math
hit = math.hypot(click_x - answer.correct_x, click_y - answer.correct_y) <= answer.tolerance
```

## 다음 단계 #45 (어뷰징 방지) 가 받을 인터페이스

`ChallengeStore.incr_rate_counter(key, window_seconds)` 로 atomic counter 증가.
미들웨어에서 호출하여 `int > limit` 이면 429 응답.

## 다음 단계 #44 (AI 모델 연동) 가 받을 인터페이스

`Verification.behavioral_summary` 컬럼이 JSONB 임. AI 모델이 산출한 피처/스코어를
이 컬럼에 자유롭게 저장 가능. 스키마 변경 없이 확장 가능.

## 환경 설정

```bash
# .env 예시 (개발 환경)
APP_ENV=local
DATABASE_URL=postgresql+asyncpg://captcha:captcha@localhost:5432/captcha
REDIS_URL=redis://localhost:6379/0
API_KEY_HMAC_PEPPER=local-dev-only-do-not-use-in-prod
```

운영(K8s):
- `DATABASE_URL` → ConfigMap (호스트/DB명 등 비밀 아닌 정보)
- `API_KEY_HMAC_PEPPER`, DB 비밀번호 → Secret

## 의존성

```
sqlalchemy>=2.0
asyncpg>=0.29
redis>=5.0
pydantic>=2.0
pydantic-settings>=2.0
fastapi>=0.110
firebase-admin>=6.0   # (#43 가 설치)
```

> `pydantic-settings` 는 Pydantic v2 부터 별도 패키지로 분리됨. 본 환경에서는
> 네트워크 차단으로 실제 import 검증을 못 했음. 로컬에서 `pip install` 후 사용.

## 검증 (이 단계에서 한 것)

- Python 파일 3개 AST 파싱 성공 (문법 정합성)
- SQL 정적 검증: 테이블 7개 / FK 8건 모두 정합 / PRIMARY KEY 7개 / CHECK 7건 / INDEX 8개
- 41번 산출물과의 인터페이스 (challenge_id 형태, 좌표 0~1 비율, TTL 계산식) 일관성 확인

## 불확실/검증 미완 사항

- **실제 DB 연결 검증 X** — 본 환경에서 PostgreSQL 인스턴스를 띄울 수 없어 schema.sql 의
  실행 가능성은 정적 검토만 했음. 로컬에서 `psql -f schema.sql` 실행으로 검증 권장.
- **redis-py async API 시그니처** — `redis.asyncio` 모듈 사용. 이는 redis-py 4.2+
  부터 안정 제공되는 API 이지만, 5.0+ 사용 권장 (로컬 검증 필요).
- **SQLAlchemy 2.0 의 `Mapped[Any]` for INET/JSONB** — Python 타입 힌트가 약간 느슨함.
  엄격한 타입은 `ipaddress.IPv4Address` 같은 변환 레이어 추가하면 개선 가능 (선택).
