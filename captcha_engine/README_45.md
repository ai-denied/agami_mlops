# Captcha Engine — Rate Limit + Dynamic Difficulty (WBS #45)

어뷰징 방지 레이어. 기존 #43 핸들러에 의존성으로 끼워 넣는 형태로 통합.

## 신규 / 수정 파일

| 파일 | 역할 |
|---|---|
| `app/api/policy.py` (신규) | 순수 결정 로직 (`is_rate_limited`, `decide_difficulty`). DB/Redis 의존 없음 → 단위 테스트 단순. |
| `app/api/deps.py` (수정) | `enforce_rate_limit` Depends 추가. IP / API key 양 축 1분 창. |
| `app/api/public.py` (수정) | `issue_challenge`, `submit_answer` 가 `enforce_rate_limit` 사용. 발급 시 동적 난이도 결정. 답 실패 시 IP 카운터 +1. |
| `app/cache/challenge_store.py` (수정) | `incr_failure`, `get_failure_count`, `k_fail_ip` 추가. |
| `tests/test_policy.py` (신규) | 정책 함수 단위 테스트 5건. |

## 정책 요약

### Rate Limit
- 윈도우: 1분 sliding (Redis INCR + EXPIRE 첫 호출에서 TTL set)
- IP 한도: `DEFAULT_PER_IP_LIMIT_PER_MIN = 30/min` — 단일 IP 폭주 차단
- API key 한도: `tenant_settings.rate_limit_per_min` (DB), 미설정 시 `settings.default_rate_limit_per_min`
- 초과 시: HTTP 429 + `Retry-After: 60` + `error.code = "rate_limit_exceeded"` (`scope: "ip" | "api_key"`)

### Dynamic Difficulty
- 같은 IP 의 최근 10분 누적 **실패** 횟수만 본다 (성공은 카운트 X — 정상 사용자에 패널티 X).
- 임계치:
  - 실패 0 → 요청값 또는 tenant 기본
  - 실패 1~2 → 최소 `medium` 으로 상향
  - 실패 3+ → `hard` 강제 (요청 명시값 무시)
- 명시값이 hard 인데 실패가 적은 경우는 hard 유지 (절대 내리지 않음).

### 트리거 지점
- `submit_answer` 가 좌표 비교에 실패하면 `incr_failure(ip, 600)` 호출.
- 다음 `issue_challenge` 가 `get_failure_count(ip)` 로 읽어 난이도 결정.
- 정답 성공 시에는 카운터를 0 으로 리셋하지 않음 — 봇이 의도적으로 한 번 맞춘 뒤 폭주하는 패턴을 막기 위함. TTL 자연 만료에 맡김.

## 의존성 체인 (deps.py)

```
verify_client_key → verify_origin → enforce_rate_limit
```

`issue_challenge`, `submit_answer` 는 `enforce_rate_limit` 만 Depends 하면 위 세 검증이 모두 자동 적용됨.

`siteverify` 는 사용자 트래픽이 아닌 기업 백엔드 호출이라 rate limit 미적용 (별도 정책 필요 시 추후 추가).

## 검증 완료

- 5개 정책 단위 테스트 PASS (`python3 tests/test_policy.py`)
- 수정/신규 5개 Python 파일 AST 파싱 PASS

## 검증 미완 (사용자 환경 필요)

- 실제 Redis + FastAPI 부팅 후 429 / Retry-After 응답 확인
- Postgres `tenant_settings.rate_limit_per_min` 변경이 즉시 반영되는지 (현재는 매 요청 SELECT — 캐시 도입은 별도 PR)
- 부하 테스트 (k6, locust 등) 로 1분 창 경계 동작 확인

## #44 / 이후 단계와의 인터페이스

- `verifier.baseline_verdict` 가 #44 AI 모델로 교체될 때, 모델이 `verdict="bot"` 으로 판정한 케이스도
  `incr_failure` 카운터에 합산하면 bot 의심 IP 의 난이도가 자연스럽게 올라감 (현재 코드는 `hit=False` 케이스만 카운트).
- 대시보드 (#46) 에서 `captcha:rate:*` / `captcha:fail:*` Redis 키를 읽어 실시간 어뷰징 시계열을 만들 수 있음.
