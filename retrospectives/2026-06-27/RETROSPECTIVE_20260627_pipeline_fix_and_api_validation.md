# 회고록 — captcha-bank 파이프라인 완주 + API 검증

**세션 기간**: 2026-06-27 (단일 세션)  
**작성**: Claude (작업 회고)  
**관련 커밋**: `863f7ba3` ~ `109a3f43`

---

## 1. 오늘의 목표

어제 세션(2026-06-26)에서 `promote_model.py` SyntaxError까지 수정하고 GitHub Actions 빌드를 기다리다 종료됐다. 오늘의 목표는:

1. 파이프라인 8단계를 끝까지 통과시켜 `pool_loaded: true` 달성
2. CronWorkflow `spec.schedule` → `spec.schedules` 수정 완료
3. 위젯팀 전달용 API 명세서 작성
4. API 동작 검증

---

## 2. 한 일

### 2-1. `_ML_PIPELINE_ROOT` parents 인덱스 버그 수정 (5개 스크립트)

파이프라인 7단계(promote-model)에서 다음 오류가 발생했다:

```
후보 폴더 없음: /workspace/model-store/captcha_bank/candidates/v1_20260626
```

5단계(package-model)에서 패키징이 성공해 `→ /workspace/model-store/captcha_bank/candidates/v1_20260626/`에 저장됐다고 출력했는데, 7단계에서 같은 경로를 찾지 못하는 것이다.

**원인**: 스크립트의 경로 계산이 잘못돼 있었다.

```
스크립트 위치: /workspace/ml-pipeline/context_emotion/captcha_bank/scripts/*.py

parents[3] → /workspace         ← 잘못됨 (컨테이너 로컬 경로)
parents[2] → /workspace/ml-pipeline ← 올바름 (PVC 마운트 경로)
```

PVC는 `/workspace/ml-pipeline/model-store`에 마운트되어 있는데, `parents[3]`을 쓰면 `/workspace/model-store`(컨테이너 로컬 임시 파일시스템)에 파일이 써진다. Pod가 교체되면 사라지는 경로라 다음 스텝 Pod에서 찾지 못하는 것이었다.

5개 스크립트를 모두 `parents[2]`로 수정했다:
- `package_model.py`
- `promote_model.py`
- `compare_candidate.py`
- `smoke_test.py`
- `run_pipeline.py`

### 2-2. 파이프라인 8/8 통과

GitHub Actions 빌드 완료 후 Argo UI에서 파이프라인 재실행.

```
✔ validate-captcha-pool    (2122 rows)
✔ build-choice-policy-report
✔ train-model              (model.joblib 3883KB)
✔ evaluate-model           (attacker_pass_rate=0.6522)
✔ package-model
✔ compare-model            (force_promote=true로 스킵)
✔ promote-model
✔ smoke-test
Status: Succeeded / Duration: 1m 21s
```

### 2-3. API `pool_loaded: true` 확인

파이프라인 완료 후 Deployment를 rollout restart해 즉시 모델 로드:

```json
{
  "status": "ok",
  "pool_loaded": true,
  "problem_count": 2122,
  "version": "v1_20260627",
  "pool_loaded_at": "2026-06-27T07:18:31.526600+00:00"
}
```

### 2-4. CronWorkflow `schedule` → `schedules` 수정

Argo v4에서 `spec.schedule` (단수 문자열)이 제거되고 `spec.schedules` (복수 배열)가 필수 필드로 변경되어 있었다. flashlight CronWorkflow 세션(2026-06-26)에서 이미 발견한 breaking change다.

`captcha-bank-pipeline.yaml`:
```yaml
# 수정 전
spec:
  schedule: "0 18 * * 1"

# 수정 후
spec:
  schedules:
  - "0 18 * * 1"
```

pool_csv 기본값도 실제 파일 경로로 함께 수정:
```
/data/context_emotion/captcha_bank/captcha_pool.csv       ← 존재하지 않음
/data/context_emotion/processed/captcha_bank_weighted_2122.csv ← 실제 경로
```

호스트 머신에서 `git stash → git pull → kubectl apply → git stash pop`으로 적용 완료:
```
workflowtemplate.argoproj.io/captcha-bank-pipeline unchanged
cronworkflow.argoproj.io/captcha-bank-pipeline-cron created
```

### 2-5. API 명세서 작성

위젯팀 전달용 명세서를 `docs/api-spec-context-emotion.md`에 작성했다.

- 플로우 다이어그램 (challenge 발급 → 풀이 제출 → 결과 처리)
- 3개 엔드포인트 상세 (요청/응답 스키마, 오류 코드)
- 재시도 처리 방법
- JavaScript pseudocode 예시
- 개인정보 주의사항 (`user_agent_hash`, `ip_hash`만 허용, 원본 금지)
- 보안 제약 (정답·내부 점수 응답 미포함)

### 2-6. 엔드포인트 실동작 검증

`kubectl exec`에 curl이 없고 터미널 63자 제약 때문에, Python urllib로 테스트 스크립트를 작성해 port-forward를 통해 검증했다.

```
H: ok True
C: 03332275
IMG: /static/images/frame_jp0abl1wirthn4ca.jpg
CH: ['anticipation', 'suffering', 'sadness', 'anger']
A: {'is_correct': False, 'retry_allowed': True}
```

3개 엔드포인트(`/health`, `/context-emotion/challenge`, `/context-emotion/attempt`) 모두 정상 동작 확인. `is_correct: false`에서 정답 레이블이 응답에 포함되지 않는 보안 제약도 확인됨.

---

## 3. 현재 상태

| 항목 | 상태 |
|---|---|
| context-emotion-api Deployment | `1/1 Running` |
| 모델 풀 | `pool_loaded: true`, 2122문항, v1_20260627 |
| CronWorkflow (초기 학습) | 매주 화요일 03:00 KST 자동 실행 설정 완료 |
| API 명세서 | `docs/api-spec-context-emotion.md` 작성 완료 |
| 엔드포인트 검증 | 3개 전부 정상 확인 |
| 피드백 파이프라인 | WorkflowTemplate + CronWorkflow 클러스터 적용 완료 |
| **외부 노출** | **미완료 — Service가 ClusterIP, Ingress는 인프라팀 관할** |

---

## 4. 좋았던 점

- **PVC 마운트 경로와 스크립트 기본 경로 불일치를 코드에서 찾아냈다**: 로그만 봤을 때는 "폴더 없음"이라 데이터 문제처럼 보였지만, 스크립트의 `_ML_PIPELINE_ROOT` 계산과 실제 마운트 경로를 대조해 원인을 정확히 짚었다. 같은 패턴이 5개 파일에 있었는데 grep으로 전수 확인하고 일괄 수정했다.
- **API 검증을 실제 요청 흐름으로 했다**: health 체크 하나만 확인하는 게 아니라 challenge 발급 → attempt 제출까지 실제 위젯 플로우와 동일한 순서로 검증했다.

## 5. 아쉬운 점 / 리스크

- **터미널 63자 제약이 세션 내내 발목을 잡았다**: 변수 분리, 스크립트 파일 분할, 블록을 나눠 붙여넣기 등 우회책을 계속 써야 했다. 근본 해결책은 터미널 너비를 늘리거나, 긴 명령어는 처음부터 파일에 작성해두는 습관이다.
- **session_id min_length=8 검증을 테스트 코드에서 놓쳤다**: `"t-001"` (5자)으로 422 오류를 맞고 수정했다. API 명세서에 제약 조건을 잘 명시해뒀으므로 위젯팀은 같은 실수를 하지 않을 것이다.
- **이미지 외부 접근 미검증**: `/static/images/frame_jp0abl1wirthn4ca.jpg`가 URL로 반환됐지만, 실제 브라우저에서 이미지가 보이려면 Ingress 또는 NodePort 설정이 필요하다. 위젯 연동 전에 확인 필요.

## 6. 피드백 파이프라인 적용 (세션 후반)

초기 학습 파이프라인과 별개로, attempt 로그 기반 자동 재학습 파이프라인을 클러스터에 적용했다.

### 발견한 문제

`captcha-bank-feedback-pipeline.yaml`이 코드에 이미 작성돼 있었고 Python 스크립트(`feedback/` 디렉토리)와 config 파일도 전부 존재했다. 단지 한 번도 클러스터에 `kubectl apply`된 적이 없었던 것이었다.

적용 전 두 가지 버그를 수정했다:

**① 중첩 마운트 문제**

`data` PVC(`/data/context_emotion`, readOnly)와 `attempt-logs` PVC(`/data/context_emotion/attempt_logs`, readOnly)가 중첩되어 있어, 컨테이너 런타임이 readOnly 경로 안에 마운트 포인트를 생성하지 못하는 문제가 발생한다. 이전 Deployment에서 겪었던 것과 동일한 패턴.

```yaml
# 수정 전
- name: attempt-logs
  mountPath: /data/context_emotion/attempt_logs  # readOnly 경로 안에 중첩

# 수정 후
- name: attempt-logs
  mountPath: /attempt-logs  # 최상위 독립 경로
```

`--log-dir` 인자 참조도 동일하게 수정 (check-trigger, aggregate-logs 두 곳).

**② CronWorkflow `schedule` → `schedules`**

Argo v4 breaking change. 이번 세션에서 세 번째 동일 수정.

### 적용 결과

```
workflowtemplate.argoproj.io/captcha-bank-feedback-pipeline created
cronworkflow.argoproj.io/captcha-bank-feedback-cron created
```

매일 KST 11:00에 자동 실행되며, 트리거 조건 미충족 시 check-trigger만 실행하고 Succeeded로 종료된다 (정상 동작).

---

## 7. 다음 액션

1. **외부 노출**: Ingress 설정은 인프라팀 관할. 확정 후 `docs/api-spec-context-emotion.md` Base URL 업데이트.
2. **피드백 파이프라인 첫 실행 확인**: attempt 로그가 쌓이면 내일 KST 11:00에 자동 실행됨. 첫 트리거 조건 충족 여부는 `feedback_trigger_policy.yaml` 설정에 따름.
