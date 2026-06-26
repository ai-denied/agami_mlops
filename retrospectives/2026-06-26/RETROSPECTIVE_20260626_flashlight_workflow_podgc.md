# 회고록 — flashlight Argo Workflow pod 생명주기 설정

**세션 기간**: 2026-06-26 (단일 세션)
**작성**: Claude (작업 회고)
**관련 커밋**: `manifests/flashlight-model-update-workflow.yaml` 수정

---

## 1. 오늘의 목표

이전 세션(2026-06-24)에서 구성한 flashlight 모델 업데이트 Argo Workflow에, 모델 비교(compare) 단계에서 승격이 실패했을 때 디버깅용 pod/로그를 일정 시간 보존하다가 자동으로 정리하는 설정을 추가하는 것이 목표였다.

---

## 2. 한 일

### podGC + ttlStrategy 추가

Argo Workflows에서 pod 생명주기를 제어하는 두 가지 설정을 WorkflowTemplate spec에 추가했다.

- **`podGC: OnWorkflowSuccess`**: 워크플로우가 성공으로 끝나면 pod를 즉시 삭제한다. 성공 케이스는 디버깅이 필요 없으므로 클러스터 리소스를 빠르게 반환한다.
- **`ttlStrategy.secondsAfterFailure: 1800`**: 워크플로우 실패 시 30분간 pod와 로그를 보존한 뒤 자동 삭제한다. 승격 실패(compare FAIL, contract 위반, smoke-test 실패 등) 원인 파악에 필요한 최소 보존 시간.
- **`ttlStrategy.secondsAfterSuccess: 600`**: 성공 시 workflow 오브젝트는 10분 뒤 삭제한다 (pod는 podGC가 이미 처리).

```yaml
podGC:
  strategy: OnWorkflowSuccess

ttlStrategy:
  secondsAfterFailure: 1800
  secondsAfterSuccess: 600
```

#### 안전성 판단 근거

변경 전 다음 항목을 점검하고 영향 없음을 확인한 뒤 진행했다.

- 운영 중인 `flashlight-inference-api` (agami namespace) 와 무관 — workflow pod 생명주기 설정은 Argo 실행 pod에만 적용됨
- PVC(`model-store`, `data`)는 pod 삭제와 독립적으로 데이터 유지
- `retryStrategy: limit: 2`는 workflow 완료 전에 동작하므로 재시도 중 podGC가 끼어들지 않음
- CronWorkflow가 주 1회 실행이라 `failedJobsHistoryLimit: 3`과 ttlStrategy 간 충돌 가능성 없음
- 승격 실패 시 `model-store/current/`는 변경되지 않으므로 pod가 30분 후 삭제돼도 롤백할 상태가 없음

### CronWorkflow `schedule` → `schedules` 수정 (Argo v4.0 대응)

`kubectl apply` 시 다음 오류가 발생했다.

```
strict decoding error: unknown field "spec.schedule"
```

`kubectl explain cronworkflow.spec`으로 확인한 결과, Argo v4.0.6에서는 `schedule` (단수 문자열)이 제거되고 `schedules` (복수 배열)가 필수 필드로 변경되어 있었다. 기존 매니페스트의 `schedule: "0 18 * * 0"`을 아래로 수정했다.

```yaml
schedules:
- "0 18 * * 0"
```

호스트 파일과 컨테이너 파일이 별도로 관리되고 있어 호스트 측은 `sed`로 직접 수정 후 적용했다.

### 클러스터 적용 결과

```
workflowtemplate.argoproj.io/flashlight-model-update   configured
cronworkflow.argoproj.io/flashlight-model-update-cron  created
```

---

## 3. pod 생명주기 정책 요약

| 시나리오 | pod 보존 | workflow 오브젝트 삭제 |
|---|---|---|
| 성공 | 즉시 삭제 (podGC) | 10분 후 |
| 실패 | **30분 보존** → 자동 삭제 | 30분 후 (pod와 동시) |

---

## 4. 좋았던 점

- **설정 전 영향 범위를 명시적으로 점검하고 진행**: podGC/ttlStrategy가 운영 Deployment와 PVC에 영향이 없음을 항목별로 확인한 뒤 적용했다. 짧은 변경이더라도 안전성 판단 근거를 남겨두는 것이 나중에 같은 질문이 생겼을 때 추적 비용을 줄여준다.
- **`kubectl explain`으로 실제 클러스터 CRD 스키마를 직접 확인**: Argo v4.0 breaking change를 추측이 아니라 클러스터에서 직접 확인하고 수정했다. 버전 문서를 찾는 것보다 빠르고 정확하다.

## 5. 아쉬운 점 / 리스크

- **컨테이너와 호스트 파일이 동기화되지 않는 문제**: 컨테이너(`/workspace/code`)에서 수정한 내용이 호스트(`~/flashlight-model/agami_mlops`)에 자동으로 반영되지 않아 같은 수정을 두 번 해야 했다. 현재 작업 흐름에서 근본적인 불편함이다. git commit/push를 통해 동기화하는 습관이 필요하다.
- **Argo v4.0 breaking change가 기존 매니페스트에 잠재해 있었음**: `schedule` 필드는 v3.6+에서 deprecated, v4.0에서 제거된 것으로 보이는데, 이번에 apply하면서 처음 발견됐다. 다른 매니페스트에도 같은 종류의 deprecated 필드가 있을 수 있으므로, 다음 apply 시 오류 가능성을 열어두는 것이 좋다.

## 6. 다음 액션 제안

1. 컨테이너와 호스트 파일 불일치 문제를 줄이기 위해, 매니페스트 변경 시 바로 commit/push하는 루틴을 확립한다.
2. 다른 Argo 매니페스트(`facial_recognition` 쪽 등)가 생기면 동일하게 `schedules` (복수) 사용 여부를 확인한다.
3. ttlStrategy 30분이 실제 디버깅에 충분한지 운영 중 판단해보고, 필요 시 `secondsAfterFailure`를 조정한다.
