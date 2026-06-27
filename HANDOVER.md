# 인수인계 — context-emotion-api 모델 파이프라인

**작성일**: 2026-06-27 (업데이트)  

---

## 현재 상태 한 줄 요약

**완료.** `context-emotion-api` 모델 파이프라인 8단계 전부 통과, API `pool_loaded: true` 확인됨.

```json
{"status":"ok","pool_loaded":true,"problem_count":2122,"version":"v1_20260627","pool_loaded_at":"2026-06-27T07:18:31.526600+00:00"}
```

---

## 다음 할 일

### 1. CronWorkflow schedule 수정 (미완료)

`captcha-bank-pipeline.yaml`의 CronWorkflow가 Argo v4 미지원 필드를 사용 중:

```yaml
# 현재 (오류)
spec:
  schedule: "0 18 * * 1"

# 수정 필요
spec:
  schedules:
  - "0 18 * * 1"
```

수정 후 적용:
```bash
kubectl apply -f ml-pipeline/k8s/argo-workflows/captcha-bank-pipeline.yaml -n agami
```

### 2. health 확인 방법 (curl 없음, port-forward 사용)

```bash
kubectl port-forward -n agami svc/context-emotion-api 8083 &
sleep 2 && wget -qO- localhost:8083/health
```

---

## 시스템 구성 요약

### 네임스페이스: `agami`

| 리소스 | 이름 | 역할 |
|---|---|---|
| Deployment | `context-emotion-api` | 감정 CAPTCHA 서빙 (포트 8083) |
| ArgoCD App | `context-emotion-api` | GitOps 자동 배포 |
| PVC | `captcha-bank-model-store-pvc` | 모델 파일 저장 (10Gi) |
| PVC | `captcha-bank-data-pvc` | 이미지 데이터 (50Gi, 4.1G 적재 완료) |
| PVC | `captcha-bank-attempt-logs-pvc` | 응답 시도 로그 |
| WorkflowTemplate | `captcha-bank-pipeline` | 8단계 모델 학습·배포 파이프라인 |

### 볼륨 마운트 (Deployment)

| 볼륨 | 컨테이너 경로 | 비고 |
|---|---|---|
| `captcha-bank-model-store-pvc` | `/model-store/context_emotion/current` | subPath: `captcha_bank/current`, readOnly |
| `captcha-bank-data-pvc` | `/data/context_emotion` | readOnly |
| `captcha-bank-attempt-logs-pvc` | `/attempt-logs` | 읽기/쓰기 |

> **주의**: `readOnly`는 `volumeMounts` 레벨에만 설정한다. `volumes.persistentVolumeClaim.readOnly`에 넣으면 init container도 쓰지 못해 subPath 디렉토리 생성이 실패한다.

### 환경변수 (Deployment)

| 키 | 값 |
|---|---|
| `CAPTCHA_POOL_DIR` | `/model-store/context_emotion/current` |
| `ATTEMPT_LOG_DIR` | `/attempt-logs` |
| `IMAGE_BASE_DIR` | `/data/context_emotion` |

---

## CD 파이프라인 구조

```
코드 수정 → git push (main)
  └→ GitHub Actions: build-captcha-bank.yml
       ├─ build-and-push: Harbor에 이미지 푸시
       │    이미지: agami-captcha.cloud:8443/agami/agami-captcha-bank:<SHA8>
       └─ update-manifest: deployment.yaml 이미지 태그 수정 → git commit/push [skip ci]
            └→ ArgoCD가 변경 감지 → 자동 Rollout
```

GitHub Actions 트리거 경로:
- `ml-pipeline/context_emotion/**`
- `ml-pipeline/Dockerfile.captcha-bank`
- `ml-pipeline/requirements-captcha-bank.txt`

> `manifests/**`는 트리거 경로 제외 (`[skip ci]` + 경로 제외로 무한 루프 방지)

---

## 모델 파이프라인 단계 (WorkflowTemplate: captcha-bank-pipeline)

| 단계 | 스크립트 | 마지막 실행 결과 |
|---|---|---|
| 1. validate-captcha-pool | `validate_pool.py` | ✔ (2122 rows) |
| 2. train-model | `train_attack_model.py` | ✔ (model.joblib 3883KB) |
| 3. evaluate-model | `evaluate_model.py` | ✔ (attacker_pass_rate=0.6522) |
| 4. build-choice-policy-report | `build_choice_policy.py` | ✔ |
| 5. package-model | `package_model.py` | ✔ (버그 수정 완료) |
| 6. compare-model | `compare_model.py` | ✔ (force_promote=true로 스킵) |
| 7. promote-model | `promote_model.py` | **수정 완료, 미실행** |
| 8. cleanup | cleanup-workdir | — |

### promote-model이 쓰는 경로 (PVC 기준)

- 후보: `/workspace/model-store/captcha_bank/candidates/v1_20260626/`
- 배포 대상: `/workspace/model-store/captcha_bank/current/`  
  → Deployment의 `/model-store/context_emotion/current`에 마운트됨

---

## 주요 파일 경로

```
ml-pipeline/
  context_emotion/captcha_bank/scripts/
    package_model.py       # LabelEncoder 직렬화 수정 완료 (2026-06-26)
    promote_model.py       # global 선언 위치 수정 완료 (2026-06-26)
    train_attack_model.py
    evaluate_model.py

  k8s/argo-workflows/
    captcha-bank-pipeline.yaml       # WorkflowTemplate
    pvc-captcha-bank.yaml            # RWO PVC 정의

manifests/context-emotion-api/
  deployment.yaml          # ArgoCD가 바라보는 매니페스트 (이미지 태그 자동 갱신됨)

argocd/apps/
  context-emotion-api.yaml # ArgoCD Application 정의

.github/workflows/
  build-captcha-bank.yml   # CI/CD (build + manifest commit)
```

---

## 알려진 미완료 사항

### CronWorkflow spec.schedule → spec.schedules

`captcha-bank-pipeline`용 CronWorkflow가 Argo v4에서 `spec.schedule`(단수)을 `spec.schedules`(복수 배열)로 변경해야 하는데 아직 미수정이다.

flashlight CronWorkflow 수정 참고:
```yaml
# 수정 전 (v3.x)
spec:
  schedule: "0 18 * * 0"

# 수정 후 (v4.x)
spec:
  schedules:
  - "0 18 * * 0"
```

`ml-pipeline/k8s/argo-workflows/captcha-bank-pipeline.yaml`에서 동일하게 수정 후 `kubectl apply -f ... -n agami` 필요.

---

## 보안 제약 (변경 금지)

- challenge 응답에 정답 라벨, internal score, attacker_proxy_score 포함 금지  
- attempt 저장 시 개인정보 저장 금지; 필요 시 `user_agent_hash`, `ip_hash`만 optional 허용

---

## 참고: 데이터 워크스페이스 서버

- 호스트명: `data-prep-workspace`
- IP: `10.42.3.212`
- 데이터 경로: `/workspace/data/context_emotion/` (4.1G)
- Python3 있음, kubectl 없음
- 이 서버에서 HTTP 서버를 띄우고 파이프라인 pod에서 wget으로 데이터를 가져왔다 (이미 완료)
