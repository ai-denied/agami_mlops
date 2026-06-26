# captcha_bank MLOps 운영 가이드

## 1. 사전 조건

### Kubernetes 리소스 적용 순서

```bash
# 1. PVC 생성
kubectl apply -f ml-pipeline/k8s/argo-workflows/pvc-captcha-bank.yaml -n agami

# 2. RBAC 적용
kubectl apply -f ml-pipeline/k8s/argo-workflows/rbac-captcha-bank-workflow.yaml -n agami

# 3. WorkflowTemplate + CronWorkflow 등록
kubectl apply -f ml-pipeline/k8s/argo-workflows/captcha-bank-pipeline.yaml -n agami
```

### 이미지 빌드 (GitHub Actions 자동 빌드 전 수동 빌드 시)

```bash
docker build -f ml-pipeline/Dockerfile.captcha-bank \
             -t agami-captcha.cloud:8443/agami/agami-captcha-bank:latest \
             ./ml-pipeline

docker push agami-captcha.cloud:8443/agami/agami-captcha-bank:latest
```

---

## 2. Argo 수동 실행

### 기본 실행

```bash
argo submit --from workflowtemplate/captcha-bank-pipeline \
  -p version=v1_20260701 \
  -p pool_csv=/data/context_emotion/captcha_bank/captcha_pool.csv \
  -n agami
```

### 드라이런 (promote 단계를 실제로 실행하지 않음)

```bash
argo submit --from workflowtemplate/captcha-bank-pipeline \
  -p version=v1_20260701 \
  -p pool_csv=/data/context_emotion/captcha_bank/captcha_pool.csv \
  -p dry_run=true \
  -n agami
```

### 비교 게이트 실패 시 강제 승격

```bash
argo submit --from workflowtemplate/captcha-bank-pipeline \
  -p version=v1_20260701 \
  -p pool_csv=/data/context_emotion/captcha_bank/captcha_pool.csv \
  -p force_promote=true \
  -n agami
```

### 실행 상태 확인

```bash
# 최근 실행 목록
argo list -n agami --selector app=captcha-bank

# 특정 실행 상세
argo get <workflow-name> -n agami

# 실시간 로그 (특정 step)
argo logs <workflow-name> -c main -n agami --follow

# 모든 step 로그
argo logs <workflow-name> --follow -n agami
```

---

## 3. 로컬 실행 (로컬 개발 환경)

```bash
cd ml-pipeline

# 전체 파이프라인 (드라이런)
python -m context_emotion.captcha_bank.scripts.run_pipeline \
  --pool-csv /workspace/data/context_emotion/captcha_bank/captcha_pool.csv \
  --version  v1_20260701 \
  --dry-run

# 단계별 실행
# 1) 풀 검증
python -m context_emotion.captcha_bank.scripts.validate_captcha_pool \
  --pool-csv /workspace/data/context_emotion/captcha_bank/captcha_pool.csv

# 2) 모델 학습
python -m context_emotion.captcha_bank.training.train_attack_model \
  --pool-csv /workspace/data/context_emotion/captcha_bank/captcha_pool.csv \
  --output   /tmp/model.joblib \
  --version  v1_20260701

# 3) 보안 평가
python -m context_emotion.captcha_bank.evaluation.run_attack_eval \
  --pool-csv /workspace/data/context_emotion/captcha_bank/captcha_pool.csv \
  --model    /tmp/model.joblib \
  --output   /tmp/evaluation_result.json \
  --version  v1_20260701

# 4) 선택지 정책 리포트
python -m context_emotion.captcha_bank.build_choice_policy_report \
  --input-csv  /workspace/data/context_emotion/captcha_bank/captcha_pool.csv \
  --output-csv /tmp/captcha_pool_with_choices.csv \
  --output-md  /tmp/choice_policy_report.md

# 5) 패키징
python -m context_emotion.captcha_bank.scripts.package_model \
  --version      v1_20260701 \
  --model-joblib /tmp/model.joblib \
  --eval-json    /tmp/evaluation_result.json \
  --pool-csv     /workspace/data/context_emotion/captcha_bank/captcha_pool.csv \
  --policy-md    /tmp/choice_policy_report.md

# 6) 비교
python -m context_emotion.captcha_bank.scripts.compare_candidate \
  --version v1_20260701

# 7) 승격 (드라이런)
python -m context_emotion.captcha_bank.scripts.promote_model \
  --version v1_20260701 --dry-run

# 8) 스모크 테스트
python -m context_emotion.captcha_bank.scripts.smoke_test
```

---

## 4. 검증 명령어

### model-store 상태 확인

```bash
ls -la ml-pipeline/model-store/captcha_bank/current/
cat ml-pipeline/model-store/captcha_bank/current/metadata.json | python3 -m json.tool
cat ml-pipeline/model-store/captcha_bank/current/evaluation_result.json | python3 -m json.tool
```

### 후보 목록 확인

```bash
ls ml-pipeline/model-store/captcha_bank/candidates/
```

### 아카이브 목록 확인

```bash
ls ml-pipeline/model-store/captcha_bank/archive/
```

### 스모크 테스트 (현재 current/ 검증)

```bash
python -m context_emotion.captcha_bank.scripts.smoke_test
```

### 후보 vs 현재 비교만 실행

```bash
python -m context_emotion.captcha_bank.scripts.compare_candidate \
  --version v1_20260701
```

---

## 5. CronWorkflow 관리

### CronWorkflow 상태 확인

```bash
argo cron list -n agami
argo cron get captcha-bank-pipeline-cron -n agami
```

### CronWorkflow 일시 정지 / 재개

```bash
# 정지
argo cron suspend captcha-bank-pipeline-cron -n agami

# 재개
argo cron resume captcha-bank-pipeline-cron -n agami
```

### CronWorkflow 즉시 수동 실행 (스케줄 무관)

```bash
argo submit --from cronworkflow/captcha-bank-pipeline-cron \
  -p version=v1_20260701 \
  -n agami
```

---

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| STEP 2 실패: `학습 데이터 부족` | 풀 CSV 행 수 < 50 | `export_captcha_pool.py` 재실행, human review 완료 수 확인 |
| STEP 6 실패: `compare gate` | 어태커 pass rate가 기준 초과 | 풀 품질 개선 후 재실행, 또는 `--force-promote` |
| STEP 7 실패: `FileNotFoundError` | model-store PVC 마운트 이상 | `kubectl get pvc -n agami` 확인 |
| STEP 8 실패: 추론 오류 | model.joblib 손상 | `--skip-promote`로 패키징부터 재실행 |
| 병렬 step 파일 없음 | PVC가 ReadWriteOnce인데 동시 접근 | PVC accessModes를 ReadWriteMany로 변경 |
| `.workdir` 잔류 | exit-handler 실패 | `rm -rf model-store/captcha_bank/.workdir/{version}` |

---

## 7. 승격 정책 수정

`context_emotion/captcha_bank/config/promotion_policy.yaml` 에서 임계값을 조정한다.

```yaml
gates:
  max_attacker_pass_rate: 0.35     # 어태커 단일 문항 정답률 상한
  max_choice_policy_pass_rate: 0.10 # 3문제 챌린지 통과율 상한
  min_robust_rate: 0.65            # 최소 robust rate
  min_pool_size: 200               # 최소 풀 문항 수
  max_ambiguous_rate: 0.20         # 최대 모호 비율
```
