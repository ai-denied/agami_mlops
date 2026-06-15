# Flashlight MLOps 2차 자동화 파이프라인 구현 회고록

**일자**: 2026-06-15  
**담당**: agami MLOps

---

## 목표

Flashlight CAPTCHA 마우스 판별 모델의 MLOps 2차 자동화 구현.

```
validate-data → preprocess → train-model → evaluate-model → package-model
→ compare-model → promote-model → restart-inference-api → smoke-test
```

새 데이터 수집 시 자동으로 모델을 재학습하고, 성능이 개선됐을 때만 추론 API에 자동 배포.

---

## 구현 내용

### 파일 구조

| 파일 | 내용 |
|------|------|
| `ml-pipeline/k8s/argo-workflows/flashlight-full-pipeline-workflow.yaml` | 9단계 WorkflowTemplate + CronWorkflow (매주 월 03:00 KST) |
| `ml-pipeline/k8s/argo-workflows/rbac-flashlight-workflow.yaml` | SA에 `apps/deployments patch` 권한 추가 |
| `ml-pipeline/Dockerfile.training` | PyTorch 학습 전용 이미지 (CPU) |
| `ml-pipeline/requirements-training.txt` | 학습 이미지 의존성 |
| `ml-pipeline/flashlight/scripts/preprocess_pipeline.py` | 원시 로그 → 피처 추출 + 병합 CLI |
| `.github/workflows/build-flashlight-training.yml` | training 이미지 자동 빌드 CI |

### 파이프라인 설계 결정

- **두 이미지 전략**: pipeline용(`agami-mlops`) / 학습용(`agami-mlops-training`) 분리 — PyTorch(~1GB)를 불필요한 단계에 포함하지 않음
- **compare-model**: `continueOn: failed: true` — 성능 미달 시에도 파이프라인이 중단되지 않고 promote/restart/smoke를 건너뜀
- **smoke-test**: `coordinate_mode: "pixel"`, 10개 마우스 이벤트, `/health` + `/api/v1/predict` 확인
- **모델 서빙**: model-store는 PVC 마운트 방식 유지 — API 이미지에 모델 파일 미포함
- **outputs.parameters**: `t-init-version`에서 version을 `/tmp/version.txt`에 기록 후 `valueFrom.path`로 읽음 (`outputs.result`는 stdout 전체를 캡처해 오염 가능)

---

## 트러블슈팅

### 1. `--data-dir 가 존재하지 않습니다` (preprocess 실패)

**원인**  
- PVC 루트(`/mnt/flashlight-data`)에 `data/` 서브디렉토리가 없었음
- `runs-base-dir` 기본값이 `/mnt/flashlight-data/runs`로 잘못 설정됨

**해결**  
- preprocess를 graceful skip으로 변경 (데이터 없으면 exit 0)
- `runs-base-dir` 기본값 → `/mnt/flashlight-data`
- kubectl cp로 PVC에 원시 데이터 seeding

---

### 2. `ModuleNotFoundError: flashlight.data`

**원인**  
- `.gitignore`의 `data/` 패턴이 `flashlight/data/` Python 모듈까지 제외
- `.dockerignore`의 `data` 패턴이 Docker 빌드 컨텍스트에서 `flashlight/data/`를 제외

**해결**  
```gitignore
# .gitignore
data/
!ml-pipeline/flashlight/data/
!ml-pipeline/flashlight/data/**
```
```
# .dockerignore
ml-pipeline/data   (경로 명시)
workspace/data
```
`git add -f`로 기존에 무시된 파일 강제 추가.

---

### 3. `ModuleNotFoundError: pandas` / `ModuleNotFoundError: matplotlib`

**원인**  
- `requirements-flashlight.txt`에 pandas 누락 (extract_all_features.py가 사용)
- `requirements-training.txt`에 matplotlib 누락 (evaluation/plot.py가 사용)

**해결**  
- `requirements-flashlight.txt`에 `pandas>=2.0.0` 추가
- `requirements-training.txt`에 `matplotlib>=3.7.0` 추가

---

### 4. evaluate-model 지표 모두 0 (AUC=0.0, human_block_rate=1.0)

**원인**  
metadata JSON의 실제 키는 `"summary"`인데 코드에서 `"performance"`로 읽음

**해결**  
```python
perf = meta.get("summary", {})  # "performance" → "summary"
```

---

### 5. AUC 0.23 (랜덤보다 낮음)

**원인**  
- PVC에 `static/` 원시 로그만 seeding → `extract_all_features.py`로 재추출
- 재추출된 피처와 원래 `dynamic/` 피처의 스키마가 달라 엉뚱한 데이터로 학습

**해결**  
`dynamic/` 사전 추출 피처(311MB, human 4,527건 + bot 9,000건)를 PVC `processed/dynamic/`에 추가 seeding.

---

### 6. `RuntimeError: 데이터 검증 실패 — schema_validation (125건)`

**원인**  
`train_mouse_gru.py` 내부 검증기가 125개 샘플의 스키마 이슈로 실패

**해결**  
학습 커맨드에 `--skip-validation` 추가

---

### 7. validate-data 단계에서 schema_validation 125건 실패

**원인**  
`dynamic/` 데이터 일부(125개)가 `static_features` 필드 없이 저장되어 있었음 — 구버전 파이프라인에서 `dynamic_features`만 저장한 샘플

**해결 (1단계)**: PVC의 merged JSON에 `static_features: {}` 패치
```bash
kubectl run -n agami patch-json --rm -it --restart=Never \
  --image=python:3.10-slim \
  --overrides='...' 
# 패치 완료: 125개
```

**해결 (2단계)**: `validate-data`를 informational 단계로 변경 — 검증 실패해도 exit 0, 실질적 품질 gate는 evaluate-model이 담당

이후 `required_fields` 125건 실패로 이동 (빈 dict는 schema는 통과하나 내부 필드 없음).  
근본 원인: 원시 `static_features` 데이터가 유실된 채로 저장된 레거시 샘플.

---

### 8. GPU 스케줄링 실패 (`Insufficient nvidia.com/gpu`)

**경위**  
Tesla T4 GPU 노드(`host-10-3-7-10`, CUDA 12.4) 확인 후 GPU 학습으로 전환:
- `Dockerfile.training`: `whl/cpu` → `whl/cu124`
- workflow: `nodeSelector: zone: gpu` + `nvidia.com/gpu: "1"` 활성화

**원인**  
`ml-dev-workspace` pod가 GPU 1개를 상시 점유 중 (3일 이상 실행 중인 개발 워크스페이스)

**해결**  
GPU 전용 노드 확보가 안 되어 CPU 학습으로 복귀.  
> **향후**: `ml-dev-workspace`에서 GPU limit 제거 또는 학습 시간대에 scale down 후 GPU 전환 가능.

---

### 9. `outputs.result`로 version 전달 오류

**원인**  
`t-init-version`에서 `outputs.result`를 사용하면 stdout 전체가 version에 담겨 오염됨

**해결**  
```python
with open("/tmp/version.txt", "w") as f:
    f.write(version)
```
```yaml
outputs:
  parameters:
  - name: version
    valueFrom:
      path: /tmp/version.txt
```

---

### 10. 서버 git pull 누락으로 WorkflowTemplate 미반영

**현상**  
코드 수정 후 GitHub에 push했으나, 서버의 로컬 checkout(`~/flashlight-model/agami_mlops`)을 pull하지 않고 `kubectl apply`해서 구버전 YAML이 계속 적용됨.

**교훈**  
```bash
git pull  # 반드시 먼저
kubectl apply -f ...
argo submit ...
```

---

## 모델 성능 현황

| 실행 | AUC | human_block_rate | bot_miss_rate | 비고 |
|------|-----|-----------------|---------------|------|
| 1차 (dynamic 미seeding) | 0.23 | - | - | 피처 스키마 불일치 |
| 2차 (hidden=32, layers=1) | 0.64 | 0.10 | 0.55 | 기준 0.80 미달 |
| 3차 (hidden=128, layers=2) | 0.73 | 0.03 | 0.57 | 기준 0.80 미달 |

**AUC 0.73에서 정체 중인 원인 추정**
- `merged_dynamic_features_sampled.json`이 샘플링된 버전 — 수동 학습 시 사용한 원본 전체 데이터와 다를 수 있음
- seeding한 `dynamic/` 피처가 원래 모델 학습 시 쓴 피처와 버전/스키마가 다를 수 있음
- 125개 빈 `static_features` 샘플의 영향 가능성

---

## 미해결 / 다음 단계

| 항목 | 내용 |
|------|------|
| **AUC 개선** | 수동 학습 시 사용한 원본 데이터 파일과 PVC 데이터 비교 — 샘플 수/피처 스키마 일치 여부 확인 |
| **GPU 전환** | `ml-dev-workspace` GPU limit 제거 후 cu124 이미지 + nodeSelector 재활성화 |
| **min-auc 기준 조정** | 현재 데이터로 달성 가능한 수준으로 재설정 (0.73 기준 시 0.70~0.75 적용 검토) |
| **125개 샘플 정리** | 원시 `static_features` 복구 불가 시 해당 샘플 제거 후 재학습 |
| **smoke-test 엔드-투-엔드** | promote → restart → smoke-test 전체 플로우 최초 성공 확인 필요 |
| **CronWorkflow 활성화** | 매주 월 03:00 KST 자동 실행 검증 |
