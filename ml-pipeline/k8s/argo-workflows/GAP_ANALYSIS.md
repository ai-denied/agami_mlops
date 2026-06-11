# flashlight Argo Workflows Gap Analysis

**작성일**: 2026-06-10  
**분석 대상**: `flashlight-model-update-workflow.yaml` (1차 단일 step 구현)  
**스코프**: package → compare → promote 파이프라인. 캡챠 엔진 연동 제외.

---

## 분석 방법

실제 코드를 직접 검사해 각 항목의 현재 상태를 판단했다.

| 검사 항목 | 방법 |
|---|---|
| 경로 계산 | 모든 스크립트의 `_SCRIPT_DIR`, `_ML_PIPELINE_ROOT` 추출 후 컨테이너 경로와 대조 |
| 의존성 | 현재 설치된 패키지 목록 확인 |
| Docker 이미지 | Dockerfile, requirements.txt 존재 여부 확인 |
| GPU 코드 | `common/device.py` 및 `train_mouse_gru.py` 분석 |
| 환경변수 | `os.environ`, `os.getenv` 사용 여부 전체 grep |
| k8s 리소스 | 기존 YAML 매니페스트 존재 여부 확인 |

---

## 현재 경로 계산 구조

모든 pipeline 스크립트가 **`__file__` 기준 상대 계산** 방식을 사용한다.

```python
_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CURRENT_DIR      = os.path.join(_ML_PIPELINE_ROOT, "model-store", "flashlight", "current")
```

컨테이너 내 코드 위치가 `/workspace/ml-pipeline/flashlight/scripts/` 이면 경로가 아래처럼 결정된다.

```
_ML_PIPELINE_ROOT → /workspace/ml-pipeline
_CURRENT_DIR      → /workspace/ml-pipeline/model-store/flashlight/current
_CANDIDATES_DIR   → /workspace/ml-pipeline/model-store/flashlight/candidates
```

따라서 **model-store PVC는 반드시 `/workspace/ml-pipeline/model-store`에 마운트**해야 한다.  
YAML에 이미 이 경로로 기재되어 있다. PVC 생성 시 이 경로를 반드시 맞춰야 한다.

---

## [READY] — 이미 구현 완료

### 파이프라인 코드

| 항목 | 파일 | 상태 |
|---|---|---|
| 통합 파이프라인 | `flashlight/scripts/run_model_update_pipeline.py` | ✅ |
| 패키징 | `flashlight/scripts/package_for_captcha_engine.py` | ✅ |
| 성능 비교 | `flashlight/scripts/compare_candidate.py` | ✅ |
| 승격 (rollback 포함) | `flashlight/scripts/promote_model.py` | ✅ |
| 테스트 스크립트 | `flashlight/scripts/test_engine_integration.py` | ✅ |

### model-store 구조

| 항목 | 경로 | 상태 |
|---|---|---|
| current/ | `model-store/flashlight/current/` | ✅ (v5_20260610 운영 중) |
| candidates/ | `model-store/flashlight/candidates/` | ✅ |
| archive/ | `model-store/flashlight/archive/` | ✅ (이전 버전 2개 백업됨) |
| README, TRIGGER_POLICY | `model-store/flashlight/` | ✅ |

### Argo Workflows 매니페스트

| 항목 | 파일 | 상태 |
|---|---|---|
| WorkflowTemplate | `flashlight-model-update-workflow.yaml` | ✅ 초안 |
| CronWorkflow | (동일 파일) | ✅ 초안 |

### 경로 이식성

| 항목 | 상태 | 근거 |
|---|---|---|
| 모든 스크립트 경로 계산 | ✅ `__file__` 기반 | 절대경로 하드코딩 없음 |
| Python 패키지 구조 | ✅ `__init__.py` 완비 | `python -m flashlight.scripts.*` 실행 가능 |
| GPU CPU fallback | ✅ `--device auto` | `get_device()` CUDA/CPU 자동 선택 |

### Python 의존성 (현재 환경 설치 확인)

| 패키지 | 버전 | 용도 |
|---|---|---|
| `onnxruntime` | 1.23.2 | ONNX 추론 |
| `joblib` | 1.5.3 | normalizer 로드 |
| `scikit-learn` | 1.7.2 | StandardScaler |
| `numpy` | 2.2.6 | 수치 연산 |
| `pandas` | 2.3.3 | 학습 결과 CSV |
| `torch` | (설치 확인 필요) | 학습 (1차 파이프라인에서는 미사용) |

---

## [MISSING] — 반드시 구현 필요

우선순위 높은 순으로 정렬한다.

---

### M-1. Dockerfile + requirements.txt 없음 ← **최우선**

**현재 상태**: Dockerfile, requirements.txt 모두 존재하지 않는다.

**문제**: Argo Workflows는 컨테이너 이미지로 실행된다. 이미지가 없으면 Workflow 자체를 실행할 수 없다.

**필요 내용**:

```
ml-pipeline/
├── Dockerfile                  ← 없음, 생성 필요
└── requirements.txt            ← 없음, 생성 필요
```

최소 `requirements.txt` 내용:
```
onnxruntime>=1.15.0
joblib>=1.3.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
torch>=2.0.0          # 학습 포함 시. 1차 pipeline만이면 제외 가능
```

최소 `Dockerfile` 구조:
```dockerfile
FROM python:3.10-slim
WORKDIR /workspace/ml-pipeline
COPY . .
RUN pip install -r requirements.txt
ENV PYTHONPATH=/workspace/ml-pipeline
```

> `WORKDIR`과 `PYTHONPATH`를 `/workspace/ml-pipeline`으로 설정해야  
> `python -m flashlight.scripts.*` 실행 및 `__file__` 기반 경로 계산이 모두 정상 동작한다.

---

### M-2. PYTHONPATH 환경변수 미설정

**현재 상태**: YAML의 `script` 템플릿에 `env` 설정이 없다.

**문제**: `python -m flashlight.scripts.run_model_update_pipeline` 실행 시  
`/workspace/ml-pipeline`이 `sys.path`에 없으면 `ModuleNotFoundError` 발생.

**해결**: Dockerfile의 `ENV PYTHONPATH=/workspace/ml-pipeline`으로 해결하거나,  
YAML `script` 템플릿에 `env` 블록 추가.

```yaml
# YAML 방식 (Dockerfile에 ENV가 없을 경우)
script:
  env:
  - name: PYTHONPATH
    value: /workspace/ml-pipeline
```

> Dockerfile에 `ENV PYTHONPATH`를 넣는 것이 더 안정적이다.  
> YAML과 Dockerfile 중 하나에만 설정하면 되며, 중복 설정해도 무방하다.

---

### M-3. model-store PVC 미생성

**현재 상태**: YAML에 `flashlight-model-store-pvc` 플레이스홀더만 있다.  
실제 PVC가 없으면 Pod가 `Pending` 상태로 머문다.

**문제**:
- promote step이 `os.rename()`으로 `current/`를 교체하므로 **읽기/쓰기 권한** 필요
- `os.rename()`은 **같은 파일시스템 내에서만 원자적** → model-store 전체(current, candidates, archive)가 단일 PVC에 있어야 한다

**필요한 PVC 매니페스트**:

```yaml
# Minikube hostPath 기반 예시
apiVersion: v1
kind: PersistentVolume
metadata:
  name: flashlight-model-store-pv
spec:
  capacity:
    storage: 5Gi
  accessModes: [ReadWriteOnce]
  hostPath:
    path: /data/agami/model-store/flashlight   # Minikube 노드 경로

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: flashlight-model-store-pvc
  namespace: argo
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 5Gi
```

**마운트 경로**: `/workspace/ml-pipeline/model-store`  
→ 코드가 `/workspace/ml-pipeline/flashlight/scripts/` 에 위치할 때 `_ML_PIPELINE_ROOT/model-store`와 일치한다.

---

### M-4. data/flashlight PVC 미생성

**현재 상태**: `flashlight-data-pvc` 플레이스홀더만 있다.

**1차 파이프라인(package→compare→promote) 기준 필요 데이터**:

```
data/flashlight에서 필요한 항목 (읽기 전용):
  runs/{run_name}/
    ├── mouse_gru_server_final_v2.onnx        ← onnx_path 파라미터
    ├── mouse_normalizer_server_final_v2.joblib ← normalizer_path 파라미터
    └── mouse_metadata_server_final_v2.json    ← metadata_path 파라미터
```

> 학습(training) 단계는 1차에 포함되지 않으므로, `processed/merged_dynamic_features_sampled.json` 등의 학습 데이터는 1차에서 불필요하다.

**필요한 PVC 매니페스트**:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: flashlight-data-pvc
  namespace: argo
spec:
  accessModes: [ReadOnlyMany]   # 1차에서는 읽기 전용
  resources:
    requests:
      storage: 10Gi
```

**마운트 경로**: `/workspace/ml-pipeline/data`  
→ onnx_path 파라미터를 `data/runs/...` 상대경로로 지정할 때 기준점이 된다.

---

### M-5. onnx_path 파라미터 기본값 미정

**현재 상태**: YAML 기본값이 `runs/mouse_gru_latest/...` 이지만  
실제 파일은 `runs/mouse_gru_final_v3_policy_tuned/`에 있다.

```yaml
# 현재 YAML (잘못된 기본값)
- name: onnx_path
  value: "runs/mouse_gru_latest/mouse_gru_server_final_v2.onnx"

# 실제 존재하는 경로
runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.onnx
```

**해결 방법 2가지**:

1. **명시적 전달**: CronWorkflow 또는 `argo submit` 시 항상 정확한 경로를 파라미터로 전달
2. **최신 run symlink**: `runs/latest/ → runs/mouse_gru_final_v3_policy_tuned/` symlink 생성

> 2차 구현에서 training step을 분리하면 step 간 output 전달로 자연스럽게 해결된다.  
> 1차에서는 수동으로 정확한 경로를 전달하는 방식을 유지한다.

---

### M-6. Argo Workflows 설치 여부 미확인

**현재 상태**: Minikube 클러스터에 Argo Workflows 컨트롤러가 설치되었는지 확인되지 않았다.

**확인 명령**:
```bash
kubectl get pods -n argo
kubectl get crd workflows.argoproj.io
```

**미설치 시 설치 명령**:
```bash
kubectl create namespace argo
kubectl apply -n argo -f \
  https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
```

---

### M-7. ServiceAccount + RBAC 미설정

**현재 상태**: YAML에 `serviceAccountName`이 지정되지 않았다.

**문제**: Argo Workflows는 workflow Pod에 `serviceAccountName`을 명시해야 한다.  
미지정 시 `default` SA를 사용하지만, Argo에서는 `argo` SA가 일반적으로 요구된다.

**최소 설정**:
```yaml
# WorkflowTemplate spec에 추가
spec:
  serviceAccountName: argo        # argo 네임스페이스의 기본 SA
```

또는 별도 SA + Role 생성:
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: flashlight-workflow-sa
  namespace: argo
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: flashlight-workflow-role
  namespace: argo
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["argoproj.io"]
  resources: ["workflows"]
  verbs: ["get", "list"]
```

> PVC 읽기/쓰기 권한은 RBAC이 아닌 Pod의 volumeMount로 처리된다.  
> Argo 자체 동작(Pod 생성, 상태 조회)에 대한 최소 RBAC만 필요하다.

---

## [OPTIONAL] — 추후 고도화

### O-1. Harbor Registry (이미지 레지스트리)

**현재 상태**: 사용 여부 미확인. Minikube 환경이면 로컬 이미지로 대체 가능하다.

**Minikube 로컬 이미지 사용 방법**:
```bash
# 이미지를 Minikube에 직접 로드 (레지스트리 불필요)
minikube image load agami-mlops-flashlight:latest
```

**Harbor 사용 시 추가 필요 항목**:
- Harbor 프로젝트 생성 및 이미지 push
- `imagePullSecret` 생성 및 YAML에 참조 추가

```yaml
# Harbor 사용 시 WorkflowTemplate에 추가
spec:
  imagePullSecrets:
  - name: harbor-registry-secret
```

---

### O-2. GPU 노드 스케줄링

**현재 상태**: 1차 파이프라인(package→compare→promote)은 GPU 미사용.  
`train_mouse_gru.py`는 `--device auto`로 CPU fallback 지원한다.

**2차 구현(training step 분리)에서 필요**:
```yaml
# training step에만 적용
resources:
  limits:
    nvidia.com/gpu: "1"
nodeSelector:
  accelerator: nvidia-gpu
tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

---

### O-3. check_trigger.py 미구현

**현재 상태**: TRIGGER_POLICY.md에 트리거 조건이 정의되어 있지만,  
이를 판단하는 `check_trigger.py` 스크립트가 없다.

**현재 영향**: CronWorkflow가 매주 트리거 조건과 무관하게 실행된다.  
→ 조건 미충족 시 불필요한 compare/promote가 실행되는 정도의 낭비만 발생한다.

**구현 시 필요 내용**:
```python
# check_trigger.py 최소 로직
def check():
    # 1. 신규 라벨 데이터 수 확인
    # 2. 운영 성능 지표 임계값 확인
    # 3. 최장 미갱신 기간 확인
    return True / False
```

---

### O-4. 알림 연동

**현재 상태**: 파이프라인 성공/실패 알림 없음.

**구현 방법**: Argo Workflows `onExit` 핸들러 또는 `exitCode` 조건으로  
Slack webhook 호출 step 추가.

```yaml
spec:
  onExit: notify-result
  templates:
  - name: notify-result
    container:
      image: curlimages/curl
      command: [sh, -c]
      args: ["curl -X POST $SLACK_WEBHOOK_URL -d '{\"text\":\"Pipeline finished\"}'"]
```

---

### O-5. Argo CD 연동

**현재 상태**: promote 완료 후 captcha-engine 자동 배포 없음.  
캡챠 엔진 연동 미구현으로 인해 현 단계에서 불필요하다.

---

## 우선순위별 요약

### MISSING — 실행 차단 항목 (블로커)

| 순위 | 항목 | 예상 작업 | 비고 |
|---|---|---|---|
| 1 | **Dockerfile + requirements.txt** | 2~4h | 없으면 이미지 빌드 불가 |
| 2 | **PYTHONPATH 환경변수** | 30min | Dockerfile ENV로 해결 |
| 3 | **model-store PVC 매니페스트** | 1h | 없으면 Pod Pending |
| 4 | **data/flashlight PVC 매니페스트** | 1h | 없으면 Pod Pending |
| 5 | **onnx_path 등 파라미터 기본값 정정** | 30min | YAML 수정 |
| 6 | **Argo Workflows 설치 확인** | 30min | 미설치 시 1h 추가 |
| 7 | **ServiceAccount + RBAC** | 30min | default SA로 임시 가능 |

총 예상 작업: **최소 6~8시간**

### OPTIONAL — 고도화 항목

| 순위 | 항목 | 예상 작업 | 필요 시점 |
|---|---|---|---|
| 1 | Harbor Registry 설정 | 1~2h | 멀티 노드 / CI 자동화 시 |
| 2 | check_trigger.py 구현 | 2~4h | 2차 구현 |
| 3 | GPU 노드 스케줄링 | 1h | training step 분리(2차) 시 |
| 4 | 알림 연동 (Slack) | 1~2h | 운영 안정화 후 |
| 5 | Argo CD 연동 | 별도 설계 필요 | 캡챠 엔진 연동(3차) 시 |

---

## 다음 단계 권장 순서

```
[1] requirements.txt 작성
[2] Dockerfile 작성 (WORKDIR, PYTHONPATH 포함)
[3] 이미지 빌드 및 Minikube 로드
      docker build -t agami-mlops-flashlight:latest .
      minikube image load agami-mlops-flashlight:latest
[4] model-store PVC 매니페스트 작성 및 적용
      k8s/argo-workflows/pvc-model-store.yaml
[5] data PVC 매니페스트 작성 및 적용
      k8s/argo-workflows/pvc-data.yaml
[6] YAML onnx_path 기본값 정정 및 PYTHONPATH env 추가
[7] Argo Workflows 설치 확인
[8] ServiceAccount 설정
[9] argo submit --dry-run 으로 최종 검증
```
