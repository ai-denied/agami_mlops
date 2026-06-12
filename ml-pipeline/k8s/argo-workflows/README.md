# k8s/argo-workflows — flashlight MLOps 자동화

flashlight (Mouse GRU 봇 탐지) 모델의 재학습/업데이트 파이프라인을  
Argo Workflows로 자동화하기 위한 쿠버네티스 매니페스트 디렉토리.

> **현재 단계**: 1차 구현 (단일 step)  
> 캡챠 엔진 서버 연동은 구현되지 않았다.

---

## 디렉토리 구조

```
k8s/argo-workflows/
├── flashlight-model-update-workflow.yaml   ← WorkflowTemplate + CronWorkflow
├── pvc-flashlight.yaml                     ← PVC 2개 (model-store, data)
├── rbac-flashlight-workflow.yaml           ← ServiceAccount + Role + RoleBinding
├── GAP_ANALYSIS.md                         ← 실행 전 Gap 분석
└── README.md

ml-pipeline/
├── Dockerfile.flashlight                   ← 파이프라인 실행용 이미지 빌드 파일
└── requirements-flashlight.txt             ← 최소 의존성 목록
```

---

## 전체 적용 순서

클러스터에서 Argo Workflow를 처음 실행할 때 아래 순서로 진행한다.

### Step 1 — Argo Workflows 설치 확인

```bash
# Argo Workflows 컨트롤러 동작 여부 확인
kubectl get pods -n argo

# CRD 등록 여부 확인 (아래 리소스가 있어야 함)
kubectl get crd | grep argoproj.io
# 필수: workflows.argoproj.io, workflowtemplates.argoproj.io, cronworkflows.argoproj.io

# 미설치 시 설치 (최신 버전)
kubectl create namespace argo
kubectl apply -n argo \
  -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
```

### Step 2 — namespace 확인

```bash
# argo namespace 존재 확인
kubectl get namespace argo

# 없으면 생성
kubectl create namespace argo
```

### Step 3 — Docker 이미지 준비

```bash
# [Minikube 환경] 로컬 이미지 로드
docker build -f ml-pipeline/Dockerfile.flashlight -t agami-mlops-flashlight:local .
minikube image load agami-mlops-flashlight:local

# [Harbor 사용 시] 이미지 빌드 및 push
docker build -f ml-pipeline/Dockerfile.flashlight \
  -t <harbor>/agami/ml-pipeline-flashlight:latest .
docker push <harbor>/agami/ml-pipeline-flashlight:latest
```

### Step 4 — PVC 적용

> 적용 전 `pvc-flashlight.yaml`의 `storageClassName` 주석을 해제하고  
> 클러스터 환경에 맞는 값으로 수정한다. (`kubectl get storageclass` 참고)

```bash
kubectl apply -f k8s/argo-workflows/pvc-flashlight.yaml -n argo

# Bound 상태 확인 (Pending이면 StorageClass 또는 PV 문제)
kubectl get pvc -n argo
```

### Step 5 — RBAC 적용

```bash
kubectl apply -f k8s/argo-workflows/rbac-flashlight-workflow.yaml -n argo

# ServiceAccount 생성 확인
kubectl get sa flashlight-workflow-sa -n argo

# Role/RoleBinding 확인
kubectl get role,rolebinding -n argo | grep flashlight
```

### Step 6 — WorkflowTemplate 적용

> 적용 전 `flashlight-model-update-workflow.yaml`의  
> `image: agami-mlops-flashlight:latest` 를 실제 이미지 경로로 수정한다.

```bash
kubectl apply -f k8s/argo-workflows/flashlight-model-update-workflow.yaml -n argo

# WorkflowTemplate 등록 확인
kubectl get workflowtemplate -n argo
kubectl get cronworkflow -n argo
```

### Step 7 — dry-run submit 검증

```bash
# dry-run 실행 (promote는 변경 없음)
argo submit --from workflowtemplate/flashlight-model-update \
  -n argo \
  -p version=v4_20260610 \
  -p onnx_path=runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.onnx \
  -p normalizer_path=runs/mouse_gru_final_v3_policy_tuned/mouse_normalizer_server_final_v2.joblib \
  -p metadata_path=runs/mouse_gru_final_v3_policy_tuned/mouse_metadata_server_final_v2.json \
  -p dry_run=true \
  --watch

# 실행 이력 확인
argo list -n argo

# 로그 확인
argo logs -n argo <workflow-name>
```

### 적용 순서 요약

```
[1] Argo Workflows 설치 확인
      ↓
[2] namespace 확인/생성
      ↓
[3] Docker 이미지 빌드 및 레지스트리 등록 (또는 minikube image load)
      ↓
[4] PVC 적용         pvc-flashlight.yaml
      ↓
[5] RBAC 적용        rbac-flashlight-workflow.yaml
      ↓
[6] WorkflowTemplate 적용  flashlight-model-update-workflow.yaml
      ↓
[7] dry-run submit   argo submit --from workflowtemplate/... -p dry_run=true
```

---

## Argo CD vs Argo Workflows — 역할 구분

두 도구는 역할이 다르며 서로를 대체하지 않는다.

| 도구 | 분류 | 역할 |
|---|---|---|
| **Argo Workflows** | 파이프라인 실행 엔진 | 학습 → 평가 → 승격 파이프라인의 각 단계를 컨테이너로 실행 |
| **Argo CD** | GitOps 배포 동기화 | Git 저장소의 변경을 감지해 클러스터 상태를 선언적으로 동기화 |

### 이 프로젝트에서의 역할

```
┌─────────────────────────────────────────────────────────────────┐
│                     Argo Workflows                              │
│                                                                 │
│  CronWorkflow (매주 월요일)                                      │
│       ↓                                                         │
│  [단일 step] run_model_update_pipeline.py                       │
│       ├─ package  → candidates/{version}/                       │
│       ├─ compare  → PASS / FAIL                                 │
│       └─ promote  → model-store/flashlight/current/ 갱신 ──┐   │
└────────────────────────────────────────────────────────────┼───┘
                                                             │
                   current/ 변경이 Git에 커밋되면            │
                                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Argo CD                   (3차 구현)      │
│                                                                 │
│  Git 저장소 변경 감지                                            │
│       ↓                                                         │
│  captcha-engine Deployment 동기화                               │
│       ↓                                                         │
│  Pod 재기동 → 새 모델 마운트 반영                                │
└─────────────────────────────────────────────────────────────────┘
```

> **현재**: Argo CD 연동은 미구현.  
> Argo Workflows 파이프라인이 `model-store/flashlight/current/`를 갱신하는 것까지가 현재 범위다.

---

## 실행 흐름

### CronWorkflow 자동 실행 (매주 월요일 03:00 KST)

```
CronWorkflow 트리거
        ↓
WorkflowTemplate: flashlight-model-update
        ↓
[STEP 1/1] run-pipeline (단일 컨테이너)
        │
        │  python -m flashlight.scripts.run_model_update_pipeline
        │    --version  {version}
        │    --onnx     {onnx_path}
        │    --normalizer {normalizer_path}
        │    --metadata  {metadata_path}
        │    [--dry-run]
        │
        ├─ [STAGE 1] package_for_captcha_engine
        │      candidates/{version}/ 생성
        │
        ├─ [STAGE 2] compare_candidate
        │      current vs candidate 성능 비교
        │      FAIL → exit 1 (promote 건너뜀)
        │
        └─ [STAGE 3] promote_model
               candidates/{version}/ → current/
               이전 current → archive/{timestamp}_{version}/
```

### exit code 처리

| exit code | 의미 | Argo Workflows 상태 |
|---|---|---|
| 0 | 전체 성공 (promote 완료) | Succeeded |
| 1 | compare FAIL 또는 오류 | Failed (retryPolicy: OnError 미해당) |

compare FAIL은 논리적 실패이므로 자동 재시도하지 않는다.  
시스템 오류(OOM, 노드 장애)만 `retryStrategy.limit: 2` 로 자동 재시도한다.

---

## 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `version` | `""` | 모델 버전명. 비워두면 `v_YYYYMMDD` 자동 생성 |
| `onnx_path` | `runs/mouse_gru_latest/...onnx` | 학습 산출물 ONNX 경로 |
| `normalizer_path` | `runs/mouse_gru_latest/...joblib` | 학습 산출물 normalizer 경로 |
| `metadata_path` | `runs/mouse_gru_latest/...json` | 학습 산출물 metadata 경로 |
| `dry_run` | `"false"` | `"true"` 이면 promote를 dry-run으로만 실행 |

---

## 필요한 볼륨 (PVC)

PVC 매니페스트: **`pvc-flashlight.yaml`**

| PVC 이름 | 마운트 경로 | 용량 | 용도 |
|---|---|---|---|
| `flashlight-model-store-pvc` | `/workspace/ml-pipeline/model-store` | 5Gi | current/, candidates/, archive/ |
| `flashlight-data-pvc` | `/workspace/ml-pipeline/data` | 20Gi | 학습 데이터, runs/ 산출물 |

### PVC 적용 전 확인 사항

kubectl apply 전에 아래 항목을 반드시 확인한다.

#### 1. storageClassName 확인

클러스터마다 사용 가능한 StorageClass가 다르다.  
`pvc-flashlight.yaml`의 `storageClassName` 주석을 해제하고 환경에 맞는 값으로 교체한다.

```bash
# 클러스터에서 사용 가능한 StorageClass 목록 확인
kubectl get storageclass

# 환경별 일반적인 기본값
#   Minikube      : standard
#   AWS EKS       : gp3
#   GKE           : standard
#   온프레미스/Longhorn: longhorn
```

#### 2. accessModes 검토

| 시나리오 | 권장 accessModes | 비고 |
|---|---|---|
| Minikube 단일 노드 | `ReadWriteOnce` | 기본값, 변경 불필요 |
| 멀티 노드 + 동일 PVC | `ReadWriteMany` | NFS/EFS/Longhorn/Ceph CSI 드라이버 필요 |
| training + pipeline 동시 실행 | `ReadWriteMany` | data-pvc 한정 |

> **model-store-pvc 주의**: `promote_model.py`의 `os.rename()`이 원자적으로 동작하려면  
> `current/`, `candidates/`, `archive/`가 반드시 **동일 PVC** 안에 있어야 한다.

#### 3. 용량 확인

현재 실측 데이터 크기:

| 항목 | 실측 크기 | PVC 할당 |
|---|---|---|
| model-store (current + candidates × 수 버전) | ~300MB 이하 | 5Gi (여유분 포함) |
| data/flashlight (raw + processed + generated) | ~870MB | 20Gi (학습 데이터 증가 여유분 포함) |
| runs/ (학습 산출물) | ~7MB (현재) | data-pvc에 포함 |

#### 4. namespace 확인

매니페스트의 `namespace: argo`가 클러스터에 존재하는지 확인한다.

```bash
kubectl get namespace argo
# 없으면 생성
kubectl create namespace argo
```

#### 5. PVC 적용 명령

확인이 완료되면 아래 명령으로 적용한다.

```bash
# 적용
kubectl apply -f pvc-flashlight.yaml -n argo

# 상태 확인 (Bound 상태여야 정상)
kubectl get pvc -n argo

# 상세 확인
kubectl describe pvc flashlight-model-store-pvc -n argo
kubectl describe pvc flashlight-data-pvc -n argo
```

#### 6. Minikube 환경에서 hostPath PV 사용 시

Minikube에는 기본 StorageClass(`standard`)가 있어 별도 PV 생성 없이 PVC만으로 동적 프로비저닝이 가능하다.  
단, `hostPath` PV는 Minikube 노드 내부 경로에 데이터가 저장되므로 초기 데이터(runs/ 산출물)를 마운트하려면 아래처럼 복사해야 한다.

```bash
# Minikube 노드에 데이터 복사 (예시)
minikube ssh -- mkdir -p /data/flashlight/runs
minikube cp ml-pipeline/runs/ /data/flashlight/runs/
```

---

## 컨테이너 이미지

### 이미지 파일 위치

```
ml-pipeline/
├── Dockerfile.flashlight          ← 이미지 빌드 파일
└── requirements-flashlight.txt    ← 최소 의존성
```

### 이미지 내 구조

```
/workspace/ml-pipeline/           ← WORKDIR, PYTHONPATH
└── flashlight/                   ← 패키지 코드 (COPY)
    ├── scripts/
    │   ├── run_model_update_pipeline.py
    │   ├── package_for_captcha_engine.py
    │   ├── compare_candidate.py
    │   └── promote_model.py
    ├── inference/
    ├── data/
    └── ...
                                  ← 아래는 이미지에 포함하지 않음 (PVC 마운트)
/workspace/ml-pipeline/model-store/   ← model-store PVC
/workspace/ml-pipeline/runs/          ← data PVC
/workspace/ml-pipeline/data/          ← data PVC
```

### 의존성 (requirements-flashlight.txt)

| 패키지 | 필요 이유 | 구분 |
|---|---|---|
| `numpy` | scaler 연산 | core |
| `joblib` | normalizer .joblib 로드 | core |
| `scikit-learn` | `MouseFeatureNormalizer` → `StandardScaler` | core |
| `onnxruntime` | ONNX 검증 및 추론 | validate |
| `torch` | **미포함** — training step 전용 | training only |
| `pandas` | **미포함** — training step 전용 | training only |

> `joblib.load(normalizer.joblib)` 실행 시 `MouseFeatureNormalizer` 클래스 정의가 필요하며,  
> 이 클래스가 `sklearn.StandardScaler`를 사용하므로 `scikit-learn`은 필수다.

### Docker 빌드 명령

빌드 컨텍스트는 **저장소 루트(`agami-mlops/`)** 에서 실행한다.

```bash
# 로컬 빌드
docker build \
  -f ml-pipeline/Dockerfile.flashlight \
  -t agami-mlops-flashlight:latest \
  .

# Minikube에 로컬 이미지 로드 (레지스트리 없이 사용 가능)
minikube image load agami-mlops-flashlight:latest

# 빌드 + 로드 한 번에
docker build -f ml-pipeline/Dockerfile.flashlight -t agami-mlops-flashlight:latest . \
  && minikube image load agami-mlops-flashlight:latest
```

### 로컬 실행 테스트 (컨테이너 동작 검증)

```bash
# dry-run 실행 (model-store, runs 볼륨 마운트 필요)
docker run --rm \
  -v $(pwd)/ml-pipeline/model-store:/workspace/ml-pipeline/model-store \
  -v $(pwd)/ml-pipeline/runs:/workspace/ml-pipeline/runs \
  agami-mlops-flashlight:latest \
  python -m flashlight.scripts.run_model_update_pipeline \
    --version    v4_20260610 \
    --onnx       runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.onnx \
    --normalizer runs/mouse_gru_final_v3_policy_tuned/mouse_normalizer_server_final_v2.joblib \
    --metadata   runs/mouse_gru_final_v3_policy_tuned/mouse_metadata_server_final_v2.json \
    --dry-run

# import 및 --help 동작 확인
docker run --rm agami-mlops-flashlight:latest
```

### Harbor Registry Push (placeholder)

Harbor Registry 사용 시 이미지 태그 및 push 예시.  
`<harbor>` 는 실제 Harbor 주소로 교체한다.

```bash
# 태그 지정
docker tag agami-mlops-flashlight:latest \
  <harbor>/agami/ml-pipeline-flashlight:latest

docker tag agami-mlops-flashlight:latest \
  <harbor>/agami/ml-pipeline-flashlight:v1.0.0

# Harbor push (로그인 필요)
docker login <harbor>
docker push <harbor>/agami/ml-pipeline-flashlight:latest
docker push <harbor>/agami/ml-pipeline-flashlight:v1.0.0

# Argo Workflow YAML에서 이미지 참조 교체
# image: agami-mlops-flashlight:latest
# → image: <harbor>/agami/ml-pipeline-flashlight:latest
```

---

## 적용 및 실행 명령

```bash
# 리소스 적용
kubectl apply -f flashlight-model-update-workflow.yaml

# 수동 즉시 실행 (version 지정)
argo submit --from workflowtemplate/flashlight-model-update \
  -n argo \
  -p version=v4_20260610 \
  -p onnx_path=runs/mouse_gru_v4/mouse_gru_server_final_v2.onnx \
  -p normalizer_path=runs/mouse_gru_v4/mouse_normalizer_server_final_v2.joblib \
  -p metadata_path=runs/mouse_gru_v4/mouse_metadata_server_final_v2.json

# dry-run으로 사전 검증
argo submit --from workflowtemplate/flashlight-model-update \
  -n argo \
  -p version=v4_20260610 \
  -p dry_run=true \
  --watch

# 실행 이력 확인
argo list -n argo

# 로그 확인
argo logs -n argo <workflow-name>

# CronWorkflow 상태 확인
kubectl get cronworkflow -n argo
```

---

## 2차 구현 — 단계별 Step 분리 방향

1차 구현의 단일 step을 6개의 독립 step으로 분리한다.  
각 step이 독립 컨테이너로 실행되어 단계별 재시도, 로그, 조건 분기가 가능해진다.

### 목표 구조

```yaml
templates:
- name: pipeline
  steps:
  - - name: trigger-check      # 트리거 조건 판단, 미충족 시 이후 건너뜀
  - - name: preprocessing      # 신규 로그 병합, 피처 추출
  - - name: training           # GRU 모델 학습 (GPU 노드)
  - - name: package            # candidates/{version}/ 생성
  - - name: compare            # PASS/FAIL 판단
  - - name: promote            # compare PASS 시에만 실행 (when 조건)
      when: "{{steps.compare.outputs.result}} == 'PASS'"
```

### 단계별 변경점

| 단계 | 1차 구현 | 2차 구현 |
|---|---|---|
| 파이프라인 구조 | 단일 컨테이너 | 6개 독립 Step |
| 실패 시 재시도 | 전체 재실행 | 실패 Step만 재시도 |
| compare FAIL | exit 1로 전체 실패 | `when` 조건으로 promote 건너뜀, 나머지는 Succeeded |
| training GPU | 파이프라인 전체가 GPU 필요 | training Step만 GPU 노드에 스케줄링 가능 |
| 트리거 체크 | 미포함 (외부에서 판단) | trigger-check Step으로 포함 |

### 선행 조건

- `check_trigger.py` 스크립트 구현 (트리거 조건 코드화)
- 각 단계별 독립 실행 가능한 컨테이너 이미지 또는 단일 이미지 내 모듈 분리
- compare step의 `outputs.result` 출력 포맷 확정

---

## 버전 이력

| 버전 | 날짜 | 변경 내용 |
|---|---|---|
| v1.0 | 2026-06-10 | 1차 구현 — 단일 step WorkflowTemplate + CronWorkflow |
| v1.1 | 2026-06-10 | Dockerfile.flashlight + requirements-flashlight.txt 추가, Docker 빌드 명령 문서화 |
| v1.2 | 2026-06-10 | pvc-flashlight.yaml 추가, PVC 적용 전 확인 사항 문서화 |
| v1.3 | 2026-06-10 | rbac-flashlight-workflow.yaml 추가, 전체 적용 순서(Step 1~7) 문서화 |
