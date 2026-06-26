# agami_mlops

Agami CAPTCHA 시스템의 ML 백엔드 저장소입니다. 마우스 궤적·얼굴 활성도·감정/상황 이미지 세 가지 AI 모델을 학습·평가·배포하는 MLOps 파이프라인을 포함합니다.

> CAPTCHA 엔진(서빙 측) 코드는 [ai-denied/agami-captcha](https://github.com/ai-denied/agami-captcha) 저장소에 분리되어 있습니다.

---

## 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                  Agami CAPTCHA ML Backend                    │
│                                                              │
│   ┌─────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│   │  Flashlight │  │ Face Liveness  │  │ Context Emotion  │ │
│   │  (봇 탐지)  │  │ (위변조 탐지)  │  │ (감정/상황 분류) │ │
│   │  Port 8080  │  │  Port 8081     │  │  (batch only)    │ │
│   └──────┬──────┘  └───────┬────────┘  └────────┬─────────┘ │
│          │                 │                    │            │
│     mouse GRU          face GRU           EfficientNet-B0   │
│   bot_risk_score       spoof_score        14 emotion classes │
└──────────────────────────────────────────────────────────────┘
         │                   │                    │
         └───────────────────┴────────────────────┘
                         FastAPI + ONNX Runtime
                    Kubernetes (agami namespace)
                    ArgoCD GitOps / Argo Workflows
```

---

## 모델 서비스 3종

### 1. Flashlight — 마우스 궤적 봇 탐지

마우스 이동 trajectory를 분석해 사람/봇을 실시간으로 구별합니다.

**모델**: `MouseGRUModelV2` — GRU 시퀀스 인코더 + Static MLP 결합

| 구분 | 피처 | 설명 |
|------|------|------|
| Dynamic (7) | dx, dy, dt, distance, velocity, acceleration, angle_change | 프레임별 움직임 |
| Static (10) | duration, log_count, total/straight distance, avg/max speed, speed_std, direction_changes, pauses | 전체 궤적 통계 |

**추론 흐름**:
```
trajectory (mouse events)
    → build_features (dynamic + static)
    → normalize (seq_scaler / static_scaler)
    → ONNX Runtime → bot_risk_score (0~1)
    → risk_band (low_risk / suspicious / high_risk)
    → 3회 누적 정책 → allow / block
```

**API 엔드포인트** (`POST :8080`):

| 엔드포인트 | 입력 | 출력 |
|-----------|------|------|
| `/api/v1/predict` | trajectory 1회 | `bot_risk_score`, `risk_band` |
| `/api/v1/decide` | score 목록 (최대 3회) | `final_decision` (allow/block) |
| `/health` | — | 서버 및 모델 상태 |
| `/model/info` | — | 로드된 모델 메타데이터 |

---

### 2. Face Liveness — 얼굴 활성도(위변조) 탐지

얼굴 랜드마크 피처 시퀀스를 분석해 실제 사람인지 스푸핑 시도인지 판단합니다.

**모델**: `FaceLivenessGRU` — GRU (스케일러 ONNX 그래프 내장)

- 입력: `x_seq (batch, 16, 20)` — 16프레임 × 20개 얼굴 피처
- 출력: `spoof_score (0~1)` — 0에 가까울수록 live, 1에 가까울수록 spoof

**3라운드 CAPTCHA 판정** (`captcha_decision.py`):

```
3라운드 결과 수집 (face 미션 + hand 미션 혼합 필수)
    → risk_band 분류 (real_safe / suspicious / spoof_detected)
    → 라운드별 risk 누적 (band 고정 가중치 사용, 연속 score 직접 미사용*)
    → total_risk 기준 판정

  total_risk < 1.20  →  PASS
  total_risk < 2.00  →  RETRY
  total_risk ≥ 2.00
    + spoof_detected ≥ 2/3  →  FAIL
    + spoof_detected < 2/3  →  RETRY (face score 단독 차단 방지)
```

> *R_live_clip 실환경 데이터에서 FRR 95%가 확인된 모델이므로, raw spoof_score 연속값이 아닌
> risk_band 기반 고정 가중치로 판정하여 실사용자 오탐을 방지합니다.

**API 엔드포인트** (`POST :8081`):

| 엔드포인트 | 입력 | 출력 |
|-----------|------|------|
| `/api/v1/predict` | `x_seq (16×20)` | `spoof_score`, `risk_band` |
| `/api/v1/decide` | 3라운드 결과 | `decision` (PASS/RETRY/FAIL) |
| `/health` | — | 서버 및 모델 상태 |
| `/model/info` | — | 로드된 모델 메타데이터 |

**모델 성능** (Main Benchmark, test set):

| 지표 | 값 |
|------|----|
| Accuracy | 91.97% |
| Attack Block Rate | 92.22% |
| Genuine Pass Rate | 91.67% |
| F1 (spoof) | 92.74% |

---

### 3. Context Emotion — CAPTCHA 이미지 감정/상황 분류

CAPTCHA 챌린지 이미지에 감정·상황 라벨을 자동으로 부여합니다.

**모델**: `EmotionClassifier` — EfficientNet-B0 백본 (ImageNet) + Linear head (파인튜닝)

**감정 14종**: happiness, calm, anticipation, affection, anger, fear, sadness, disconnection, suffering, aversion, embarrassment, confidence, confusion, yearning

**상황 7종**: conflict, danger, loss_absence, pressure, safety, teasing, vanity

**EMOTIC 데이터셋** 26개 원본 카테고리를 위 14종으로 매핑하여 학습 데이터를 구성합니다.

---

## 디렉토리 구조

```
agami_mlops/
├── ml-pipeline/
│   ├── flashlight/                    # 마우스 봇 탐지
│   │   ├── model/mouse_gru.py         # MouseGRUModelV2
│   │   ├── inference/                 # ONNX 추론기 (joblib / JSON normalizer)
│   │   ├── training/                  # 학습 루프
│   │   ├── evaluation/                # 메트릭, threshold 정책, ONNX 컨트랙트
│   │   ├── export/                    # PyTorch → ONNX
│   │   ├── scripts/
│   │   │   └── run_model_update_pipeline.py  # package → compare → promote
│   │   └── api/main.py                # FastAPI (port 8080)
│   │
│   ├── facial_recognition/            # 얼굴 활성도 탐지
│   │   ├── model/face_liveness_gru.py # FaceLivenessGRU + ONNX wrapper
│   │   ├── captcha_decision.py        # 3라운드 판정 로직 (PASS/RETRY/FAIL)
│   │   ├── inference/                 # ONNX 추론기
│   │   ├── preprocessing/             # 영상 피처 추출
│   │   ├── evaluation/                # 메트릭, ONNX 컨트랙트
│   │   ├── scripts/
│   │   │   └── run_model_update_pipeline.py  # package → compare → promote
│   │   └── api/main.py                # FastAPI (port 8081)
│   │
│   ├── context_emotion/               # 감정/상황 분류
│   │   ├── model/emotion_classifier.py  # EfficientNet-B0 + head
│   │   ├── common/constants.py        # 감정 14종, 상황 7종, EMOTIC 매핑
│   │   ├── preprocessing/             # 데이터셋 빌드 (v1/v2)
│   │   ├── evaluation/                # 후보 평가, 승격 게이트, 공격 프록시 평가
│   │   ├── deployment/                # model_store, promote, rollback
│   │   ├── ops_metrics/recorder.py    # 운영 지표 기록
│   │   └── scripts/
│   │       └── run_model_update_pipeline.py  # evaluate → package → compare → promote → smoke_test → record
│   │
│   ├── common/api/dashboard.py        # 공통 대시보드 API 라우터
│   ├── model-store/                   # 서빙 중 모델 (PVC 마운트)
│   │   └── facial_recognition/current/
│   │       ├── face_liveness.onnx
│   │       ├── seq_scaler.joblib
│   │       └── metadata.json
│   └── k8s/argo-workflows/            # Argo Workflows 매니페스트
│
├── manifests/                         # Kubernetes 리소스 정의
│   ├── flashlight-inference-api/      # Deployment, PVC
│   ├── face-liveness-api.yaml
│   ├── dashboard-api/
│   └── ...
├── argocd/apps/                       # ArgoCD GitOps 앱 정의
├── .github/workflows/                 # GitHub Actions (Docker 빌드/push)
└── retrospectives/                    # 날짜별 운영 회고록
```

---

## MLOps 파이프라인

세 모델 모두 동일한 **package → compare → promote** 패턴을 따릅니다.

```
학습 완료 (ONNX + normalizer + metadata)
        │
        ▼
  [1] package       candidates/{version}/ 생성, 아티팩트 복사
        │
        ▼
  [2] compare       ONNX 컨트랙트 검증 + 현재 모델 대비 성능 비교
        │  FAIL → 중단
        ▼
  [3] promote       candidates/{version}/ → model-store/current/
        │           이전 current → archive/{timestamp}_{version}/
        ▼
  [4] smoke_test    빠른 동작 확인   (context_emotion 추가)
        │
        ▼
  [5] record        운영 지표 기록   (context_emotion 추가)
```

> **context_emotion**은 6단계 파이프라인 (`evaluate → package → compare → gate → promote → smoke_test → record`)으로, 클래스별 회귀 검사 및 공격 프록시 평가까지 포함합니다.

### ONNX 컨트랙트

각 모델은 `config/runtime_contract.yaml`에 정의된 입출력 텐서 계약을 승격 전 자동으로 검증합니다.

| 모델 | 입력 | 출력 |
|------|------|------|
| Flashlight | `x_seq (batch, seq, 7)`, `lengths (batch)`, `x_static (batch, 10)` | `bot_risk_score (batch)` |
| Face Liveness | `x_seq (batch, 16, 20)` | `spoof_score (batch)` |

---

## 인프라 / 배포

### Kubernetes 구성

```
[GitHub push]
    │ webhook
    ▼
[GitHub Actions]  →  Docker build  →  Harbor push
    │
    ▼
[K8s: agami namespace]
    ├── flashlight-inference-api  (Deployment, port 8080)
    ├── face-liveness-api         (Deployment, port 8081)
    └── dashboard-api             (Deployment + CronJob)
```

### 모델 교체 (이미지 재빌드 없음)

```bash
# 1. 새 모델 패키징 및 승격
python -m flashlight.scripts.run_model_update_pipeline \
  --version v5_20260701 \
  --onnx     runs/.../mouse_gru.onnx \
  --normalizer runs/.../normalizer.joblib \
  --metadata runs/.../metadata.json

# 2. Pod 재시작 → 새 모델 자동 로드 (이미지 재빌드 불필요)
kubectl rollout restart deployment/flashlight-inference-api -n agami
```

model-store PVC가 Deployment와 Argo Workflow Pod에 공유 마운트되어 있어, 모델 파일만 교체하면 됩니다.

### Argo Workflows (자동화)

- **CronWorkflow**: 매주 월요일 03:00 KST 자동 실행
- 단계별 재시도, compare FAIL 시 promote 자동 건너뜀
- 배포 흐름: `run_model_update_pipeline.py` → model-store/current/ 갱신

자세한 내용은 [`ml-pipeline/k8s/argo-workflows/README.md`](ml-pipeline/k8s/argo-workflows/README.md)를 참고하세요.

### 이미지 빌드

```bash
# Flashlight 추론 API
docker build -f ml-pipeline/Dockerfile.api -t agami/flashlight-api:latest .

# Face Liveness API
docker build -f ml-pipeline/Dockerfile.face-api -t agami/face-api:latest .

# MLOps 파이프라인 (학습/평가/승격)
docker build -f ml-pipeline/Dockerfile.flashlight -t agami/ml-pipeline-flashlight:latest .
```

---

## Emotion CAPTCHA Serving 흐름

감정 CAPTCHA는 **풀 생성 → 서빙 → 풀이 → 피드백** 의 순환 구조로 운영된다.

```
┌─────────────────────────────────────────────────────────────────┐
│             Emotion CAPTCHA 전체 운영 루프                      │
│                                                                 │
│  ① 풀 생성 (offline)                                            │
│     VLM 공격 (build_review_queue.py)                            │
│       → 인간 검수 (human_review_server :8765)                   │
│       → 풀 내보내기 (export_captcha_pool.py)                    │
│       → captcha_pool.csv 생성                                   │
│                                                                 │
│  ② MLOps 파이프라인 (Argo Workflows / run_pipeline.py)          │
│     validate → train → eval → choice-report                     │
│       → package → compare → promote → smoke-test               │
│     model-store/captcha_bank/current/ 갱신                      │
│                                                                 │
│  ③ Serving API (context-emotion-captcha-api :8083)              │
│     GET  /health                    풀 로드 상태 확인           │
│     POST /context-emotion/challenge current 풀에서 문제 출제    │
│     POST /context-emotion/attempt   사용자 풀이 제출 + 로깅     │
│                                                                 │
│  ④ 사용자 풀이                                                   │
│     → image_url 이미지 확인 → 감정 레이블 선택 → attempt 제출   │
│     ← is_correct, retry_allowed 반환 (정답·점수 미노출)         │
│                                                                 │
│  ⑤ attempt log 축적                                             │
│     /data/context_emotion/attempt_logs/attempts_YYYYMMDD.jsonl  │
│     저장 필드: challenge_id, sample_id, selected_label,         │
│               is_correct, points, solve_time_ms, pool_version   │
│     미저장:   final_emotion(정답), raw IP, raw user-agent       │
│                                                                 │
│  ⑥ feedback MLOps 입력 (TODO)                                   │
│     attempt_log 분석 → 취약 문항 식별                           │
│       → 인간 검수 재요청 or 풀에서 제거                         │
│       → 새 candidate pool 생성 → 평가/승격 → current 교체      │
│                                                                 │
│  (③ → ④ → ⑤ → ⑥ → ② 반복)                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Serving API 빠른 테스트

```bash
cd ml-pipeline
pip install -r requirements-captcha-bank.txt

# current/ 풀이 있을 때 (model-store/captcha_bank/current/captcha_pool.csv)
CAPTCHA_POOL_DIR=model-store/captcha_bank/current \
ATTEMPT_LOG_DIR=/tmp/attempt_logs \
uvicorn context_emotion.serving.app:app --port 8083

# 헬스체크
curl http://localhost:8083/health

# 문제 출제
curl -s -X POST http://localhost:8083/context-emotion/challenge \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session-001"}' | python3 -m json.tool

# 풀이 제출 (challenge_id는 위 응답에서 복사)
curl -s -X POST http://localhost:8083/context-emotion/attempt \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-001",
    "challenge_id": "<위에서받은ID>",
    "selected_label": "happiness",
    "solve_time_ms": 3200,
    "retry_count": 0
  }' | python3 -m json.tool
```

---

## Feedback MLOps 완전 자동 모드

운영 중 사람 개입 없이 attempt_logs → 새 current pool 자동 갱신.
최초 문제은행 생성 단계(build_review_queue + human_review_server)를 제외하면
이후 운영은 전부 자동화된다.

### 트리거 조건 (4개 게이트 모두 통과 시 파이프라인 실행)

| 게이트 | 조건 | 설정 파일 |
|--------|------|-----------|
| G1 신규 attempt 건수 | ≥ 1,000건 (마지막 승격 이후) | `feedback_trigger_policy.yaml` |
| G2 문제별 평균 attempt | ≥ 30건/문제 | `feedback_trigger_policy.yaml` |
| G3 경과일 | ≥ 7일 | `feedback_trigger_policy.yaml` |
| G4 bad problem 비율 | ≥ 10% (quality_scores 존재 시) | `feedback_trigger_policy.yaml` |

### CronWorkflow 상태 확인

```bash
# 현재 CronWorkflow 상태 (suspend 여부 확인)
kubectl get cronworkflow captcha-bank-feedback-cron -n agami -o jsonpath='{.spec.suspend}'
# → false 이어야 자동 실행됨

# 다음 실행 예정 시각
kubectl get cronworkflow captcha-bank-feedback-cron -n agami

# 최근 실행 이력 (7개 보관)
argo list -n agami --prefix captcha-bank-feedback-cron
```

### 수동 즉시 실행

```bash
# 트리거 조건 확인 후 실행 (조건 미충족 시 check-trigger 만 실행, 나머지 skip)
argo submit --from workflowtemplate/captcha-bank-feedback-pipeline \
  -n agami --watch

# 조건 무시 강제 실행
argo submit --from workflowtemplate/captcha-bank-feedback-pipeline \
  -p force=true -n agami --watch

# 신규 검수 완료 문제 포함
argo submit --from workflowtemplate/captcha-bank-feedback-pipeline \
  -p force=true \
  -p new-problems-path=/data/new_reviewed_problems.csv \
  -n agami --watch
```

### 아티팩트 확인

파이프라인 실행 후 다음 위치에 아티팩트가 보존된다.

```bash
MODEL_STORE=/path/to/model-store/captcha_bank
VERSION=20260701_feedback   # 실제 버전으로 교체

# 트리거 결정 내역
cat ${MODEL_STORE}/.workdir/${VERSION}/trigger_decision.json

# attempt 집계 결과 (마지막 승격 이후)
cat ${MODEL_STORE}/.workdir/${VERSION}/aggregated_stats.csv | head

# 품질 레이블 분포
python3 -c "
import pandas as pd
df = pd.read_csv('${MODEL_STORE}/.workdir/${VERSION}/quality_scores.csv')
print(df['quality_label'].value_counts())
print(df['status'].value_counts())
"

# 승격 게이트 결과
cat ${MODEL_STORE}/candidates/${VERSION}/promotion_decision.json

# 현재 배포된 버전
python3 -c "
import json
print(json.loads(open('${MODEL_STORE}/current/metadata.json').read())['version'])
"
```

### 자동 reload 확인 (Serving API)

promote 완료 후 5분 이내에 Serving API가 새 pool을 자동 reload한다.

```bash
# 현재 로드된 풀 버전 확인
curl -s http://context-emotion-captcha-api:8083/health | python3 -m json.tool
# → version 필드가 새 버전으로 바뀌어야 함

# 실시간 로그 감시 (reload 확인)
kubectl logs -n agami -l app=context-emotion-captcha-api -f | grep "풀 갱신"
```

### compare gate 실패 시 동작

```
compare-candidate → exit 1 (gate 실패)
promote-model     → Skipped (Argo DAG: 이전 태스크 실패 시 downstream skip)
smoke-test        → Skipped
current/          → 변경 없음 (기존 모델 유지)
```

promotion_decision.json 에서 실패 원인을 확인할 수 있다.

```bash
# 어떤 게이트가 실패했는지 확인
cat ${MODEL_STORE}/candidates/${VERSION}/promotion_decision.json | python3 -m json.tool
```

---

## 빠른 시작

```bash
# 1. 코드 클론
git clone https://github.com/ai-denied/agami_mlops.git
cd agami_mlops

# 2. 의존성 설치 (서빙용)
pip install -r ml-pipeline/requirements-flashlight.txt

# 3. Flashlight API 로컬 실행
uvicorn flashlight.api.main:app --host 0.0.0.0 --port 8080

# 4. 헬스체크
curl http://localhost:8080/health
curl http://localhost:8080/model/info

# 5. 추론 테스트
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"trajectory": [...], "coordinate_mode": "normalized"}'
```

---

## 관련 문서

| 문서 | 위치 |
|------|------|
| K8s 배포 절차 | [`manifests/DEPLOY.md`](manifests/DEPLOY.md) |
| Argo Workflows 설정 | [`ml-pipeline/k8s/argo-workflows/README.md`](ml-pipeline/k8s/argo-workflows/README.md) |
| Context Emotion MLOps 설계 | [`ml-pipeline/context_emotion/MLOPS_OPERATION_DESIGN.md`](ml-pipeline/context_emotion/MLOPS_OPERATION_DESIGN.md) |
| captcha_bank MLOps 한 장 요약 | [`ml-pipeline/context_emotion/docs/captcha_bank_mlops_overview.md`](ml-pipeline/context_emotion/docs/captcha_bank_mlops_overview.md) |
| captcha_bank 운영 가이드 | [`ml-pipeline/context_emotion/captcha_bank/CAPTCHA_BANK_OPS.md`](ml-pipeline/context_emotion/captcha_bank/CAPTCHA_BANK_OPS.md) |
| 운영 회고록 | [`retrospectives/`](retrospectives/) |
