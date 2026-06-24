# 회고록 — 모델 승격 파이프라인 안정성 보강 + 운영 namespace 정리 + 실시간 시간정규화 버그 수정

**세션 기간**: 2026-06-24 (단일 세션, 07:00~09:00경)
**작성**: Claude (작업 회고)
**관련 커밋**: `618ecd15` ~ `23a90a68` (main)

---

## 1. 오늘의 목표

이전 세션에서 `agami_mlops` 레포 구조를 git으로 연결하고 전체 폴더 구조를 분석한 뒤, "모델이 계속 승격되는 구조인데 API 제공에 문제가 없는지" 검증해달라는 요청에서 시작했다. 검증 결과 실제로 여러 개의 안전성 구멍이 발견되어, 하루 동안 다음을 순차적으로 처리했다.

1. flashlight / facial_recognition 모델 승격 파이프라인의 안전성 진단
2. flashlight에 ONNX runtime contract 게이트 + 승격 후 자동 rollout restart/smoke test 추가
3. 클러스터 실측으로 발견된 `argo` vs `agami` namespace 불일치 정리 (CI 매니페스트 경로, deprecated 처리)
4. facial_recognition도 동일한 수준으로 안전성 보강
5. 캡차위젯팀 연동 중 발견된 `frame_interval=1` 고정 가정 버그 수정 + 신규 연동 경로(`extract_from_landmarks`) 추가
6. 핸드오프 스냅샷 폴더가 운영 빌드/배포에 의도치 않게 영향을 주던 문제 정리

---

## 2. 진단 — 모델 승격이 API 안정성을 깨뜨릴 수 있는 지점들

요청을 받고 코드를 직접 추적해 다음 두 가지 구멍을 확인했다.

- **승격 ≠ 즉시 반영**: `flashlight`/`facial_recognition` API 서버 모두 FastAPI `lifespan`에서 ONNX 모델을 **서버 기동 시 한 번만** 로드하는 싱글턴 구조(`api/loader.py`). `promote_model.py`가 `model-store/.../current/`를 원자적으로 교체해도, 이미 떠 있는 pod는 재시작 전까지 옛 모델을 계속 서빙한다.
- **승격 전 ONNX 입출력 검증 부재**: `flashlight`의 `compare_candidate.py`는 정확도 지표만 비교하고 텐서 이름/shape/dtype은 전혀 안 본다. `facial_recognition`은 더 심해서 `compare_candidate.py` 자체가 없고, 자동 파이프라인(`run_model_update_pipeline.py`)이 `promote(..., skip_validate=True)`를 **고정값으로** 넘겨 그 약한 검증마저 항상 스킵하고 있었다.

이 두 가지를 같은 패턴으로 양쪽 모두에 적용해 보강했다.

---

## 3. flashlight — ONNX contract 게이트 + 자동 restart/smoke-test

### 한 일
- `flashlight/config/runtime_contract.yaml` 신설: `x_seq`/`lengths`/`x_static` → `bot_risk_score`의 이름/shape/dtype을 고정 계약으로 명시.
- `flashlight/evaluation/onnx_contract_check.py` 신설: 실제 onnx 그래프(onnxruntime으로 읽음)를 contract.yaml + metadata.json 양쪽과 대조. metadata.json 필드 존재 여부만 보는 게 아니라 실제 그래프와 직접 비교.
- `compare_candidate.py`/`promote_model.py`에 이 검증을 하드 게이트로 추가 — 성능 지표가 전부 PASS여도 contract가 FAIL이면 전체 판정 FAIL.
- `run_model_update_pipeline.py`의 `run_promote`에서 `skip_validate=True` → `False`로 변경(이중 방어선 확보).
- Argo Workflow(`manifests/flashlight-model-update-workflow.yaml`)에 `restart-inference-api`(kubectl rollout restart + rollout status 대기) → `smoke-test-api`(`/health` 버전 일치 + `/predict` 정상 동작 확인) 단계 추가, RBAC(`manifests/rbac-flashlight-workflow.yaml`)에 agami namespace deployment patch 권한 추가.

### 검증
더미 onnx/metadata로 양성(통과)·음성(shape 위조 → 차단, exit 1) 케이스를 직접 실행해 확인. 테스트 중 실수로 `git status`에 `model-store/flashlight` 추적 파일이 안 걸려있어 안전하게 정리됨.

---

## 4. namespace 불일치 발견 및 정리 — `argo` vs `agami`

### 발견
flashlight Argo Workflow에 namespace를 채우면서 `manifests/flashlight-inference-api.yaml`(namespace `argo`)과 `manifests/flashlight-inference-api/deployment.yaml`(namespace `agami`) 두 매니페스트가 동시에 존재하는 걸 발견. 정적 분석(git log, CI 트리거 경로, ArgoCD Application 정의)으로 추적한 뒤, **사용자가 실제 클러스터에서 직접 확인**:

| namespace | READY | AVAILABLE |
|---|---|---|
| `agami` | 1/1 | 1 |
| `argo` | 0/2 | 0 |

`replicas` 필드도 agami=1, argo=2로 매니페스트와 일치 — agami가 실제 운영, argo는 죽은 레거시였음이 명확히 확인됨.

### 조치
- CI(`build-flashlight-api.yml`)의 `MANIFEST_FILE`을 `manifests/flashlight-inference-api.yaml`(argo) → `manifests/flashlight-inference-api/deployment.yaml`(agami)로 변경 — 그동안 CI가 죽은 namespace 매니페스트만 갱신해온 게 원인.
- `manifests/flashlight-inference-api.yaml`을 Deployment/Service 리소스 없는 deprecated 안내 문서로 교체 (실수로 apply해도 무영향).
- `manifests/DEPLOY.md` 런북 전체를 `-n agami`로 교정, 존재하지 않는 경로(`manifests/pvc-flashlight.yaml`)도 실제 경로(`ml-pipeline/k8s/argo-workflows/pvc-flashlight.yaml`)로 수정.
- 작업 중 `model-store/facial_recognition/current/`의 추적 파일을 실수로 `rm -rf`했다가 `git checkout`으로 즉시 복구한 사고가 있었음 (운영 모델 파일이라 git에 커밋되어 있었던 게 다행).

### 근본 원인 메모
2026-06-15 커밋(`b590febc "fix: unify namespace to agami"`)이 `ml-pipeline/k8s/argo-workflows/*`와 `manifests/flashlight-inference-api/deployment.yaml`만 고치고 **CI가 실제로 갱신하는 root `manifests/flashlight-inference-api.yaml`은 빠뜨려서**, agami로의 마이그레이션이 절반만 적용된 상태로 9일간 방치돼 있었다.

---

## 5. facial_recognition — 동일 수준 보강

flashlight에서 검증한 패턴을 그대로 facial_recognition에 적용했다. 다만 facial_recognition은 Argo Workflow/전용 Docker 이미지 자체가 없어서, restart/smoke-test를 **`run_model_update_pipeline.py` 안에 직접 내장**하는 방식을 선택(사용자 결정).

### 한 일
- `facial_recognition/config/runtime_contract.yaml` + `evaluation/onnx_contract_check.py` 신설 (flashlight와 동일 패턴, metadata.json 스키마는 flat 필드로 다름).
- `promote_model.py`에 contract 검증 추가.
- `run_model_update_pipeline.py`: `skip_validate=True` 제거, STEP 4(`kubectl rollout restart` + `rollout status`)/STEP 5(`smoke_test_model.py`) 추가, `--skip-restart` 등 CLI 옵션 추가.
- `facial_recognition/scripts/smoke_test_model.py` 신설 (`/health` 버전 일치 + `/predict` 정상 동작, 재시도 로직 포함).
- `facial_recognition/requirements-ops.txt` 신설 — 이 파이프라인이 어느 Docker 이미지에도 안 묶여 있어서 의존성을 선언할 곳이 없었음(context_emotion의 `requirements-ops.txt` 패턴을 따름).

### 검증
더미 onnx/metadata로 양성/음성 케이스 확인. **실제 운영 모델**(`current/`, `v3_w8_aspectfix_20260618`)도 새 contract 검증을 통과하는지 직접 확인 — 기존 운영에 영향 없음.

---

## 6. 캡차위젯팀 연동 중 발견된 `frame_interval=1` 고정 가정 버그

### 발견 경위
위젯팀이 face-liveness-api `/predict` 연동 중 실시간 프레임 간격이 67~119ms(평균 ~83ms)로 불균일하다고 보고. 코드를 추적한 결과, **ML팀이 이미 캡차엔진팀에 넘긴 핸드오프 코드(`face_feature_extractor.py`) 자체에 버그가 있었음**:

```python
# 기존: "실시간 캡처는 항상 R_live_clip과 같은 30fps"라고 가정
FRAME_INTERVAL = 1
```

학습 코드(`extract_features_time_norm.py`)의 `frame_interval`은 실측 ms가 아니라 파일명 인덱스 간격 추정값이며, R_live_clip은 항상 `frame_interval=1`로 추정되어 DEFAULT_FPS=30(33.33ms) 네이티브 캡처를 가정하고 있었다. 실측 평균 83ms를 그대로 fi=1로 나누면 velocity_tn이 학습분포보다 **평균 ~2.5배 과대**해지는, 과거 R_live_clip vs S_dataset_sequence 종횡비 버그와 같은 종류의 문제가 재현되는 상황이었다.

### 조치
- `extract_from_frames()`/`FrameBuffer`가 `timestamps_ms`를 받을 수 있게 확장. `feature_tn = raw_diff * (1000/30) / Δt_ms` 공식으로 실측 Δt를 학습 기준 단위(33.33ms)에 맞춰 보정.
- Δt 이상치(프레임 드롭, timestamp 역전) 방어: 비정상 구간은 median Δt로 대체, 최소 1ms clamp.
- timestamp 미제공 시 기존 동작(fi=1) 유지 + 로그 경고로 정확도 리스크 명시.
- 후속 질문(엔진은 위젯의 랜드마크+timestamp만 받고 x_seq는 엔진이 직접 만드는 구조)에 맞춰 `extract_from_landmarks(landmarks_list, timestamps_ms, widths, heights)` 진입점 추가 — raw MediaPipe 랜드마크(`dict[index, [x,y]]`)를 직접 받아 MediaPipe 재추론 없이 처리. 종횡비 보정용 `widths`/`heights`를 필수 인자로 강제(과거 버그 재발 방지). 이 경로만 쓰면 mediapipe 설치도 불필요하도록 FaceMesh 초기화를 lazy로 변경.

### 검증
위젯팀이 설명한 시나리오(480×480 정사각형, raw landmark, 67~119ms 불균일 timestamp)를 그대로 시뮬레이션해 end-to-end 확인. legacy fi=1 대비 보정된 velocity가 정확히 기대 비율(`fi_eff`)만큼 줄어드는지, 필수 인덱스 누락/width·height 누락 방어 로직까지 전부 단위 테스트.

---

## 7. 핸드오프 스냅샷 폴더의 의도치 않은 빌드/배포 영향 정리

위 5번 작업을 하면서 `captcha_engine_team_handoff/`만 수정했는데도 face-liveness-api 이미지가 매번 재빌드·재배포(rolling restart)되는 것을 git log에서 발견(`Dockerfile.face-api`가 `facial_recognition/` 전체를 COPY, CI 트리거 경로도 전체 글롭).

### 조치
- `ml-pipeline/.dockerignore` 신설(face-api 등 모든 ml-pipeline Dockerfile의 실제 빌드 컨텍스트 위치) — `captcha_engine_team_handoff/`, `ml_pipeline_team_handoff/`, `agami_liveness_final_package/` 제외.
- `build-face-api.yml`에 해당 3개 디렉토리 음수 path 패턴 추가 — 이 폴더만 바뀌면 워크플로우 자체가 트리거되지 않음.
- 폴더 자체는 레포에 그대로 유지(삭제하지 않음, 다른 팀에 전달용으로 계속 필요).
- 추가로 사용자가 지적한 "추가한 한글 주석이 터미널에서 mojibake로 보인다" 피드백을 받아, 이번에 새로 추가한 주석들(workflow, dockerignore, Dockerfile.face-api)만 짧은 영어로 교체. 기존에 있던 한글 주석들은 그대로 둠(범위 확대 안 함).

### 부수적으로 발견한 사실 (미조치, 메모만)
- 저장소 루트의 `.dockerignore`는 모든 ml-pipeline 빌드(`context: ./ml-pipeline`)에 **애초에 적용되지 않고 있었다** — Docker는 빌드 컨텍스트 디렉토리 안의 `.dockerignore`만 읽는데, 컨텍스트가 `ml-pipeline/`이므로 레포 루트 파일은 무관했음. `ml-pipeline/.dockerignore`가 이번에 처음 생겼다.
- `Dockerfile.face-api` 헤더의 로컬 빌드 예시 명령어(`docker build ... .`)가 컨텍스트를 레포 루트로 잘못 안내하고 있음(실제 CI는 `./ml-pipeline`을 씀) — 문서만의 문제이지만 다음에 누가 그대로 따라하면 빌드가 깨질 수 있음.

---

## 8. 좋았던 점 (Went Well)

- **flashlight에서 검증한 패턴을 facial_recognition에 그대로 재사용**: contract.yaml + onnx_contract_check.py 구조를 복제하면서도, 각 모델의 실제 metadata 스키마 차이(nested `onnx_spec` vs flat 필드)를 반영해 적절히 변형함.
- **모든 변경을 실제로 실행해서 검증**: 더미 모델 생성 → 양성/음성 케이스 → 실제 운영 모델까지 대조. "코드만 보고 맞다고 주장"하지 않고 직접 돌려본 게 여러 번 숨은 문제(예: model-store 추적 파일 실수 삭제)를 그 자리에서 잡아냄.
- **클러스터 실측을 기다림**: namespace 문제를 추측으로 끝내지 않고, 정적 분석 결과를 먼저 정리해서 보고한 뒤 사용자가 실제 `kubectl get deploy` 결과를 가져올 때까지 기다렸다가 그걸 근거로 조치함 — 잘못된 가정으로 무언가를 고쳤다가 틀릴 위험을 피함.
- **핸드오프 코드의 버그를 "위젯팀 책임이 아니라 ML팀 책임"으로 명확히 정리**: 사용자가 스스로 "이거 우리가 해야 하는 거 아니야?"라고 알아차렸고, 이미 배포된 handoff 코드 자체에 같은 버그가 박혀있던 것을 코드로 직접 찾아 보여줌.

## 9. 아쉬운 점 / 리스크 (Could Be Better)

- **🔴 model-store 추적 파일 실수 삭제 사고**: facial_recognition contract 검증 테스트 중 `rm -rf model-store/facial_recognition`을 실행해 실제 운영 모델 파일(`current/face_liveness.onnx` 등)이 git에 커밋되어 있다는 걸 모른 채 지웠다가, 다행히 추적 파일이라 `git checkout`으로 즉시 복구함. model-store가 보통 PVC 마운트/gitignore 대상이라는 가정이 틀렸던 사례 — 테스트 전에 `git status`로 먼저 확인하는 습관이 필요했음.
- **facial_recognition의 restart/smoke-test가 Argo Workflow가 아니라 Python 스크립트에 내장됨**: flashlight와 아키텍처가 비대칭이 됨. 나중에 facial_recognition도 전용 Dockerfile/Argo Workflow를 갖추게 되면, 지금 만든 내장형 로직과 중복되거나 충돌할 수 있음 — 그때 통합이 필요.
- **argo namespace의 죽은 Deployment(0/2)가 아직 클러스터에 남아있음**: 레포 쪽 정리(매니페스트 deprecated 처리, CI 경로 변경)는 끝났지만, 실제 클러스터 리소스 삭제는 사용자가 직접 확인 후 진행해야 하는 채로 남아있음 — 이번 세션에서 그 사람이 가능한 명령어만 안내함.
- **root `.dockerignore`가 무용지물이라는 사실을 이번에 우연히 발견**: facial_recognition 핸드오프 작업을 안 했다면 영영 몰랐을 수 있음. 다른 Dockerfile들(flashlight, training, dashboard)도 같은 맹점이 있는지 전체 점검이 안 됨.
- **위젯팀의 종횡비 보정 여부(raw vs corrected) 확인에 의존**: 이번엔 운 좋게 위젯팀이 명확히 "raw, 정사각형 480x480이라 보정이 항등"이라고 답해줘서 안전하게 구현했지만, 만약 잘못 답했거나 나중에 카메라 비율이 바뀌는데 그 사실이 전달 안 되면 같은 종류의 스케일 버그가 또 재발할 수 있음 — `widths`/`heights`를 필수 인자로 만든 것 정도가 유일한 방어선.

## 10. 다음 액션 제안

1. argo namespace의 죽은 `flashlight-inference-api`(0/2) Deployment/Service를 클러스터에서 확인 후 삭제, `manifests/flashlight-inference-api.yaml` 파일도 git에서 제거.
2. facial_recognition도 전용 Dockerfile + Argo Workflow를 갖추게 되면, 지금 Python 스크립트에 내장한 restart/smoke-test 로직을 Argo Workflow 단계로 옮기고 중복 제거.
3. `Dockerfile.flashlight`/`Dockerfile.training`/`Dockerfile.dashboard`도 동일한 `.dockerignore` 컨텍스트 맹점이 있는지 점검.
4. `Dockerfile.face-api` 헤더의 로컬 빌드 예시 명령어를 실제 컨텍스트(`./ml-pipeline`)에 맞게 수정.
5. 캡차위젯팀의 `widths`/`heights` 입력이 실제로 매 요청마다 정확히 들어오는지, 운영 로그의 `used_real_timestamps` 필드로 모니터링.
6. context_emotion에도 같은 클래스의 문제(승격 전 contract 검증, 승격 후 자동 반영)가 있는지 — 모델이 아직 안 나온 단계라 당장은 아니지만, 모델 도착 시 가장 먼저 점검할 목록에 추가.
