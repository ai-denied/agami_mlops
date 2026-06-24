# flashlight-api 배포 절차

> **2026-06-24 클러스터 확인**: flashlight-inference-api는 `agami` namespace에서
> 1/1 Available로 실제 서빙 중이다 (`argo` namespace의 동명 Deployment는 0/2,
> 가동되지 않음 — deprecated, `manifests/flashlight-inference-api.yaml` 참고).
> 아래 명령어는 모두 `agami` 기준으로 갱신했다.

## 환경 구조

```
[data-prep-workspace Pod]  ← 현재 작업 위치 (Docker 없음)
        │ git push
        ▼
[GitHub: ai-denied/agami_mlops]
        │ webhook
        ▼
[빌드 환경] ──────────────────────────────────────────────────────
  A) GPU 서버 (Docker 있음) ← 즉시 수동 빌드 가능
  B) GitHub Actions runner  ← 자동화 권장
  C) Kaniko Job (K8s 내)    ← Docker/GPU 서버 없을 때
        │ docker push
        ▼
[Harbor: agami-captcha.cloud:8443/agami/flashlight-api:latest]
        │ kubectl rollout restart
        ▼
[K8s: flashlight-inference-api Deployment]
        │ PVC mount (ReadOnly)
        ▼
[model-store/flashlight/current/]
  mouse_gru.onnx
  normalizer.json
  metadata.json
```

---

## 방법 A — GPU 서버에서 직접 빌드 (즉시 가능)

> Docker가 설치된 GPU 서버 또는 다른 서버에서 실행한다.

```bash
# 1. 코드 클론 (최초 1회)
git clone https://github.com/ai-denied/agami_mlops.git
cd agami_mlops

# 이후 작업 시
git pull origin main

# 2. 이미지 빌드
docker build \
  -f ml-pipeline/Dockerfile.api \
  -t agami-captcha.cloud:8443/agami/flashlight-api:latest \
  .

# 3. Harbor 로그인 및 push
docker login agami-captcha.cloud:8443
docker push agami-captcha.cloud:8443/agami/flashlight-api:latest

# 4. K8s Deployment 재시작
kubectl rollout restart deployment/flashlight-inference-api -n agami
kubectl rollout status  deployment/flashlight-inference-api -n agami
```

---

## 방법 B — GitHub Actions 자동화 (권장)

### 1회성 설정

#### GitHub Secrets 등록
`https://github.com/ai-denied/agami_mlops/settings/secrets/actions`

| Secret 이름 | 값 |
|-------------|-----|
| `HARBOR_USERNAME` | Harbor 로그인 ID |
| `HARBOR_PASSWORD` | Harbor 비밀번호 또는 Robot Token |
| `KUBECONFIG_DATA` | `cat ~/.kube/config \| base64 -w0` 출력값 |

#### (권장) Self-hosted Runner — GPU 서버에 등록
```bash
# GPU 서버에서 실행
# GitHub: Settings → Actions → Runners → New self-hosted runner
mkdir actions-runner && cd actions-runner
curl -O -L https://github.com/actions/runner/releases/download/v2.x.x/actions-runner-linux-x64-2.x.x.tar.gz
tar xzf ./actions-runner-linux-x64-*.tar.gz
./config.sh --url https://github.com/ai-denied/agami_mlops --token <RUNNER_TOKEN>
sudo ./svc.sh install && sudo ./svc.sh start
```

`build-flashlight-api.yml`에서 아래 줄로 교체:
```yaml
runs-on: [self-hosted, gpu-server]
```

### 일상 배포 흐름
```bash
# data-prep-workspace Pod 안에서
git add ml-pipeline/
git commit -m "feat: update flashlight API"
git push origin main
# → GitHub Actions 자동 트리거 → 빌드 → push → rolling restart
```

---

## 방법 C — Kaniko (K8s 내 빌드, Docker/GPU 서버 불필요)

```bash
# 1. Harbor 인증 Secret 생성 (최초 1회)
kubectl create secret docker-registry harbor-registry-secret \
  --docker-server=agami-captcha.cloud:8443 \
  --docker-username=<USER> \
  --docker-password=<PASS> \
  -n default

# 2. Kaniko Job 실행
kubectl apply -f manifests/kaniko-build-flashlight-api.yaml

# 3. 빌드 로그 확인
kubectl logs -f job/kaniko-build-flashlight-api -n default

# 4. 완료 후 Deployment 재시작
kubectl rollout restart deployment/flashlight-inference-api -n agami
```

---

## 전체 K8s 리소스 적용 순서 (최초 배포)

```bash
# 1. PVC 적용 (model-store) — 실제 경로는 ml-pipeline/k8s/argo-workflows/ 아래에 있음
#    (manifest 자체에 namespace: agami가 명시되어 있어 -n 플래그는 참고용)
kubectl apply -f ml-pipeline/k8s/argo-workflows/pvc-flashlight.yaml
kubectl get pvc -n agami   # Bound 상태 확인

# 2. RBAC 적용
#    manifests/rbac-flashlight-workflow.yaml은 ServiceAccount/Role/RoleBinding을
#    argo namespace에 두고, agami namespace의 Deployment를 patch할 수 있도록
#    별도 Role([4])을 agami에 추가로 부여하는 cross-namespace 구성이다.
#    각 리소스의 namespace는 manifest에 명시되어 있으므로 -n 플래그는 불필요.
kubectl apply -f manifests/rbac-flashlight-workflow.yaml

# 3. 이미지 빌드 및 push (방법 A/B/C 중 선택)

# 4. Deployment + Service 적용 (운영 정본 — agami)
kubectl apply -f manifests/flashlight-inference-api/deployment.yaml -n agami

# 5. 배포 상태 확인
kubectl get pods -n agami -l app=flashlight-inference-api
kubectl rollout status deployment/flashlight-inference-api -n agami

# 6. 헬스체크 (port-forward 사용)
kubectl port-forward svc/flashlight-inference-api-svc 8080:80 -n agami &
curl http://localhost:8080/health
curl http://localhost:8080/model/info
```

---

## 모델 갱신 흐름 (운영 중 모델 교체)

```
GPU 서버에서 학습 완료
        │
        ▼
python -m flashlight.scripts.package_for_captcha_engine \
  --onnx       runs/.../mouse_gru_server_final_v2.onnx \
  --normalizer runs/.../mouse_normalizer_server_final_v2.joblib \
  --metadata   runs/.../mouse_metadata_server_final_v2.json \
  --version    v5_20260612
        │
        ▼
python -m flashlight.scripts.promote_model --version v5_20260612
        │  (model-store/flashlight/current/ 교체)
        ▼
kubectl rollout restart deployment/flashlight-inference-api -n agami
        │  (Pod 재시작 → 새 모델 자동 로드)
        ▼
curl http://.../health  →  "model_version": "v5_20260612"
```

> 이미지 재빌드 없이 모델만 교체할 수 있다.  
> model-store PVC가 Deployment와 Argo Workflow Pod에 공유 마운트되어 있기 때문이다.
