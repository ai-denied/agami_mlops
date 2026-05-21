# 카카오 클라우드 K3s 배포 가이드

이 문서는 `captcha_engine` (FastAPI 백엔드) 와 `captcha-frontend` (Vite/React 프론트엔드) 를 카카오 클라우드 K3s 클러스터로 배포하는 절차입니다. HTTP 단계 MVP 기준이며, HTTPS 전환은 [HTTPS_MIGRATION.md](HTTPS_MIGRATION.md) 를 참고하세요.

---

## 0. 인프라 전제

| 노드 | 역할 |
|---|---|
| Bastion (210.109.53.140) | nginx-ingress-controller + Harbor 레지스트리 |
| Master | K3s control plane |
| Worker1 | `captcha-frontend` Pod |
| Worker2 (10.3.4.10) | `captcha-api` Pod |
| GPU | 향후 AI 모델 |

PostgreSQL / Redis 는 MVP 단계에서 K3s 안 별도 Pod 로 운영합니다. Harbor 프로젝트명은 `agami` 입니다.

---

## 1. 사전 준비 (1회성)

### 1-1. Worker 노드의 K3s 가 HTTP Harbor 를 신뢰하도록 설정

`/etc/rancher/k3s/registries.yaml` 을 모든 worker 노드에 작성:

```yaml
mirrors:
  "210.109.53.140":
    endpoint:
      - "http://210.109.53.140"
configs:
  "210.109.53.140":
    auth:
      username: <HARBOR_USER>
      password: <HARBOR_PASSWORD>
    tls:
      insecure_skip_verify: true
```

```bash
sudo systemctl restart k3s-agent
```

> Master 노드에서 동일한 설정이 필요한 경우 `k3s` 서비스를 restart.

### 1-2. Bastion 의 Nginx Ingress Controller 가 K3s Ingress 리소스를 watch 하도록 구성돼있는지 확인

(이미 존재해야 정상. 미구성이면 ingress-nginx 를 `kubectl apply -f` 로 설치.)

### 1-3. 작업 디렉토리 가정

이 가이드의 명령은 다음 두 디렉토리가 같은 부모에서 보이는 위치에서 실행한다고 가정합니다:

```
<workdir>/
  captcha_engine/
  captcha-frontend/
```

실제 사용자 환경(`/Users/shingwanho/Desktop/`) 에서는 두 폴더가 같은 부모에 있지 않을 수 있으므로, 필요 시 심볼릭 링크 또는 환경변수로 정리:

```bash
export ENGINE_DIR="/Users/shingwanho/Desktop/AI Captcha 프로젝트/captcha_engine"
export FRONTEND_DIR="/Users/shingwanho/Desktop/captcha-frontend"
```

이후 명령에서 상대 경로 `captcha_engine/...` 대신 `"$ENGINE_DIR/..."` 로 치환해 사용하세요.

---

## 2. 배포 단계

### 2-1. 환경변수 로드 & 매니페스트 placeholder 치환

`.env.deploy.example` 을 복사해서 값을 채운 뒤 로드:

```bash
cd "$ENGINE_DIR"
cp .env.deploy.example .env.deploy
# 편집해서 HARBOR_PASSWORD / API_KEY_HMAC_PEPPER / CAPTCHA_TOKEN_SECRET 채우기
set -a; source .env.deploy; set +a
```

매니페스트 사본 만들고 `HARBOR_URL`, `VERSION` 치환:

```bash
rm -rf "$ENGINE_DIR/k8s.rendered" "$FRONTEND_DIR/k8s.rendered"
cp -r "$ENGINE_DIR/k8s"  "$ENGINE_DIR/k8s.rendered"
cp -r "$FRONTEND_DIR/k8s" "$FRONTEND_DIR/k8s.rendered"

find "$ENGINE_DIR/k8s.rendered" "$FRONTEND_DIR/k8s.rendered" -name '*.yaml' -exec \
  sed -i.bak \
    -e "s|HARBOR_URL|${HARBOR_URL}|g" \
    -e "s|VERSION|${VERSION}|g" \
  {} \;

# .bak 파일 정리
find "$ENGINE_DIR/k8s.rendered" "$FRONTEND_DIR/k8s.rendered" -name '*.yaml.bak' -delete
```

### 2-2. 이미지 빌드 & Harbor push

빌드 호스트에서 (M1/M2 맥에서 빌드한다면 `--platform linux/amd64` 필수):

```bash
docker login "$HARBOR_URL" -u "$HARBOR_USER" -p "$HARBOR_PASSWORD"

# 백엔드
docker buildx build --platform linux/amd64 \
  -f "$ENGINE_DIR/Dockerfile.prod" \
  -t "$HARBOR_URL/$HARBOR_PROJECT/captcha-api:$VERSION" \
  "$ENGINE_DIR" --push

# 프론트엔드
docker buildx build --platform linux/amd64 \
  -f "$FRONTEND_DIR/Dockerfile.prod" \
  -t "$HARBOR_URL/$HARBOR_PROJECT/captcha-frontend:$VERSION" \
  "$FRONTEND_DIR" --push
```

### 2-3. Secret 생성 (Master 의 `kubectl` 사용)

```bash
# API 비밀 — 절대 .env.deploy 의 placeholder 그대로 쓰지 말 것.
kubectl create secret generic captcha-api-secrets \
  --from-literal=API_KEY_HMAC_PEPPER="$API_KEY_HMAC_PEPPER" \
  --from-literal=CAPTCHA_TOKEN_SECRET="$CAPTCHA_TOKEN_SECRET"

# Postgres 비밀번호 — backend-configmap.yaml 의 DATABASE_URL 과 일치해야 함.
# MVP 기본은 'captcha'. 다른 값 사용 시 ConfigMap 의 DATABASE_URL 도 함께 수정.
kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD"
```

### 2-4. Postgres init ConfigMap 생성 (스키마 + 시드 데이터)

```bash
kubectl create configmap postgres-init \
  --from-file=01-schema.sql="$ENGINE_DIR/app/db/schema.sql" \
  --from-file=02-seed.sql="$ENGINE_DIR/app/db/seed.sql"
```

> 주의: 이 ConfigMap 은 `postgres-init-configmap.yaml` 의 빈 placeholder 를 대체합니다.

### 2-5. 매니페스트 적용 (순서 중요)

```bash
# 데이터 계층 먼저
kubectl apply -f "$ENGINE_DIR/k8s.rendered/postgres-pvc.yaml"
kubectl apply -f "$ENGINE_DIR/k8s.rendered/postgres-deployment.yaml"
kubectl apply -f "$ENGINE_DIR/k8s.rendered/postgres-service.yaml"
kubectl apply -f "$ENGINE_DIR/k8s.rendered/redis-deployment.yaml"
kubectl apply -f "$ENGINE_DIR/k8s.rendered/redis-service.yaml"

# Postgres ready 대기 (init 스크립트 실행 후)
kubectl wait --for=condition=ready pod -l app=postgres --timeout=180s

# 백엔드
kubectl apply -f "$ENGINE_DIR/k8s.rendered/backend-configmap.yaml"
kubectl apply -f "$ENGINE_DIR/k8s.rendered/backend-deployment.yaml"
kubectl apply -f "$ENGINE_DIR/k8s.rendered/backend-service.yaml"

# 프론트엔드
kubectl apply -f "$FRONTEND_DIR/k8s.rendered/frontend-deployment.yaml"
kubectl apply -f "$FRONTEND_DIR/k8s.rendered/frontend-service.yaml"

# Ingress (마지막)
kubectl apply -f "$ENGINE_DIR/k8s.rendered/ingress.yaml"
```

### 2-6. DB 초기화 확인

```bash
kubectl exec -it deploy/postgres -- psql -U captcha -d captcha \
  -c "SELECT client_key FROM api_keys;"
# → ck_test 한 줄 나오면 OK
```

빈 결과가 나오면 PVC 가 이미 데이터를 갖고 있던 것이므로 강제 적용:

```bash
kubectl exec -i deploy/postgres -- psql -U captcha -d captcha < "$ENGINE_DIR/app/db/schema.sql"
kubectl exec -i deploy/postgres -- psql -U captcha -d captcha < "$ENGINE_DIR/app/db/seed.sql"
```

### 2-7. 외부 접속 검증

```bash
curl http://210.109.53.140/                     # 프론트 index.html
curl http://210.109.53.140/api/health           # {"status":"ok"}

curl -X POST http://210.109.53.140/api/v1/challenges \
  -H "X-Captcha-Client-Key: ck_test" \
  -H "Content-Type: application/json" \
  -d '{"kind":"flashlight","difficulty":"easy"}'
# → spec JSON 반환되면 OK
```

브라우저에서 `http://210.109.53.140` 접속해 캡챠 데모 전체 흐름(시작 → 클릭 → success/fail) 을 확인.

---

## 3. 트러블슈팅

| 증상 | 원인 / 점검 |
|---|---|
| `ImagePullBackOff` | 워커의 `/etc/rancher/k3s/registries.yaml` 누락 또는 비밀번호 오타. `kubectl describe pod <name>` 으로 상세 확인. |
| Postgres `CrashLoopBackOff` | PVC 권한 문제 또는 `POSTGRES_PASSWORD` 변경 후 PVC 가 옛 비밀번호로 초기화돼있는 충돌. `kubectl logs deploy/postgres` 확인. PVC 를 지우고 (`kubectl delete pvc postgres-data`) 재생성하거나 비밀번호 동기화. |
| API 가 `relation "api_keys" does not exist` | init ConfigMap 적용 안 됐거나 PVC 가 빈 상태가 아니라 init 이 스킵됨. 2-6 의 강제 적용 명령 실행. |
| `502 Bad Gateway` | Pod 가 ready 아님. `kubectl get pods`, `kubectl logs deploy/<...>`. readiness probe 가 통과하는지 확인. |
| `404 on /api` | Ingress 의 `rewrite-target`/`use-regex` annotation 또는 `ingressClassName: nginx` 누락. `kubectl describe ingress captcha-ingress` 확인. |
| CORS 에러 (브라우저 콘솔) | 프론트의 `VITE_API_URL` 이 `/api` (상대) 인지 확인. 절대 URL 이면 ConfigMap 의 `CORS_ORIGINS` 와 백엔드 `allowed_origins` 시드에 해당 origin 이 들어있어야 함. |
| Origin 차단 (`origin_not_allowed`) | DB `allowed_origins` 테이블에 `Origin` 헤더 값이 없음. seed.sql 갱신 후 강제 적용 또는 `INSERT` 직접 실행. |

---

## 4. 운영 단계 보안 권장 사항

- **PEPPER / TOKEN_SECRET 회전**: `openssl rand -hex 32` 로 새 값을 만들고 Secret 재생성 → `kubectl rollout restart deployment/captcha-api`. 회전 시점의 미만료 captcha_token 들은 무효화됨.
- **이미지 태그 고정**: `latest` 금지. semver(`v0.1.0`) 또는 git short sha. 디플로이먼트의 `imagePullPolicy: IfNotPresent` 와 함께 사용해 의도치 않은 자동 갱신 차단.
- **PVC 백업**: K3s `local-path` 는 단일 노드 디스크에 의존. 정기적으로 `pg_dump` 를 cronjob 으로 외부 스토리지에 떠두기. PVC 삭제 시 데이터 영구 손실 위험.
- **Postgres 비밀번호 강력화**: MVP 기본 `captcha` 는 운영 사용 금지. `openssl rand -base64 24` 등으로 교체 후 ConfigMap 의 DATABASE_URL 도 동기화.
- **Redis 영속화**: 답안/토큰이 휘발돼도 사용자 영향이 "다시 시도" 수준이라 MVP 무시 가능. 그러나 rate limit 카운터까지 휘발되면 어뷰저가 잠시 우회 가능 → 영속화 필요 시 `redis-deployment.yaml` 에 PVC + `--appendonly yes` 추가.
- **Ingress allowlist / WAF**: Bastion 의 nginx 에 IP allowlist, rate limit, ModSecurity 등 추가.
- **HTTPS 전환**: 우선순위 높음. [HTTPS_MIGRATION.md](HTTPS_MIGRATION.md) 참조.
- **K8s RBAC**: 배포 계정과 운영 조회 계정 분리. ServiceAccount 별 token rotation.
- **Harbor 권한**: `agami` 프로젝트에 push 권한 가진 계정/로봇 토큰을 분리. CI/CD 에서는 robot account 사용.
