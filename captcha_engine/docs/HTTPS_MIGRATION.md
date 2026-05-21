# HTTPS 전환 가이드

HTTP MVP 에서 HTTPS 운영으로 전환할 때의 변경 사항을 정리합니다. 도메인이 확보된 시점에 수행하세요.

`<DOMAIN>` 는 실제 운영 도메인 (예: `captcha.example.com`) 으로, `<HARBOR_DOMAIN>` 은 Harbor 가 노출될 도메인 (예: `harbor.example.com`) 으로 치환해서 읽으세요.

---

## 1. 인증서 발급 인프라

### cert-manager 설치 (1회성)

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
```

### ClusterIssuer 등록 (Let's Encrypt 예시)

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: ops@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
```

사내 CA 를 쓴다면 Let's Encrypt 대신 `ca`/`vault` issuer 를 사용.

---

## 2. Harbor HTTPS 전환

### 2-1. Harbor 인증서 적용

Harbor 운영 절차에 따라 도메인 + 인증서를 설정합니다 (Harbor 공식 문서: `Configure HTTPS Access to Harbor`).

### 2-2. K3s worker 노드의 `registries.yaml` 수정

```yaml
mirrors:
  "<HARBOR_DOMAIN>":
    endpoint:
      - "https://<HARBOR_DOMAIN>"
configs:
  "<HARBOR_DOMAIN>":
    auth:
      username: <HARBOR_USER>
      password: <HARBOR_PASSWORD>
    # tls.insecure_skip_verify 항목 삭제 (또는 false)
```

```bash
sudo systemctl restart k3s-agent
```

### 2-3. 기존 HTTP IP 키 (`210.109.53.140`) 제거

위 yaml 에서 IP 기반 설정을 삭제. 기존 이미지 참조 (`210.109.53.140/...`) 는 새 도메인 (`<HARBOR_DOMAIN>/...`) 으로 갱신 필요.

---

## 3. 코드 / 매니페스트 변경 사항

### 3-1. `.env.deploy`

```diff
- HARBOR_URL=210.109.53.140
+ HARBOR_URL=<HARBOR_DOMAIN>
- PUBLIC_URL=http://210.109.53.140
+ PUBLIC_URL=https://<DOMAIN>
```

### 3-2. `captcha_engine/k8s/backend-configmap.yaml`

```diff
- CORS_ORIGINS: "http://210.109.53.140,http://localhost:5173"
+ CORS_ORIGINS: "https://<DOMAIN>,http://localhost:5173"
```

### 3-3. `captcha_engine/app/db/seed.sql` (신규 배포 시) 또는 직접 INSERT (기존 DB)

```sql
INSERT INTO allowed_origins (tenant_id, origin)
VALUES ('11111111-1111-1111-1111-111111111111', 'https://<DOMAIN>')
ON CONFLICT (tenant_id, origin) DO NOTHING;
-- 필요하다면 기존 http origin 을 제거:
-- DELETE FROM allowed_origins WHERE origin = 'http://210.109.53.140';
```

### 3-4. `captcha_engine/k8s/ingress.yaml`

다음 두 가지 추가:

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"            # 추가
    cert-manager.io/cluster-issuer: letsencrypt-prod            # 추가
spec:
  ingressClassName: nginx
  tls:                                                          # 추가
    - hosts:
        - <DOMAIN>
      secretName: captcha-tls
  rules:
    - host: <DOMAIN>                                            # host 추가
      http:
        paths:
          - path: /api(/|$)(.*)
            ...
```

### 3-5. 프론트엔드

`VITE_API_URL=/api` 는 상대 경로이므로 **수정 불필요**. 도메인이 바뀌어도 같은 호스트의 `/api` 로 라우팅됨.

`captcha-frontend/.env.production` 도 그대로.

---

## 4. 재배포 절차

```bash
# .env.deploy 수정 후
set -a; source "$ENGINE_DIR/.env.deploy"; set +a

# 매니페스트 재렌더링
rm -rf "$ENGINE_DIR/k8s.rendered" "$FRONTEND_DIR/k8s.rendered"
cp -r "$ENGINE_DIR/k8s"  "$ENGINE_DIR/k8s.rendered"
cp -r "$FRONTEND_DIR/k8s" "$FRONTEND_DIR/k8s.rendered"
find "$ENGINE_DIR/k8s.rendered" "$FRONTEND_DIR/k8s.rendered" -name '*.yaml' -exec \
  sed -i.bak \
    -e "s|HARBOR_URL|${HARBOR_URL}|g" \
    -e "s|VERSION|${VERSION}|g" {} \;

# 새 도메인으로 이미지 다시 push
docker login "$HARBOR_URL" -u "$HARBOR_USER" -p "$HARBOR_PASSWORD"
docker buildx build --platform linux/amd64 \
  -f "$ENGINE_DIR/Dockerfile.prod" \
  -t "$HARBOR_URL/$HARBOR_PROJECT/captcha-api:$VERSION" \
  "$ENGINE_DIR" --push
docker buildx build --platform linux/amd64 \
  -f "$FRONTEND_DIR/Dockerfile.prod" \
  -t "$HARBOR_URL/$HARBOR_PROJECT/captcha-frontend:$VERSION" \
  "$FRONTEND_DIR" --push

# ConfigMap / Ingress / Deployment 재적용
kubectl apply -f "$ENGINE_DIR/k8s.rendered/backend-configmap.yaml"
kubectl apply -f "$ENGINE_DIR/k8s.rendered/ingress.yaml"
kubectl rollout restart deployment/captcha-api
kubectl rollout restart deployment/captcha-frontend

# DNS 가 새 도메인 → Bastion IP 로 가리키게 설정.
# cert-manager 가 처음 인증서 발급 시 1~2분 소요. 발급 진행 상황:
kubectl describe certificate captcha-tls
```

---

## 5. 검증

```bash
# HTTPS 응답 확인
curl -v https://<DOMAIN>/api/health
# → HTTP/2 200 + 인증서 chain 정상 (cert-manager 가 자동 발급한 cert)

# HTTP → HTTPS 자동 리다이렉트
curl -v http://<DOMAIN>/
# → 308 Permanent Redirect Location: https://<DOMAIN>/

# 캡챠 발급
curl -X POST https://<DOMAIN>/api/v1/challenges \
  -H "X-Captcha-Client-Key: ck_test" \
  -H "Content-Type: application/json" \
  -H "Origin: https://<DOMAIN>" \
  -d '{"kind":"flashlight","difficulty":"easy"}'
# → 200 OK + spec JSON
```

브라우저에서 `https://<DOMAIN>` 접속 → 자물쇠 아이콘 + 캡챠 정상 동작 확인.

---

## 6. 변경 영향 받는 파일 한눈에

| 파일 | HTTPS 전환 시 변경 |
|---|---|
| `.env.deploy` | `HARBOR_URL`, `PUBLIC_URL` |
| `captcha_engine/k8s/backend-configmap.yaml` | `CORS_ORIGINS` https 로 교체 |
| `captcha_engine/k8s/ingress.yaml` | `tls:`, `host:`, cert-manager annotation, ssl-redirect annotation |
| `captcha_engine/app/db/seed.sql` (신규 배포) | `INSERT INTO allowed_origins ... 'https://<DOMAIN>'` |
| 기존 DB | `INSERT` 또는 `DELETE` 로 `allowed_origins` 동기화 |
| Worker 노드 `/etc/rancher/k3s/registries.yaml` | endpoint http→https, `insecure_skip_verify` 제거 |
| Harbor 자체 설정 | 인증서 적용 (Harbor 공식 문서) |
| 프론트엔드 `.env.production` | **변경 없음** (`VITE_API_URL=/api` 상대 경로) |
| 프론트엔드 코드 | **변경 없음** |
