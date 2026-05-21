# =============================================================================
# captcha-frontend — Production image
# 빌더(node) 가 dist 산출 → nginx:alpine 가 정적 호스팅.
# 빌드: docker buildx build --platform linux/amd64 -f Dockerfile.prod ...
# =============================================================================

# ---------- builder ----------
FROM --platform=$BUILDPLATFORM node:20-alpine AS builder

WORKDIR /app

# package*.json 만 먼저 복사 → npm ci 캐시 효율
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# vite 가 .env.production 을 자동 인식 (mode=production 일 때)
RUN npm run build


# ---------- runtime ----------
FROM nginx:alpine AS runtime

# default.conf 제거 후 우리 설정으로 교체
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 빌드 산출물 정적 호스팅
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
