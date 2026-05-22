-- =====================================================================
-- 로컬 개발용 시드 데이터
-- ---------------------------------------------------------------------
-- - tenant 1개 (id 고정)
-- - tenant_settings: 3종 캡챠 (flashlight / face_mission / context_inference) 활성
-- - allowed_origins:
--     localhost:5173 (vite dev), localhost:3000,
--     localhost:8000 / 127.0.0.1:8000 (widget 통합 docker compose),
--     localhost:8001 (test-embed.html 부모 페이지 dev 서버),
--     210.109.53.140 (bastion)
-- - api_keys: client_key='ck_test', secret_key 평문='sk_test'
--
-- secret_hash 는 hmac.new(API_KEY_HMAC_PEPPER, 'sk_test', sha256).hexdigest().
-- 아래 값은 PEPPER='local-pepper-do-not-use-in-prod' 기준으로 계산됨.
-- .env 의 API_KEY_HMAC_PEPPER 를 바꿨다면 다시 계산해서 갱신할 것:
--   python3 -c "import hmac,hashlib;print(hmac.new(b'<PEPPER>', b'sk_test', hashlib.sha256).hexdigest())"
-- =====================================================================

INSERT INTO tenants (id, name)
VALUES ('11111111-1111-1111-1111-111111111111', 'Test Tenant')
ON CONFLICT (id) DO NOTHING;

-- enabled_kinds 를 명시적으로 채워둠 (schema default 는 ["flashlight"] 단일).
-- 새 DB 초기화 시 바로 3종 모두 활성 상태로 시작.
INSERT INTO tenant_settings (tenant_id, enabled_kinds)
VALUES (
  '11111111-1111-1111-1111-111111111111',
  '["flashlight","face_mission","context_inference"]'::jsonb
)
ON CONFLICT (tenant_id) DO NOTHING;

-- 기존 DB (ON CONFLICT 로 INSERT 스킵된 경우) 도 동일하게 맞춰주기 위한 UPDATE.
UPDATE tenant_settings
SET enabled_kinds = '["flashlight","face_mission","context_inference"]'::jsonb
WHERE tenant_id = '11111111-1111-1111-1111-111111111111';

-- HTTPS 전환 시 https:// 도 함께 추가해야 함 (docs/HTTPS_MIGRATION.md 참고).
INSERT INTO allowed_origins (tenant_id, origin)
VALUES
  ('11111111-1111-1111-1111-111111111111', 'http://localhost:5173'),
  ('11111111-1111-1111-1111-111111111111', 'http://localhost:3000'),
  ('11111111-1111-1111-1111-111111111111', 'http://localhost:8000'),
  ('11111111-1111-1111-1111-111111111111', 'http://127.0.0.1:8000'),
  ('11111111-1111-1111-1111-111111111111', 'http://localhost:8001'),
  ('11111111-1111-1111-1111-111111111111', 'http://210.109.53.140')
ON CONFLICT (tenant_id, origin) DO NOTHING;

INSERT INTO api_keys (tenant_id, name, client_key, secret_hash)
VALUES (
  '11111111-1111-1111-1111-111111111111',
  'local-test',
  'ck_test',
  '2ff33a43651c0773973a569c635e19f4d41f373a76ae7734d1ccd73d0185892d'
)
ON CONFLICT (client_key) DO NOTHING;
