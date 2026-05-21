-- =====================================================================
-- 로컬 개발용 시드 데이터
-- ---------------------------------------------------------------------
-- - tenant 1개 (id 고정)
-- - tenant_settings 기본값
-- - allowed_origins: localhost:5173 (vite), localhost:3000, 210.109.53.140 (bastion)
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

INSERT INTO tenant_settings (tenant_id)
VALUES ('11111111-1111-1111-1111-111111111111')
ON CONFLICT (tenant_id) DO NOTHING;

-- HTTPS 전환 시 https:// 도 함께 추가해야 함 (docs/HTTPS_MIGRATION.md 참고).
INSERT INTO allowed_origins (tenant_id, origin)
VALUES
  ('11111111-1111-1111-1111-111111111111', 'http://localhost:5173'),
  ('11111111-1111-1111-1111-111111111111', 'http://localhost:3000'),
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
