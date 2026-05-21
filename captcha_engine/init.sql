-- 1. api_keys 캐비닛(테이블)을 만듭니다.
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    client_key VARCHAR(100) UNIQUE NOT NULL
);

-- 2. 클로드가 찾고 있는 'ck_test'라는 서류를 미리 넣어둡니다.
INSERT INTO api_keys (client_key) VALUES ('ck_test') ON CONFLICT DO NOTHING;