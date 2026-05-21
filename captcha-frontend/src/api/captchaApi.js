// =============================================================================
// captcha_engine 백엔드 (FastAPI) 와 통신하는 얇은 래퍼.
// - 모든 함수는 throw 하지 않고 { ok, data?, error? } 형태로 반환.
// - VITE_API_URL 미지정 시 http://localhost:8000 으로 fallback.
// - 모든 요청에 X-Captcha-Client-Key 헤더가 필요 (백엔드 deps.verify_client_key).
// =============================================================================

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const BASE_URL = API_BASE_URL;
const CLIENT_KEY = import.meta.env.VITE_CAPTCHA_CLIENT_KEY || 'ck_test';

const COMMON_HEADERS = {
  'Content-Type': 'application/json',
  'X-Captcha-Client-Key': CLIENT_KEY,
};

async function parseError(res) {
  // 백엔드 error 포맷: { error: { code, message, request_id } }
  try {
    const body = await res.json();
    if (body?.error) return body.error;
    return { code: `http_${res.status}`, message: res.statusText || 'Request failed' };
  } catch {
    return { code: `http_${res.status}`, message: res.statusText || 'Request failed' };
  }
}

/**
 * 챌린지 발급.
 * @param {('flashlight'|'face_mission'|'context_inference')} kind
 * @param {('easy'|'medium'|'hard'|null)} difficulty
 * @returns {Promise<{ ok: true, data: object } | { ok: false, error: object }>}
 */
export async function issueChallenge(kind = 'flashlight', difficulty = null) {
  try {
    const res = await fetch(`${BASE_URL}/v1/challenges`, {
      method: 'POST',
      headers: COMMON_HEADERS,
      body: JSON.stringify({ kind, difficulty }),
    });
    if (!res.ok) return { ok: false, error: await parseError(res) };
    return { ok: true, data: await res.json() };
  } catch (err) {
    return { ok: false, error: { code: 'network_error', message: String(err) } };
  }
}

/**
 * 정답 제출. payload 는 캡챠 종류에 따라 달라짐:
 *   - flashlight: { click_x, click_y, behavioral_data }
 *   - face_mission: { completed_instructions, face_behavioral_data, behavioral_data }
 * 백엔드의 SubmitAnswerRequest 가 모든 필드를 Optional 로 받고 kind 별 분기 검증함.
 *
 * @param {string} challengeId
 * @param {object} payload
 * @returns {Promise<{ ok: true, data: { captcha_token, expires_in } } | { ok: false, error: object }>}
 */
export async function submitAnswer(challengeId, payload) {
  try {
    const res = await fetch(
      `${BASE_URL}/v1/challenges/${encodeURIComponent(challengeId)}/answer`,
      {
        method: 'POST',
        headers: COMMON_HEADERS,
        body: JSON.stringify(payload ?? {}),
      },
    );
    if (!res.ok) return { ok: false, error: await parseError(res) };
    return { ok: true, data: await res.json() };
  } catch (err) {
    return { ok: false, error: { code: 'network_error', message: String(err) } };
  }
}
