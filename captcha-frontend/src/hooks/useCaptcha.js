import { useCallback, useEffect, useRef, useState } from 'react';
import { issueChallenge, submitAnswer } from '../api/captchaApi';

// =============================================================================
// 캡챠 상태 머신 훅
// status: idle → loading → active → (success | fail)
// active 동안 spec.time_limit_sec 카운트다운, 0 이 되면 자동 fail.
//
// submit(payload) 는 payload 를 그대로 백엔드로 전달하되,
// behavioral_data.time_taken_ms 만 자체 측정값으로 보강한다.
// 캡챠 종류별 payload 형태:
//   flashlight  → { click_x, click_y, behavioral_data }
//   face_mission → { completed_instructions, face_behavioral_data }
// =============================================================================

export function useCaptcha({ kind = 'flashlight', difficulty = null } = {}) {
  const [status, setStatus] = useState('idle'); // idle|loading|active|success|fail
  const [spec, setSpec] = useState(null);
  const [token, setToken] = useState(null);
  const [error, setError] = useState(null);
  const [remainingSec, setRemainingSec] = useState(0);

  // active 시점 (ms). time_taken_ms 계산용.
  const startedAtRef = useRef(null);

  const start = useCallback(async () => {
    setStatus('loading');
    setError(null);
    setSpec(null);
    setToken(null);

    const res = await issueChallenge(kind, difficulty);
    if (!res.ok) {
      setError(res.error);
      setStatus('fail');
      return;
    }
    const s = res.data;
    setSpec(s);
    setRemainingSec(s.time_limit_sec);
    startedAtRef.current = Date.now();
    setStatus('active');
  }, [kind, difficulty]);

  const reset = useCallback(() => {
    setStatus('idle');
    setSpec(null);
    setToken(null);
    setError(null);
    setRemainingSec(0);
    startedAtRef.current = null;
  }, []);

  const submit = useCallback(
    async (payload) => {
      if (!spec || status !== 'active') return;

      const timeTakenMs = startedAtRef.current
        ? Date.now() - startedAtRef.current
        : null;

      // payload 의 behavioral_data 가 없거나 time_taken_ms 가 비어있으면 보강.
      const enriched = {
        ...(payload ?? {}),
        behavioral_data: {
          ...(payload?.behavioral_data ?? {}),
          time_taken_ms: payload?.behavioral_data?.time_taken_ms ?? timeTakenMs,
        },
      };

      const res = await submitAnswer(spec.challenge_id, enriched);
      if (res.ok) {
        // 손전등 캡챠는 200 OK 본문에 decision='block' 로 차단을 전달한다.
        // decision 필드가 없는 다른 캡챠는 기존 success 분기로 진입.
        if (res.data?.decision === 'block') {
          setError({ code: 'verification_failed', message: '인증 실패' });
          setStatus('fail');
        } else {
          setToken(res.data.captcha_token);
          setStatus('success');
        }
      } else {
        setError(res.error);
        setStatus('fail');
      }
    },
    [spec, status],
  );

  // 타이머: 1초마다 카운트다운, 0 도달 시 자동 fail.
  useEffect(() => {
    if (status !== 'active' || !spec) return;
    const tick = setInterval(() => {
      setRemainingSec((prev) => {
        if (prev <= 1) {
          clearInterval(tick);
          setError({ code: 'timeout', message: '시간 초과' });
          setStatus('fail');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(tick);
  }, [status, spec]);

  return {
    status,
    spec,
    token,
    error,
    remainingSec,
    start,
    submit,
    reset,
  };
}
