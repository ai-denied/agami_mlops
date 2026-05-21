import { useCallback, useRef } from 'react';

// =============================================================================
// 마우스 궤적 수집 훅
// - 학습 데이터(`app/ml/flashlight/sample_raw_mouse_log.json`)와 동일한 포맷으로
//   샘플 저장: x/y 는 컨테이너 좌상단 기준 픽셀 정수, t 는 첫 sample 시점부터의
//   elapsed ms 정수. dt = t[i] - t[i-1] 가 학습 시 velocity/acceleration 계산
//   분포와 일치하도록 보장한다.
// - 50ms throttle: 브라우저가 mousemove 를 ~16ms 간격으로 발화하므로 그대로
//   푸시하면 학습 분포(dt_mean≈72) 와 어긋나 false-positive 가 폭증한다.
// - 시그니처 (sample/get/reset) 은 보존 — 호출부 (FlashlightCaptcha.jsx) 무수정.
// - 클릭 핸들러도 sample 을 부르는데, 직전 mousemove 가 50ms 안에 있었으면
//   클릭 좌표는 trajectory 에 들어가지 않을 수 있다 (좌표 1개 손실).
//   학습 분포 정합이 우선이라 의도된 트레이드오프.
// =============================================================================

const THROTTLE_MS = 50;

export default function useMouseTracker() {
  const trajectoryRef = useRef([]);
  // 첫 sample 시점의 performance.now(). 새 sub 시작 시 reset() 으로 null 로
  // 돌려놓아야 다음 sample 이 t=0 부터 다시 시작.
  const startTimeRef = useRef(null);
  // 마지막 채택된 sample 의 startTime 기준 elapsed ms.
  // -Infinity 로 두면 첫 sample (elapsed=0) 이 throttle 검사를 무조건 통과한다.
  const lastSampleElapsedRef = useRef(-Infinity);

  const reset = useCallback(() => {
    trajectoryRef.current = [];
    startTimeRef.current = null;
    lastSampleElapsedRef.current = -Infinity;
  }, []);

  const sample = useCallback((event, rect) => {
    const now = performance.now();
    if (startTimeRef.current === null) {
      startTimeRef.current = now;
    }
    const elapsed = now - startTimeRef.current;
    if (elapsed - lastSampleElapsedRef.current < THROTTLE_MS) {
      return;
    }
    const x = Math.round(event.clientX - rect.left);
    const y = Math.round(event.clientY - rect.top);
    trajectoryRef.current.push({ x, y, t: Math.round(elapsed) });
    lastSampleElapsedRef.current = elapsed;
  }, []);

  const get = useCallback(() => trajectoryRef.current.slice(), []);

  return { sample, get, reset };
}
