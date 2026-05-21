import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';

// =============================================================================
// FishTimer
// -----------------------------------------------------------------------------
// 왼쪽 봇 ─ 출렁이는 sine wave 트랙 ─ 곡선 위에 올라타 헤엄치는 물고기.
// 물고기는 곡선 기울기에 따라 머리 각도가 부드럽게 회전.
// 시간이 줄면 트랙이 물고기 입 쪽으로 짧아지고, 물고기는 봇 쪽으로 다가온다.
// 봇 도달 시 chomp 진동 → 봇이 시계반대 2바퀴 회전+축소+페이드아웃 → 물고기 1.6배.
//
// 60fps 부드러운 모션: prop 의 remainingMs 가 1Hz 로만 갱신되지만, 컴포넌트 내부에서
// performance.now() 로 보간하여 매 rAF 프레임에 미세하게 갱신.
//
// Props
//   remainingMs : 남은 시간 (ms)
//   totalMs     : 전체 시간 (ms) — 변경 시 새 챌린지로 간주, 'swimming' 리셋.
//   className   : 컨테이너 추가 클래스 (마진 등)
// =============================================================================

const BOT_WIDTH_PX = 50;
const FISH_SIZE_PX = 56;
const FISH_MOUTH_OFFSET_PX = 14;        // 이미지 좌측 ~ 물고기 입까지 (px)
const CURVE_AMPLITUDE_PX = 10;          // sin 진폭
const CURVE_PERIOD_PX = 120;            // sin 주기
const CONTAINER_HEIGHT_PX = 64;         // h-16
const MID_Y = CONTAINER_HEIGHT_PX / 2;
const FISH_SCALE_FED = 1.6;             // 먹기 후 물고기 크기
const SCALE_TRANSITION_MS = 2100;       // unmount 전에 시각적으로 fully grown 도달
const ROTATE_TRANSITION_MS = 200;       // 회전 lag (잔잔한 smoothing)
const CHOMP_DURATION_MS = 200;          // CSS keyframe 1회 길이 (× 2 iterations = 400ms)
const CHOMP_ITERATIONS = 2;
const EATING_DURATION_MS = 2500;
const BOT_DISAPPEAR_MS = 1500;
const EATING_THRESHOLD_FALLBACK = 0.15;
const PATH_STEP_PX = 4;                 // SVG line segment 간격

// 곡선 수식 — fish 위치 / 회전 / SVG path 가 모두 같은 함수에서 파생되어 정확히 일치.
function curveY(x) {
  return MID_Y + CURVE_AMPLITUDE_PX
    * Math.sin(2 * Math.PI * (x - BOT_WIDTH_PX) / CURVE_PERIOD_PX);
}

function curveSlope(x) {
  return CURVE_AMPLITUDE_PX * (2 * Math.PI / CURVE_PERIOD_PX)
    * Math.cos(2 * Math.PI * (x - BOT_WIDTH_PX) / CURVE_PERIOD_PX);
}

// 곡선 path 를 line segment 들로 근사. step=4px 면 시각적으로 충분히 부드러움.
function buildCurvePath(startX, endX, step) {
  if (endX <= startX) return '';
  let d = `M ${startX.toFixed(2)} ${curveY(startX).toFixed(2)}`;
  for (let x = startX + step; x < endX; x += step) {
    d += ` L ${x.toFixed(2)} ${curveY(x).toFixed(2)}`;
  }
  d += ` L ${endX.toFixed(2)} ${curveY(endX).toFixed(2)}`;
  return d;
}

export default function FishTimer({ remainingMs, totalMs, className = '' }) {
  const uid = useId().replace(/:/g, '');

  const [phase, setPhase] = useState('swimming'); // 'swimming' | 'eating' | 'fed'
  const [reduceMotion, setReduceMotion] = useState(false);
  const [containerWidth, setContainerWidth] = useState(0);
  const [, setTick] = useState(0);                 // rAF 가 강제 리렌더용으로만 사용

  const containerRef = useRef(null);
  // 1Hz prop 사이를 60fps 로 보간하기 위한 기준점.
  const baseRemainingMsRef = useRef(remainingMs);
  const baseTsRef = useRef(performance.now());
  const rafRef = useRef(null);

  // 1) prefers-reduced-motion
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduceMotion(mq.matches);
    const handler = (e) => setReduceMotion(e.matches);
    mq.addEventListener?.('change', handler);
    return () => mq.removeEventListener?.('change', handler);
  }, []);

  // 2) 컨테이너 width — 첫 paint 전 측정 + ResizeObserver 로 추적
  useLayoutEffect(() => {
    if (containerRef.current) setContainerWidth(containerRef.current.offsetWidth);
  }, []);
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      setContainerWidth(entries[0].contentRect.width);
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // 3) prop 의 remainingMs 가 갱신될 때마다 보간 기준점 새로고침
  useEffect(() => {
    baseRemainingMsRef.current = remainingMs;
    baseTsRef.current = performance.now();
  }, [remainingMs]);

  // 4) totalMs 변경 → 새 챌린지, phase 리셋
  useEffect(() => {
    setPhase('swimming');
  }, [totalMs]);

  // 5) rAF 루프 — swimming + !reduceMotion 동안만 60fps 강제 리렌더
  useEffect(() => {
    if (reduceMotion || phase !== 'swimming') return;
    const loop = () => {
      setTick((t) => (t + 1) % 1e9);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [phase, reduceMotion]);

  // ---------------------------------------------------------------------
  // 보간된 effective remaining ms 계산 (매 렌더마다 performance.now() 읽음)
  // ---------------------------------------------------------------------
  const now = performance.now();
  const elapsedSinceBase = now - baseTsRef.current;
  const effectiveRemainingMs = reduceMotion
    ? remainingMs
    : Math.max(0, baseRemainingMsRef.current - elapsedSinceBase);

  const progress = totalMs > 0
    ? Math.max(0, Math.min(1, effectiveRemainingMs / totalMs))
    : 0;

  const eatingThreshold = totalMs > 0
    ? Math.min(EATING_DURATION_MS / totalMs, 0.5)
    : EATING_THRESHOLD_FALLBACK;

  const visualProgress = progress > eatingThreshold
    ? (progress - eatingThreshold) / (1 - eatingThreshold)
    : 0;

  // 6) phase 전환
  useEffect(() => {
    if (phase !== 'swimming') return;
    if (progress <= eatingThreshold) {
      setPhase('eating');
      const t = setTimeout(
        () => setPhase('fed'),
        CHOMP_DURATION_MS * CHOMP_ITERATIONS,
      );
      return () => clearTimeout(t);
    }
  }, [progress, phase, eatingThreshold]);

  // ---------------------------------------------------------------------
  // 렌더용 파생값
  // ---------------------------------------------------------------------
  const fishScale = phase === 'swimming' ? 1.0 : FISH_SCALE_FED;

  const trackSpan = Math.max(1, containerWidth - BOT_WIDTH_PX);
  const fishLeft = BOT_WIDTH_PX + trackSpan * visualProgress;

  // 곡선 라이딩은 swimming + !reduceMotion 일 때만. eating/fed/reduceMotion 은 직선상.
  const onCurve = phase === 'swimming' && !reduceMotion;
  const fishTop = onCurve ? curveY(fishLeft) : MID_Y;

  // 회전각 — 곡선 접선 (사용자 관점: 곡선이 오를 때 머리 위로, 내릴 때 머리 아래로)
  // 부호 컨벤션 (사용자 시각 테스트로 확정):
  //   fish 가 진행하면서 위로 올라가는 중 → 머리 위 (negative CSS rotate = CCW)
  //   fish 가 진행하면서 아래로 내려가는 중 → 머리 아래 (positive CSS rotate = CW)
  // fish 는 우→좌 이동 → 좌측 방향의 곡선 기울기가 head 방향. dy/d(-x) = -slope.
  // CSS rotate(positive) 시 fish 머리가 진행 방향(좌측 + 아래) 으로 기울게 하려면
  // angleDeg = +atan(slope) (이전 -atan 의 부호 반전).
  const slopeAtFish = onCurve ? curveSlope(fishLeft) : 0;
  const angleDeg = Math.atan(slopeAtFish) * 180 / Math.PI;

  // 트랙 끝 = fish 입 위치
  const mouthFromCenterPx = FISH_SIZE_PX / 2 - FISH_MOUTH_OFFSET_PX;
  const trackEndX = Math.max(BOT_WIDTH_PX, fishLeft - mouthFromCenterPx);

  // 동적 path — BOT 에서 fish 의 입까지만. 나머지는 그리지 않음.
  const pathD = containerWidth > BOT_WIDTH_PX && trackEndX > BOT_WIDTH_PX
    ? buildCurvePath(BOT_WIDTH_PX, trackEndX, PATH_STEP_PX)
    : '';

  // Bot transforms
  const botTransform = phase === 'swimming'
    ? 'translateY(-50%)'
    : 'translateY(-50%) rotate(-720deg) scale(0.3)';
  const botOpacity = phase === 'swimming' ? 1 : 0;
  const botTransition = reduceMotion
    ? 'none'
    : `transform ${BOT_DISAPPEAR_MS}ms cubic-bezier(.4,0,.2,1), opacity ${BOT_DISAPPEAR_MS}ms ease-in`;

  // Fish OUTER transitions — 개별 transform 속성 별 독립 transition.
  // - rotate: 200ms linear (잔잔한 smoothing)
  // - scale:  2100ms ease-out (점진 grow, unmount 전 도달)
  // - translate(중앙정렬용)/left/top: transition 없음 (rAF 가 매 프레임 갱신)
  const fishOuterTransition = reduceMotion
    ? 'none'
    : `rotate ${ROTATE_TRANSITION_MS}ms linear, scale ${SCALE_TRANSITION_MS}ms ease-out`;

  return (
    <div
      ref={containerRef}
      className={`relative h-16 overflow-visible ${className}`}
      role="progressbar"
      aria-label="남은 시간"
      aria-valuemin={0}
      aria-valuemax={totalMs}
      aria-valuenow={Math.max(0, effectiveRemainingMs)}
    >
      {/* Bot — 왼쪽 끝 고정, eating 시 회전+축소+페이드 */}
      <div
        className="absolute left-0 w-9 h-9 rounded-xl bg-gradient-to-br from-[#4a8bff] to-[#7aa9ff] flex items-center justify-center shadow-sm"
        style={{
          top: '50%',
          transform: botTransform,
          transformOrigin: 'center',
          opacity: botOpacity,
          transition: botTransition,
        }}
        aria-hidden
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-white"
        >
          <path d="M12 8V4H8" />
          <rect width="16" height="12" x="4" y="8" rx="2" />
          <path d="M2 14h2" />
          <path d="M20 14h2" />
          <path d="M15 13v2" />
          <path d="M9 13v2" />
        </svg>
      </div>

      {/* SVG 곡선 트랙 — BOT 에서 fish 입까지 line segments 로 정확한 sine wave */}
      <svg
        className="absolute inset-0 w-full h-full overflow-visible pointer-events-none"
        aria-hidden
      >
        <defs>
          {/* userSpaceOnUse 로 path 가 짧아져도 색상 위치 일관 */}
          <linearGradient
            id={`trk-${uid}`}
            gradientUnits="userSpaceOnUse"
            x1={BOT_WIDTH_PX}
            x2={Math.max(BOT_WIDTH_PX + 1, containerWidth)}
            y1={MID_Y}
            y2={MID_Y}
          >
            <stop offset="0%" stopColor="#4a8bff" />
            <stop offset="100%" stopColor="#7aa9ff" />
          </linearGradient>
        </defs>
        {pathD && (
          <path
            d={pathD}
            stroke={`url(#trk-${uid})`}
            strokeWidth="6"
            fill="none"
            strokeLinecap="round"
          />
        )}
      </svg>

      {/* 물고기 — 2-level
          OUTER: 개별 transform 속성 (translate/rotate/scale) + 독립 transition
          INNER: chomp keyframe class (CSS animation, .fish-chomp) */}
      {containerWidth > 0 && (
        <div
          style={{
            position: 'absolute',
            left: `${fishLeft}px`,
            top: `${fishTop}px`,
            width: FISH_SIZE_PX,
            height: FISH_SIZE_PX,
            // 개별 transform 속성 — 각각 별도 transition 가능 (CSS Transforms Level 2)
            translate: '-50% -50%',
            rotate: `${angleDeg}deg`,
            scale: String(fishScale),
            transformOrigin: 'center',
            transition: fishOuterTransition,
            pointerEvents: 'none',
          }}
        >
          <div
            className={
              phase === 'eating' && !reduceMotion ? 'fish-chomp' : ''
            }
            style={{
              width: '100%',
              height: '100%',
              transformOrigin: 'center',
            }}
          >
            <img
              src="/timer-fish.png"
              alt=""
              aria-hidden
              width={FISH_SIZE_PX}
              height={FISH_SIZE_PX}
              draggable={false}
              className="block w-full h-full select-none"
              style={{ objectFit: 'contain' }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
