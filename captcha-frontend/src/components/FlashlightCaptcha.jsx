import { useEffect, useRef, useState } from 'react';
import FishTimer from './FishTimer';
import useMouseTracker from '../hooks/useMouseTracker';
import { API_BASE_URL } from '../api/captchaApi';

// =============================================================================
// 손전등 캡챠 (1챌린지 = 3장 묶음, 실제 이미지 데이터셋 기반)
// - 배경: <img> 로 spec.sub_challenges[i].image_url 표시 (800x600, 4:3)
// - 손전등 효과: 이미지 위에 radial-gradient overlay div. rAF 로 DOM 직접 갱신
//   하여 React 리렌더 우회 (기존 캔버스 패턴과 동일한 성능 특성).
// - 클릭 시 정규화 좌표(0~1)로 onSubmit. 백엔드 verifier 가 bbox 매칭.
// =============================================================================
export default function FlashlightCaptcha({ spec, onSubmit, onRefresh, status, error }) {
  const wrapRef = useRef(null);
  const overlayRef = useRef(null);
  const cursorRingRef = useRef(null);
  const mouseRef = useRef({ x: 0.5, y: 0.5 });
  const submissionsRef = useRef([]);
  const startedAtRef = useRef(Date.now());

  const [currentIndex, setCurrentIndex] = useState(0);
  const [timeLeft, setTimeLeft] = useState(spec?.time_limit_sec ?? 60);
  const [hintVisible, setHintVisible] = useState(false);

  const tracker = useMouseTracker();
  const currentSub = spec?.sub_challenges?.[currentIndex];

  // [spec] effect — 번들 라이프사이클 1회 초기화: 타이머 / 힌트 / submissions / index / tracker
  useEffect(() => {
    if (!spec) return;
    setTimeLeft(spec.time_limit_sec);
    setHintVisible(false);
    setCurrentIndex(0);
    submissionsRef.current = [];
    tracker.reset();
    startedAtRef.current = Date.now();

    const tick = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) { clearInterval(tick); return 0; }
        return t - 1;
      });
    }, 1000);

    let hintTimer;
    if (spec.hint_after_sec) {
      hintTimer = setTimeout(() => setHintVisible(true), spec.hint_after_sec * 1000);
    }
    return () => { clearInterval(tick); if (hintTimer) clearTimeout(hintTimer); };
  }, [spec]);

  // [currentIndex] effect — sub 전환 시 trajectory 만 클리어. 타이머는 유지.
  useEffect(() => {
    tracker.reset();
  }, [currentIndex]);

  // [spec] effect — 손전등 overlay 의 radial-gradient 를 rAF 로 매 프레임 갱신.
  // wrap 크기 측정 + ResizeObserver 로 반응형 대응.
  useEffect(() => {
    if (!spec) return;
    const wrap = wrapRef.current;
    const overlay = overlayRef.current;
    const ring = cursorRingRef.current;
    if (!wrap || !overlay) return;

    let cssW = 0;
    let cssH = 0;
    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      cssW = rect.width;
      cssH = rect.height;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    let raf = 0;
    const draw = () => {
      const { x, y } = mouseRef.current;
      const mx = x * cssW;
      const my = y * cssH;
      const radius = spec.flashlight_radius * Math.min(cssW, cssH);

      // radial-gradient 으로 마우스 위치에 손전등 구멍, 그 외엔 어두운 마스크.
      overlay.style.background =
        `radial-gradient(circle ${radius}px at ${mx}px ${my}px, ` +
        `rgba(0,0,0,0) 0%, ` +
        `rgba(0,0,0,0) 70%, ` +
        `rgba(0,0,0,1) 100%)`;

      if (ring) {
        ring.style.transform = `translate(${mx}px, ${my}px) translate(-50%, -50%)`;
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [spec]);

  const handleMouseMove = (e) => {
    const rect = wrapRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    mouseRef.current = { x, y };
    tracker.sample(e, rect);
  };

  const handleClick = (e) => {
    const rect = wrapRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    tracker.sample(e, rect);

    submissionsRef.current = [
      ...submissionsRef.current,
      { index: currentIndex, click_x: x, click_y: y, trajectory: tracker.get() },
    ];

    if (currentIndex < 2) {
      setCurrentIndex(currentIndex + 1);
    } else {
      onSubmit({ flashlight_submissions: submissionsRef.current });
    }
  };

  if (!spec || !currentSub) return null;

  return (
    <div className="w-full max-w-[900px] min-w-0 bg-white rounded-xl shadow-[0_20px_60px_rgba(70,130,255,0.15)] overflow-hidden mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-[#4a8bff] to-[#6da5ff] text-white">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-white/20 rounded-lg flex items-center justify-center text-lg">
            🔦
          </div>
          <div>
            <div className="font-bold text-[15px] leading-tight">손전등 탐색 캡챠</div>
            <div className="text-xs opacity-85 mt-0.5">어둠 속에 숨겨진 물건을 3번 찾아주세요</div>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-white/20 px-4 py-1.5 rounded-full">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="12" cy="12" r="9" />
            <path d="M12 7v5l3 2" />
          </svg>
          <span className="font-bold text-sm tabular-nums">{timeLeft}s</span>
        </div>
      </div>

      {/* Body */}
      <div className="px-6 pt-5">
        {/* 진행도 바 */}
        <div className="flex gap-2 mb-2">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={`flex-1 h-1.5 rounded-full transition-colors ${
                i < currentIndex
                  ? 'bg-[#4a8bff]'
                  : i === currentIndex
                    ? 'bg-[#9ec3ff]'
                    : 'bg-[#e0e7f3]'
              }`}
            />
          ))}
        </div>
        <div className="text-xs text-[#8a96ad] mb-3">
          진행 <span className="font-bold text-[#2563eb]">{currentIndex + 1}</span> / 3
        </div>

        <div className="flex items-center justify-between mb-3.5">
          <div className="flex items-center gap-2.5">
            <span className="text-xs text-[#8a96ad] font-semibold uppercase tracking-wide">찾을 물건</span>
            <div className="inline-flex items-center gap-2 bg-[#eef4ff] border-[1.5px] border-[#c8dcff] px-3.5 py-1.5 rounded-full">
              <span className="font-bold text-[#2563eb] text-sm">{currentSub.target_hint.label}</span>
            </div>
          </div>
          <div className="text-xs text-[#8a96ad]">
            난이도 · <span className="text-[#4a8bff] font-bold uppercase">{spec.difficulty}</span>
          </div>
        </div>

        {/* Image canvas — 4:3 aspect, 손전등 overlay */}
        <div
          ref={wrapRef}
          onMouseMove={handleMouseMove}
          onClick={handleClick}
          className="relative w-full aspect-[4/3] bg-[#0a0a14] rounded-lg overflow-hidden border-2 border-[#1a1a28]"
          style={{ cursor: 'none' }}
        >
          <img
            key={currentSub.image_url}
            src={`${API_BASE_URL}${currentSub.image_url}`}
            alt=""
            className="absolute inset-0 w-full h-full object-cover pointer-events-none select-none"
            draggable={false}
          />
          <div
            ref={overlayRef}
            className="absolute inset-0 pointer-events-none"
          />

          <div
            ref={cursorRingRef}
            className="absolute top-0 left-0 w-5 h-5 border-2 border-[rgba(255,235,180,0.85)] rounded-full pointer-events-none will-change-transform"
            style={{
              mixBlendMode: 'screen',
              transform: 'translate(0px, 0px) translate(-50%, -50%)',
            }}
          />

          <div className="absolute top-3.5 right-3.5 bg-white/10 backdrop-blur px-3 py-1.5 rounded-full text-[11px] text-white/70 flex items-center gap-1.5 pointer-events-none">
            <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
            마우스로 손전등 조작
          </div>

          {hintVisible && (
            <div className="absolute bottom-3.5 left-1/2 -translate-x-1/2 bg-amber-400/90 text-amber-950 px-4 py-1.5 rounded-full text-xs font-bold pointer-events-none">
              💡 천천히 둘러보세요
            </div>
          )}
        </div>

        <FishTimer
          remainingMs={timeLeft * 1000}
          totalMs={spec.time_limit_sec * 1000}
          className="mt-3.5"
        />
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-6 py-5">
        <div className="flex items-center gap-2 text-[#8a96ad] text-xs">
          <span>🛡️</span>
          <span>agami로 보호되는 페이지</span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onRefresh}
            className="bg-transparent border-[1.5px] border-[#e0e7f3] text-[#6b7891] px-4 py-2 rounded-xl text-sm font-semibold hover:border-[#c8dcff] hover:text-[#4a8bff] transition-colors"
          >
            🔄 새로고침
          </button>
        </div>
      </div>
    </div>
  );
}
