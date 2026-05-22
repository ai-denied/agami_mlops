import { useEffect, useRef, useState } from 'react';
import FishTimer from './FishTimer';

// =============================================================================
// 감정 맥락 추론 캡챠 위젯 (N문제 시퀀스 인터랙션)
// -----------------------------------------------------------------------------
// 한 챌린지가 spec.questions = [{index, image_url, choices}, ...] 형태로 도착.
// 사용자는 step 마다 1문제를 풀며, 마지막 문제에서 "제출하기" 클릭 시 누적 답안을
// onSubmit({ submitted_answers: [...] }) 으로 한 번에 백엔드로 전달.
//
// prop 시그니처는 기존과 동일 — FlashlightCaptcha / FaceMissionCaptcha 와 통일.
// (CaptchaRouter / ContextCaptchaPage 가 import 하는 이름 `ImageGridCaptcha` 유지)
// =============================================================================

const EMOTION_KO = {
  joy: '기쁨',
  sadness: '슬픔',
  anger: '분노',
  fear: '두려움',
  surprise: '놀람',
  disgust: '혐오',
  contempt: '경멸',
};

const EMOTION_ICON = {
  joy: '😊',
  sadness: '😢',
  anger: '😠',
  fear: '😨',
  surprise: '😲',
  disgust: '🤢',
  contempt: '😒',
};

export default function ImageGridCaptcha({ spec, onSubmit, onRefresh, status, error }) {
  // 카운트다운 (전체 시간 제한, 문제 이동 무관 유지)
  const [timeLeft, setTimeLeft] = useState(spec?.time_limit_sec ?? 30);

  // 진행 상태
  const [step, setStep] = useState(0);            // 현재 문제 인덱스 0..total-1
  const [answers, setAnswers] = useState([]);     // 확정된 이전 문제 답안
  const [selected, setSelected] = useState(null); // 현재 문제 선택값 (null = 미선택)
  const [submitting, setSubmitting] = useState(false);

  // 현재 step 이미지 로딩 추적 (step 변경마다 리셋)
  const [imgLoaded, setImgLoaded] = useState(false);

  const startedAtRef = useRef(Date.now());

  // spec 변경 (재시도/새 챌린지) 시 모든 상태 초기화
  useEffect(() => {
    if (!spec) return;
    setTimeLeft(spec.time_limit_sec);
    setStep(0);
    setAnswers([]);
    setSelected(null);
    setSubmitting(false);
    setImgLoaded(false);
    startedAtRef.current = Date.now();
  }, [spec]);

  // 문제가 바뀔 때마다 이미지 로딩 상태 리셋
  useEffect(() => {
    setImgLoaded(false);
  }, [step]);

  // 1Hz 카운트다운 — 문제 이동과 무관하게 전체 시간 유지
  useEffect(() => {
    if (!spec) return;
    const tick = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) { clearInterval(tick); return 0; }
        return t - 1;
      });
    }, 1000);
    return () => clearInterval(tick);
  }, [spec]);

  if (!spec) return null;

  const total = spec.total_count ?? (spec.questions?.length ?? 0);
  const currentQ = spec.questions?.[step];
  const isLastStep = step >= total - 1;
  const canAdvance = selected != null && !submitting;

  const handlePick = (emotion) => {
    if (submitting) return;
    setSelected(emotion);
  };

  const handleNext = () => {
    if (!canAdvance) return;
    if (isLastStep) {
      // 마지막 문제 → 누적 답안 + 현재 선택을 합쳐서 백엔드로 한 번에 제출
      setSubmitting(true);
      onSubmit({
        submitted_answers: [...answers, selected],
        behavioral_data: {
          time_taken_ms: Date.now() - startedAtRef.current,
        },
      });
    } else {
      setAnswers((prev) => [...prev, selected]);
      setStep((s) => s + 1);
      setSelected(null);
    }
  };

  // 문제 단계 진행률 (전체 시간 진행률은 FishTimer 가 자체 계산)
  const stepPct = ((step + 1) / total) * 100;

  return (
    <div className="w-full max-w-[480px] min-w-0 bg-white rounded-xl shadow-[0_20px_60px_rgba(70,130,255,0.15)] overflow-hidden mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-[#4a8bff] to-[#6da5ff] text-white">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-white/20 rounded-lg flex items-center justify-center text-lg">
            🧠
          </div>
          <div>
            <div className="font-bold text-[15px] leading-tight">감정 맥락 추론 캡챠</div>
            <div className="text-xs opacity-85 mt-0.5">사진을 보고 감정을 골라주세요</div>
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
        {/* 단계 표시 */}
        <div className="flex items-center justify-between mb-2.5">
          <div className="text-xs text-[#8a96ad] font-semibold uppercase tracking-wide">
            진행 상태
          </div>
          <div className="text-sm font-bold text-[#1d2a44] tabular-nums">
            문제 {step + 1}<span className="text-[#8a96ad]"> / {total}</span>
          </div>
        </div>
        <div className="mb-4 w-full h-1.5 bg-[#f0f4fb] rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-[#4a8bff] to-[#7aa9ff] rounded-full transition-all duration-300"
            style={{ width: `${stepPct}%` }}
          />
        </div>

        {/* 현재 문제 이미지 — 컨테이너 자체에는 사이즈 강제 없이 flex 가운데 정렬.
            이미지가 max-h-[40vh] 로 viewport 높이 40% 까지만 차지하고 비율(object-contain)
            을 유지해서, 4지선다 / FishTimer / 푸터 가 같은 화면에 함께 보이도록 한다.
            spinner 는 이미지 위에 absolute 로 겹쳐 그린다. */}
        <div className="relative w-full bg-[#f0f4fb] rounded-xl overflow-hidden border-2 border-[#e0e7f3] flex items-center justify-center" style={{ minHeight: '160px' }}>
          <img
            key={step}
            src={currentQ?.image_url}
            alt={`감정 추론 문제 ${step + 1}`}
            onLoad={() => setImgLoaded(true)}
            onError={() => setImgLoaded(true)}
            className={
              'block max-h-[40vh] w-auto max-w-full object-contain transition-opacity duration-300 ' +
              (imgLoaded ? 'opacity-100' : 'opacity-0')
            }
            draggable={false}
          />
          {!imgLoaded && (
            <div className="absolute inset-0 flex items-center justify-center text-[#8a96ad]">
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-[#e0e7f3] border-t-[#4a8bff]" />
            </div>
          )}
        </div>

        {/* 질문 */}
        <div className="mt-4 mb-3">
          <span className="text-xs text-[#8a96ad] font-semibold uppercase tracking-wide mr-2">Q.</span>
          <span className="text-[#1d2a44] font-bold text-base">
            이 사진에서 느껴지는 감정은?
          </span>
        </div>

        {/* 2×2 선택지 */}
        <div className="grid grid-cols-2 gap-2.5">
          {(currentQ?.choices ?? []).map((emotion) => {
            const isPicked = selected === emotion;
            const isOtherPicked = selected != null && !isPicked;
            return (
              <button
                key={emotion}
                type="button"
                disabled={submitting}
                onClick={() => handlePick(emotion)}
                className={
                  'flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-bold transition border-[1.5px] ' +
                  (isPicked
                    ? 'bg-[#4a8bff] border-[#4a8bff] text-white shadow-[0_8px_24px_rgba(74,139,255,0.35)]'
                    : isOtherPicked
                    ? 'bg-[#eef4ff] border-[#c8dcff] text-[#2563eb] opacity-50 cursor-pointer'
                    : 'bg-[#eef4ff] border-[#c8dcff] text-[#2563eb] hover:bg-[#dceaff] hover:border-[#4a8bff] cursor-pointer')
                }
              >
                <span className="text-base leading-none">{EMOTION_ICON[emotion] ?? '🎯'}</span>
                <span>{EMOTION_KO[emotion] ?? emotion}</span>
              </button>
            );
          })}
        </div>

        {/* 다음 / 제출 버튼 */}
        <button
          type="button"
          onClick={handleNext}
          disabled={!canAdvance}
          className={
            'mt-4 w-full px-4 py-3 rounded-xl text-sm font-bold transition-transform ' +
            (canAdvance
              ? 'bg-gradient-to-r from-[#4a8bff] to-[#6da5ff] text-white shadow-[0_8px_24px_rgba(74,139,255,0.35)] hover:-translate-y-0.5'
              : 'bg-[#eef4ff] text-[#8a96ad] cursor-not-allowed')
          }
        >
          {submitting
            ? '제출 중…'
            : isLastStep
            ? '제출하기'
            : '다음 →'}
        </button>

        {/* 전체 남은 시간 — 물고기 한 마리가 우측에서 좌측으로 헤엄친다 */}
        <FishTimer
          remainingMs={timeLeft * 1000}
          totalMs={spec.time_limit_sec * 1000}
          className="mt-4"
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
