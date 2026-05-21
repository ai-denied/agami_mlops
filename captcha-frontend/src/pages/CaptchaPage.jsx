import { useState } from 'react';
import { useCaptcha } from '../hooks/useCaptcha';
import CaptchaRouter from '../components/CaptchaRouter';

// =============================================================================
// /captcha 라우트 페이지
// status 별 화면을 분기:
//   idle    → "캡챠 시작" 카드
//   loading → 스피너 카드
//   active  → CaptchaRouter (kind='flashlight')
//   success → 성공 + captcha_token 표시
//   fail    → 실패 + 재시도
// =============================================================================

const KIND_LABEL = {
  flashlight: '🔦 손전등',
  face_mission: '😶 안면 미션',
  context_inference: '🧠 맥락 추론',
};

export default function CaptchaPage() {
  const [kind, setKind] = useState('flashlight');
  const [difficulty, setDifficulty] = useState('easy');

  const {
    status,
    spec,
    token,
    error,
    start,
    submit,
    reset,
  } = useCaptcha({ kind, difficulty });

  // 위젯이 보내주는 payload 를 hook 으로 그대로 통과.
  // 캡챠 종류별 payload 형태:
  //   flashlight  → { click_x, click_y, behavioral_data }
  //   face_mission → { completed_instructions, face_behavioral_data }
  // hook 이 behavioral_data.time_taken_ms 만 자체 측정값으로 보강한다.
  const handleSubmit = (payload) => {
    submit(payload);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#f5f8ff] to-[#e8f0ff] px-4 py-12">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8 text-center">
          <div className="mb-1.5 text-xs font-semibold uppercase tracking-[0.3em] text-[#4a8bff]">
            agami captcha
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-[#1d2a44]">
            행동 기반 캡챠 데모
          </h1>
          <p className="mt-2 text-sm text-[#6b7891]">
            FastAPI ↔ React 위젯 E2E 테스트
          </p>
        </header>

        {/* 컨트롤 카드 */}
        {status === 'idle' && (
          <div className="mx-auto w-full max-w-[640px] rounded-3xl bg-white p-8 shadow-[0_20px_60px_rgba(70,130,255,0.15)]">
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
              <div>
                <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-[#8a96ad]">
                  캡챠 종류
                </label>
                <select
                  value={kind}
                  onChange={(e) => setKind(e.target.value)}
                  className="w-full rounded-xl border-[1.5px] border-[#e0e7f3] bg-white px-4 py-2.5 text-sm font-medium text-[#1d2a44] focus:border-[#4a8bff] focus:outline-none"
                >
                  {Object.entries(KIND_LABEL).map(([k, label]) => (
                    <option key={k} value={k}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-[#8a96ad]">
                  난이도
                </label>
                <select
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                  className="w-full rounded-xl border-[1.5px] border-[#e0e7f3] bg-white px-4 py-2.5 text-sm font-medium text-[#1d2a44] focus:border-[#4a8bff] focus:outline-none"
                >
                  <option value="easy">easy</option>
                  <option value="medium">medium</option>
                  <option value="hard">hard</option>
                </select>
              </div>
            </div>
            <button
              onClick={start}
              className="mt-6 w-full rounded-xl bg-gradient-to-r from-[#4a8bff] to-[#6da5ff] px-6 py-3 text-sm font-bold text-white shadow-[0_8px_24px_rgba(74,139,255,0.35)] transition-transform hover:-translate-y-0.5"
            >
              캡챠 시작
            </button>
          </div>
        )}

        {status === 'loading' && (
          <div className="mx-auto flex h-48 w-full max-w-[640px] items-center justify-center rounded-3xl bg-white shadow-[0_20px_60px_rgba(70,130,255,0.15)]">
            <div className="flex items-center gap-3 text-[#6b7891]">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-[#e0e7f3] border-t-[#4a8bff]" />
              <span className="text-sm font-medium">챌린지를 발급받는 중…</span>
            </div>
          </div>
        )}

        {status === 'active' && spec && (
          <CaptchaRouter
            kind={kind}
            spec={spec}
            status={status}
            error={error}
            onSubmit={handleSubmit}
            onRefresh={start}
          />
        )}

        {status === 'success' && (
          <div className="mx-auto w-full max-w-[640px] rounded-3xl bg-white p-8 shadow-[0_20px_60px_rgba(70,130,255,0.15)]">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-100 text-2xl">
                ✅
              </div>
              <div>
                <div className="text-lg font-bold text-[#1d2a44]">검증 성공</div>
                <div className="text-xs text-[#6b7891]">
                  captcha_token 을 기업 백엔드의 /v1/siteverify 로 전달하세요.
                </div>
              </div>
            </div>
            <pre className="mt-5 overflow-x-auto rounded-xl bg-[#f5f8ff] p-4 text-xs text-[#2563eb]">
              {token}
            </pre>
            <button
              onClick={reset}
              className="mt-5 rounded-xl border-[1.5px] border-[#e0e7f3] bg-white px-4 py-2 text-sm font-semibold text-[#6b7891] hover:border-[#c8dcff] hover:text-[#4a8bff]"
            >
              처음으로
            </button>
          </div>
        )}

        {status === 'fail' && (
          <div className="mx-auto w-full max-w-[640px] rounded-3xl bg-white p-8 shadow-[0_20px_60px_rgba(70,130,255,0.15)]">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-rose-100 text-2xl">
                ❌
              </div>
              <div>
                <div className="text-lg font-bold text-[#1d2a44]">검증 실패</div>
                <div className="text-xs text-[#6b7891]">
                  {error?.message || '알 수 없는 오류'}{' '}
                  {error?.code ? <span className="text-rose-500">({error.code})</span> : null}
                </div>
              </div>
            </div>
            <div className="mt-5 flex gap-2">
              <button
                onClick={start}
                className="rounded-xl bg-gradient-to-r from-[#4a8bff] to-[#6da5ff] px-4 py-2 text-sm font-bold text-white shadow-[0_8px_24px_rgba(74,139,255,0.35)] hover:-translate-y-0.5 transition-transform"
              >
                다시 시도
              </button>
              <button
                onClick={reset}
                className="rounded-xl border-[1.5px] border-[#e0e7f3] bg-white px-4 py-2 text-sm font-semibold text-[#6b7891] hover:border-[#c8dcff] hover:text-[#4a8bff]"
              >
                처음으로
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
