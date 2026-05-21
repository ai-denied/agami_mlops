import { useCaptcha } from '../hooks/useCaptcha';
import ImageGridCaptcha from '../components/ImageGridCaptcha';

// =============================================================================
// /captcha/context 전용 페이지 — 감정 맥락 추론 캡챠 위젯의 미니멀 임베드 페이지.
// CaptchaPage 와 달리 kind/difficulty 셀렉터 없음. 단일 흐름.
// =============================================================================

export default function ContextCaptchaPage() {
  const {
    status,
    spec,
    token,
    error,
    start,
    submit,
    reset,
  } = useCaptcha({ kind: 'context_inference', difficulty: 'easy' });

  // 위젯이 보낸 payload (selected_emotion, behavioral_data) 를 hook 으로 그대로 전달
  const handleSubmit = (payload) => submit(payload);

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#f5f8ff] to-[#e8f0ff] px-4 py-12">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8 text-center">
          <div className="mb-1.5 text-xs font-semibold uppercase tracking-[0.3em] text-[#4a8bff]">
            agami captcha
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-[#1d2a44]">
            감정 맥락 추론 캡챠
          </h1>
          <p className="mt-2 text-sm text-[#6b7891]">
            사진 속 인물이 느낄 감정을 골라주세요
          </p>
        </header>

        {/* idle */}
        {status === 'idle' && (
          <div className="mx-auto w-full max-w-[480px] rounded-xl bg-white p-8 shadow-[0_20px_60px_rgba(70,130,255,0.15)] text-center">
            <div className="text-5xl mb-3">🧠</div>
            <div className="text-lg font-bold text-[#1d2a44] mb-2">
              4지선다 감정 추론
            </div>
            <p className="text-sm text-[#6b7891] mb-6">
              화면에 표시되는 이미지를 보고 가장 어울리는 감정을 골라주세요.
            </p>
            <button
              onClick={start}
              className="w-full rounded-xl bg-gradient-to-r from-[#4a8bff] to-[#6da5ff] px-6 py-3 text-sm font-bold text-white shadow-[0_8px_24px_rgba(74,139,255,0.35)] transition-transform hover:-translate-y-0.5"
            >
              캡챠 시작
            </button>
          </div>
        )}

        {/* loading */}
        {status === 'loading' && (
          <div className="mx-auto flex h-48 w-full max-w-[480px] items-center justify-center rounded-xl bg-white shadow-[0_20px_60px_rgba(70,130,255,0.15)]">
            <div className="flex items-center gap-3 text-[#6b7891]">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-[#e0e7f3] border-t-[#4a8bff]" />
              <span className="text-sm font-medium">챌린지를 발급받는 중…</span>
            </div>
          </div>
        )}

        {/* active */}
        {status === 'active' && spec && (
          <ImageGridCaptcha
            spec={spec}
            status={status}
            error={error}
            onSubmit={handleSubmit}
            onRefresh={start}
          />
        )}

        {/* success */}
        {status === 'success' && (
          <div className="mx-auto w-full max-w-[480px] rounded-xl bg-white p-8 shadow-[0_20px_60px_rgba(70,130,255,0.15)]">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-100 text-2xl">
                ✅
              </div>
              <div>
                <div className="text-lg font-bold text-[#1d2a44]">검증 성공</div>
                <div className="text-xs text-[#6b7891]">
                  아래 captcha_token 을 기업 백엔드 /v1/siteverify 로 전달
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

        {/* fail */}
        {status === 'fail' && (
          <div className="mx-auto w-full max-w-[480px] rounded-xl bg-white p-8 shadow-[0_20px_60px_rgba(70,130,255,0.15)]">
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
