import { useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useCaptcha } from '../hooks/useCaptcha';
import CaptchaRouter from '../components/CaptchaRouter';

// =============================================================================
// /embed 라우트 — iframe 임베드 전용 진입점
//   URL 쿼리:
//     kind        flashlight (기본) | face_mission | context_inference
//     difficulty  easy (기본) | normal | medium | hard
//                 ('normal' 은 백엔드 enum 'medium' 으로 자동 매핑)
//
//   동작:
//     1) 마운트 즉시 캡챠 발급 + 시작 (Home/선택 화면 안 거침)
//     2) status 가 success/fail 로 전환되는 순간 단 한 번 parent.postMessage 발신
//        payload = { type:'agami-result', success, challengeId, challengeType, captchaToken }
//
//   부모 페이지 수신 예:
//     window.addEventListener('message', e => {
//       if (e.data?.type === 'agami-result') { /* e.data.success / e.data.captchaToken */ }
//     });
// =============================================================================

const ALLOWED_KINDS = ['flashlight', 'face_mission', 'context_inference'];
// 'normal' 은 사용자 친화 별칭. 백엔드는 medium 만 인식.
const DIFFICULTY_MAP = {
  easy: 'easy',
  normal: 'medium',
  medium: 'medium',
  hard: 'hard',
};

function postResult({ success, spec, token }) {
  if (typeof window === 'undefined') return;
  try {
    window.parent.postMessage(
      {
        type: 'agami-result',
        success,
        challengeId: spec?.challenge_id ?? null,
        challengeType: spec?.kind ?? null,
        captchaToken: success ? token ?? null : null,
      },
      '*',
    );
  } catch {
    // 부모 컨텍스트가 없거나(iframe 아님) cross-origin 정책 위반 시 무시.
  }
}

export default function EmbedEntry() {
  const [searchParams] = useSearchParams();
  const rawKind = (searchParams.get('kind') ?? 'flashlight').toLowerCase();
  const kind = ALLOWED_KINDS.includes(rawKind) ? rawKind : 'flashlight';
  const rawDiff = (searchParams.get('difficulty') ?? 'easy').toLowerCase();
  const difficulty = DIFFICULTY_MAP[rawDiff] ?? 'easy';

  const { status, spec, token, error, start, submit } = useCaptcha({ kind, difficulty });

  // 첫 마운트 시 자동 시작 — idle → loading → active 자동 전환
  const startedRef = useRef(false);
  useEffect(() => {
    if (startedRef.current) return;
    if (status === 'idle') {
      startedRef.current = true;
      start();
    }
  }, [status, start]);

  // status 종료 시 단 한 번 postMessage 발신
  const sentRef = useRef(false);
  useEffect(() => {
    if (sentRef.current) return;
    if (status === 'success') {
      postResult({ success: true, spec, token });
      sentRef.current = true;
    } else if (status === 'fail') {
      postResult({ success: false, spec, token });
      sentRef.current = true;
    }
  }, [status, spec, token]);

  // 재시도: postMessage 한 번 발신 후에는 부모가 iframe reload 로 처리하는 것이 원칙.
  // 다만 UX 차원의 자체 재시도 버튼 1회만 허용 — sent flag 해제 후 start().
  const handleRetry = () => {
    sentRef.current = false;
    start();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#f5f8ff] to-[#e8f0ff] flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-5xl">
        {(status === 'idle' || status === 'loading') && (
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
            onSubmit={submit}
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
                  결과가 부모 페이지로 전송되었습니다.
                </div>
              </div>
            </div>
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
                  {error?.message || '알 수 없는 오류'}
                  {error?.code ? <span className="text-rose-500 ml-1">({error.code})</span> : null}
                </div>
              </div>
            </div>
            <div className="mt-5">
              <button
                onClick={handleRetry}
                className="rounded-xl bg-gradient-to-r from-[#4a8bff] to-[#6da5ff] px-4 py-2 text-sm font-bold text-white shadow-[0_8px_24px_rgba(74,139,255,0.35)] hover:-translate-y-0.5 transition-transform"
              >
                다시 시도
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
