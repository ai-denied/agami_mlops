import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import CaptchaPage from './pages/CaptchaPage';
import ContextCaptchaPage from './pages/ContextCaptchaPage';
import EmbedEntry from './pages/EmbedEntry';

// =============================================================================
// 라우팅 엔트리.
//   /                → 홈 (캡챠 페이지로 이동하는 링크)
//   /captcha         → 행동 기반 캡챠 (FastAPI /v1/* 사용, kind 드롭다운)
//   /captcha/context → 감정 맥락 추론 캡챠 전용 페이지 (kind 고정)
//   /embed           → iframe 임베드 진입점 (?kind=&difficulty=)
//
// BrowserRouter basename:
//   WIDGET_BUILD=1  → '/widget' (FastAPI 가 /widget/ 아래 마운트)
//   기본 dev/build  → '/'
//   import.meta.env.BASE_URL 는 vite 가 base 설정값을 그대로 client 에 노출.
//   trailing slash 를 제거해 BrowserRouter 가 받는 형태로 정규화.
// =============================================================================

const ROUTER_BASENAME = (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '') || '/';

function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-slate-900 px-6 text-slate-100">
      <div className="max-w-xl text-center">
        <div className="mb-2 text-sm uppercase tracking-[0.3em] text-emerald-400">
          AI Captcha Engine
        </div>
        <h1 className="text-4xl font-bold tracking-tight">행동 기반 AI 캡챠</h1>
        <p className="mt-4 text-slate-300">
          어두운 화면에서 손전등으로 목표 물건을 찾는 행동 기반 캡챠입니다.
          마우스 궤적과 정답 좌표를 함께 분석해 사람/봇을 판별합니다.
        </p>
        <Link
          to="/captcha"
          className="mt-8 inline-block rounded-lg bg-emerald-500 px-6 py-3 text-base font-semibold text-slate-900 shadow transition-colors hover:bg-emerald-400"
        >
          캡챠 데모 열기 →
        </Link>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter basename={ROUTER_BASENAME}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/captcha" element={<CaptchaPage />} />
        <Route path="/captcha/context" element={<ContextCaptchaPage />} />
        <Route path="/embed" element={<EmbedEntry />} />
      </Routes>
    </BrowserRouter>
  );
}
