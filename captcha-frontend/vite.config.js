import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// MediaPipe 패키지(face_mesh / camera_utils / drawing_utils)는 IIFE-CommonJS 형태라
// Rolldown 의 정적 export 분석이 실패함. esbuild 사전 번들링에 강제로 포함시키고
// CJS interop 옵션을 켜서 named export 가 동작하도록 한다.
//
// base 분기:
//   WIDGET_BUILD=1 npm run build  → FastAPI 의 /widget/ 아래 마운트되는 iframe 위젯 산출물
//                                   (asset URL 이 /widget/assets/... 로 prefix 됨)
//   npm run dev / npm run build   → 기본 '/' (단독 실행 모드, http://localhost:5173)
//
// 클라이언트 코드 (BrowserRouter basename 등) 는 vite 내장 `import.meta.env.BASE_URL`
// 로 동일 값을 읽는다. process.env 를 직접 client 에서 참조할 수 없기 때문.
const isWidgetBuild = process.env.WIDGET_BUILD === '1'

export default defineConfig({
  base: isWidgetBuild ? '/widget/' : '/',
  plugins: [react()],
  optimizeDeps: {
    include: [
      '@mediapipe/face_mesh',
      '@mediapipe/camera_utils',
      '@mediapipe/drawing_utils',
    ],
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
})
