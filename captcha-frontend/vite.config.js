import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// MediaPipe 패키지(face_mesh / camera_utils / drawing_utils)는 IIFE-CommonJS 형태라
// Rolldown 의 정적 export 분석이 실패함. esbuild 사전 번들링에 강제로 포함시키고
// CJS interop 옵션을 켜서 named export 가 동작하도록 한다.
export default defineConfig({
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
