import { useEffect, useRef, useState } from 'react';
// MediaPipe 패키지는 Closure Compiler IIFE 출력이라 module.exports 가 비어있고
// named export 가 잡히지 않는다. side-effect 로 import 하면 IIFE 가 실행되며
// globalThis (브라우저=window) 에 심볼을 등록하므로, 그 다음 globalThis 에서 읽는다.
import '@mediapipe/face_mesh';
import '@mediapipe/camera_utils';
import '@mediapipe/drawing_utils';
import { detectInstruction } from '../lib/faceDetection';
import FishTimer from './FishTimer';

const g = /** @type {any} */ (globalThis);
const FaceMesh = g.FaceMesh;
const Camera = g.Camera;
const drawConnectors = g.drawConnectors;
const FACEMESH_FACE_OVAL = g.FACEMESH_FACE_OVAL;
const FACEMESH_LEFT_EYE = g.FACEMESH_LEFT_EYE;
const FACEMESH_RIGHT_EYE = g.FACEMESH_RIGHT_EYE;
const FACEMESH_LIPS = g.FACEMESH_LIPS;

// =============================================================================
// 안면 미션 캡챠 (MediaPipe Face Mesh 기반 실시간 자동 감지)
// -----------------------------------------------------------------------------
// 동작 흐름
//   1) 마운트 시 FaceMesh 초기화 + Camera 시작
//   2) 매 프레임 onResults → 랜드마크 추출 → 캔버스에 메쉬 오버레이
//   3) 현재 지시 타입을 detectInstruction 으로 판정. true 가
//      duration_sec 만큼 연속 유지되면 해당 단계 완료, 다음 단계로 자동 진행.
//   4) 모든 단계 완료 시 onSubmit(payload) 1회 호출.
//
// 정리
//   - useEffect cleanup 에서 camera.stop / faceMesh.close / track.stop 모두 수행.
//
// 알려진 한계 (MVP)
//   - 클라이언트 사이드 검출이라 사용자가 마음먹고 우회 가능 (사진/녹화 영상 등).
//   - 진짜 검증은 서버에서 행동 시퀀스 + 시간 + 행동 패턴 종합 분석 필요.
//   - 팀원 AI 모델 합류 시 백엔드로 영상/랜드마크 시퀀스 전송 후 검증으로 교체 예정.
//   - 현재 모델은 클라이언트가 completed_instructions 만 신뢰 보고하는 구조.
// =============================================================================

const ICON_FOR = {
  blink_left: '👁️',
  blink_right: '👁️',
  turn_left: '⬅️',
  turn_right: '➡️',
  smile: '😊',
  nod: '🙇',
};

const COLOR_BLUE = '#4a8bff';
const COLOR_YELLOW = '#fbbf24';
const COLOR_WHITE = 'rgba(255, 255, 255, 0.95)';

// MediaPipe WASM/asset CDN. 패키지 버전과 일치하는 디렉터리를 가리킴.
const MP_FACE_MESH_CDN = (file) =>
  `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;


export default function FaceMissionCaptcha({ spec, onSubmit, onRefresh }) {
  // DOM
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // MediaPipe 인스턴스
  const faceMeshRef = useRef(null);
  const cameraRef = useRef(null);

  // 콜백/상태 미러 ref (onResults 안에서 stale closure 회피)
  const onSubmitRef = useRef(onSubmit);
  const specRef = useRef(spec);
  const instructionIdxRef = useRef(0);
  const progressStartedAtRef = useRef(null);
  const noseHistoryRef = useRef([]); // NOD 검출용
  const completedRef = useRef([]);
  const startedAtRef = useRef(Date.now());
  const submittedRef = useRef(false);
  const advanceTimerRef = useRef(null);

  // 렌더 트리거용 상태
  const [detectionStatus, setDetectionStatus] = useState('initializing');
  // initializing | no_face | instruction_active | instruction_complete | denied | error
  const [currentInstructionIndex, setCurrentInstructionIndex] = useState(0);
  const [progressFraction, setProgressFraction] = useState(0);
  const [timeLeft, setTimeLeft] = useState(spec?.time_limit_sec ?? 30);
  const [hintVisible, setHintVisible] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // ref 동기화
  useEffect(() => { onSubmitRef.current = onSubmit; }, [onSubmit]);
  useEffect(() => {
    specRef.current = spec;
    instructionIdxRef.current = 0;
    progressStartedAtRef.current = null;
    noseHistoryRef.current = [];
    completedRef.current = [];
    submittedRef.current = false;
    startedAtRef.current = Date.now();
    setCurrentInstructionIndex(0);
    setProgressFraction(0);
    setTimeLeft(spec?.time_limit_sec ?? 30);
    setHintVisible(false);
  }, [spec]);

  // 디스플레이용 카운트다운 + 힌트 (자동 fail 은 useCaptcha 훅이 처리)
  useEffect(() => {
    if (!spec) return;
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

  // ---------------------------------------------------------------------------
  // MediaPipe + Camera 초기화 (마운트 시 1회)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    // side-effect import 로도 globalThis 에 심볼이 안 올라온 경우 (네트워크 차단 등)
    // 명확한 에러 UI 로 떨어뜨려 사용자에게 알림.
    if (typeof FaceMesh !== 'function' || typeof Camera !== 'function') {
      setDetectionStatus('error');
      setErrorMessage('MediaPipe 라이브러리 로드 실패 — 페이지를 새로고침 후 다시 시도해주세요.');
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    let cancelled = false;

    const faceMesh = new FaceMesh({ locateFile: MP_FACE_MESH_CDN });
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults((results) => handleResults(results, canvas, ctx));
    faceMeshRef.current = faceMesh;

    const camera = new Camera(video, {
      onFrame: async () => {
        if (cancelled || !faceMeshRef.current) return;
        try {
          await faceMeshRef.current.send({ image: video });
        } catch (err) {
          // 모델이 닫힌 후 들어오는 마지막 frame 등은 무시
          if (!cancelled) console.warn('faceMesh.send failed:', err);
        }
      },
      width: 480,
      height: 480,
    });
    cameraRef.current = camera;

    camera
      .start()
      .then(() => {
        if (!cancelled) setDetectionStatus('no_face');
      })
      .catch((err) => {
        console.error('camera start failed:', err);
        if (cancelled) return;
        if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
          setDetectionStatus('denied');
        } else {
          setDetectionStatus('error');
          setErrorMessage(err?.message || String(err));
        }
      });

    return () => {
      cancelled = true;
      if (advanceTimerRef.current) {
        clearTimeout(advanceTimerRef.current);
        advanceTimerRef.current = null;
      }
      try { camera.stop(); } catch (_) {}
      try { faceMesh.close(); } catch (_) {}
      faceMeshRef.current = null;
      cameraRef.current = null;
      // Camera 클래스가 만든 stream 도 명시적으로 해제 (LED off 보장)
      const stream = video.srcObject;
      if (stream && typeof stream.getTracks === 'function') {
        stream.getTracks().forEach((t) => {
          try { t.stop(); } catch (_) {}
        });
      }
      video.srcObject = null;
    };
  }, []);

  // ---------------------------------------------------------------------------
  // onResults : 매 프레임 호출 (faceMesh.onResults 콜백)
  // ---------------------------------------------------------------------------
  function handleResults(results, canvas, ctx) {
    if (submittedRef.current) return;

    const currentSpec = specRef.current;
    if (!currentSpec) return;

    // 캔버스 크기 동기화 (video 의 실제 해상도에 맞춤)
    const vw = results.image?.width || 480;
    const vh = results.image?.height || 480;
    if (canvas.width !== vw) canvas.width = vw;
    if (canvas.height !== vh) canvas.height = vh;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const lm = results.multiFaceLandmarks?.[0];
    if (!lm) {
      // 얼굴 미검출
      ctx.restore();
      setDetectionStatus('no_face');
      progressStartedAtRef.current = null;
      setProgressFraction(0);
      return;
    }

    // 코 Y 누적 (NOD 검출용)
    noseHistoryRef.current.push({ y: lm[1].y, t: Date.now() });
    if (noseHistoryRef.current.length > 60) noseHistoryRef.current.shift();

    // 현재 지시
    const idx = instructionIdxRef.current;
    const inst = currentSpec.instructions[idx];

    // 메쉬 오버레이
    drawMesh(ctx, lm, inst?.type);
    ctx.restore();

    if (!inst) return;

    // 동작 검출 + 진행도 누적
    const detected = detectInstruction(inst.type, lm, noseHistoryRef.current);
    setDetectionStatus(detected ? 'instruction_active' : 'no_face');

    if (detected) {
      if (progressStartedAtRef.current == null) {
        progressStartedAtRef.current = Date.now();
      }
      const elapsed = Date.now() - progressStartedAtRef.current;
      const target = inst.duration_sec * 1000;
      setProgressFraction(Math.min(1, elapsed / target));

      if (elapsed >= target) {
        // 단계 완료
        completedRef.current.push(inst.type);
        progressStartedAtRef.current = null;
        setProgressFraction(0);
        setDetectionStatus('instruction_complete');

        const nextIdx = idx + 1;
        if (nextIdx >= currentSpec.instructions.length) {
          // 마지막 지시 → 제출 1회
          submittedRef.current = true;
          onSubmitRef.current({
            completed_instructions: [...completedRef.current],
            face_behavioral_data: {
              time_taken_ms: Date.now() - startedAtRef.current,
              steps_count: currentSpec.instructions.length,
            },
          });
        } else {
          // 0.6s 동안 체크마크 보여주고 다음 단계로
          if (advanceTimerRef.current) clearTimeout(advanceTimerRef.current);
          advanceTimerRef.current = setTimeout(() => {
            instructionIdxRef.current = nextIdx;
            setCurrentInstructionIndex(nextIdx);
            setDetectionStatus('instruction_active');
          }, 600);
        }
      }
    } else {
      // 끊기면 진행 게이지 리셋
      if (progressStartedAtRef.current != null) {
        progressStartedAtRef.current = null;
        setProgressFraction(0);
      }
    }
  }

  if (!spec) return null;

  const totalSteps = spec.instructions.length;
  const currentInstruction = spec.instructions[currentInstructionIndex];
  const isCompleteFlash = detectionStatus === 'instruction_complete';

  return (
    <div className="w-full max-w-[520px] min-w-0 bg-white rounded-xl shadow-[0_20px_60px_rgba(70,130,255,0.15)] overflow-hidden mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-[#4a8bff] to-[#6da5ff] text-white">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-white/20 rounded-lg flex items-center justify-center text-lg">
            😶
          </div>
          <div>
            <div className="font-bold text-[15px] leading-tight">안면 미션 캡챠</div>
            <div className="text-xs opacity-85 mt-0.5">카메라가 동작을 자동 감지합니다</div>
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
        {/* 단계 카운터 */}
        <div className="flex items-center justify-between mb-3.5">
          <div className="text-xs text-[#8a96ad] font-semibold uppercase tracking-wide">
            진행 상태
          </div>
          <div className="text-sm font-bold text-[#1d2a44] tabular-nums">
            {Math.min(currentInstructionIndex + 1, totalSteps)}<span className="text-[#8a96ad]">/{totalSteps}</span> 단계
          </div>
        </div>

        {/* 카메라 영역 */}
        <div className="relative w-full aspect-square bg-[#0a0a14] rounded-lg overflow-hidden border-2 border-[#1a1a28]">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-cover"
            style={{ transform: 'scaleX(-1)' }}
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ transform: 'scaleX(-1)', mixBlendMode: 'screen' }}
          />

          {/* 큰 지시문 (상단 오버레이) */}
          {currentInstruction && (
            <div className="absolute top-3 left-3 right-3 flex justify-center pointer-events-none">
              <div className="inline-flex items-center gap-2 bg-black/60 backdrop-blur px-4 py-2 rounded-full">
                <span className="text-xl leading-none">
                  {ICON_FOR[currentInstruction.type] ?? '🎯'}
                </span>
                <span className="text-white font-bold text-sm">
                  {currentInstruction.label}
                </span>
                <span className="text-white/60 text-xs">
                  ({currentInstruction.duration_sec}s)
                </span>
              </div>
            </div>
          )}

          {/* 상태 안내 */}
          {detectionStatus === 'initializing' && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-white text-sm pointer-events-none">
              <div className="flex items-center gap-3">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                얼굴 인식 모델 로딩 중…
              </div>
            </div>
          )}

          {detectionStatus === 'no_face' && (
            <div className="absolute bottom-3.5 left-1/2 -translate-x-1/2 bg-black/70 backdrop-blur px-4 py-2 rounded-full text-xs text-white/90 pointer-events-none">
              📷 얼굴이 보이도록 카메라 앞에 위치해주세요
            </div>
          )}

          {detectionStatus === 'denied' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white px-6 text-center bg-black/80">
              <div className="text-3xl mb-2">📷</div>
              <div className="font-bold mb-1">카메라 권한이 필요합니다</div>
              <div className="text-xs text-white/70 mb-4">
                브라우저 주소창의 카메라 아이콘에서 허용 후 새로고침하세요.
              </div>
              <button
                onClick={onRefresh}
                className="bg-white text-[#2563eb] px-4 py-1.5 rounded-lg text-xs font-bold hover:bg-[#eef4ff] transition-colors"
              >
                다시 시도
              </button>
            </div>
          )}

          {detectionStatus === 'error' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white px-6 text-center bg-black/80">
              <div className="text-3xl mb-2">⚠️</div>
              <div className="font-bold mb-1">카메라를 열 수 없습니다</div>
              <div className="text-xs text-white/70 mb-4 break-words">
                {errorMessage || '알 수 없는 오류'}
              </div>
              <button
                onClick={onRefresh}
                className="bg-white text-[#2563eb] px-4 py-1.5 rounded-lg text-xs font-bold hover:bg-[#eef4ff] transition-colors"
              >
                다시 시도
              </button>
            </div>
          )}

          {/* LIVE 표식 */}
          {(detectionStatus === 'instruction_active' || detectionStatus === 'instruction_complete' || detectionStatus === 'no_face') && (
            <div className="absolute top-3.5 right-3.5 bg-white/10 backdrop-blur px-3 py-1.5 rounded-full text-[11px] text-white/80 flex items-center gap-1.5 pointer-events-none">
              <span className="w-1.5 h-1.5 bg-rose-400 rounded-full animate-pulse" />
              LIVE
            </div>
          )}

          {/* 단계 완료 체크마크 */}
          {isCompleteFlash && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="bg-emerald-500/85 text-white text-5xl w-24 h-24 rounded-full flex items-center justify-center shadow-2xl animate-pulse">
                ✓
              </div>
            </div>
          )}

          {hintVisible && detectionStatus !== 'denied' && detectionStatus !== 'error' && (
            <div className="absolute bottom-12 left-1/2 -translate-x-1/2 bg-amber-400/90 text-amber-950 px-4 py-1.5 rounded-full text-xs font-bold pointer-events-none">
              💡 천천히 또렷하게 동작해보세요
            </div>
          )}
        </div>

        {/* 동작 유지 게이지 (instructionProgressMs / duration_sec * 1000) */}
        <div className="mt-3.5 mb-1">
          <div className="flex items-center justify-between text-xs text-[#8a96ad] mb-1.5">
            <span>현재 동작 유지</span>
            <span className="tabular-nums font-semibold text-[#2563eb]">
              {Math.round(progressFraction * 100)}%
            </span>
          </div>
          <div className="w-full h-2 bg-[#f0f4fb] rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[#4a8bff] to-[#7aa9ff] rounded-full transition-all duration-150"
              style={{ width: `${progressFraction * 100}%` }}
            />
          </div>
        </div>

        {/* 전체 남은 시간 — 물고기 한 마리가 우측에서 좌측으로 헤엄친다 */}
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


// ---------------------------------------------------------------------------
// 메쉬 오버레이 — 현재 지시 타입에 따라 관련 부위만 노란색으로 강조
// ---------------------------------------------------------------------------
function drawMesh(ctx, landmarks, currentType) {
  const FACE_OPTS = { color: COLOR_WHITE, lineWidth: 1.5 };
  const BLUE_OPTS = { color: COLOR_BLUE, lineWidth: 1.5 };
  const HIGHLIGHT_OPTS = { color: COLOR_YELLOW, lineWidth: 2.5 };

  const isBlinkLeft = currentType === 'blink_left';
  const isBlinkRight = currentType === 'blink_right';
  const isSmile = currentType === 'smile';
  const isHeadAction = currentType === 'turn_left'
    || currentType === 'turn_right'
    || currentType === 'nod';

  // 캔버스가 CSS scaleX(-1) 로 거울 반전되므로, MediaPipe 의 LEFT_EYE(이미지 좌측)
  // 는 시각적으로 viewer 의 RIGHT 에 나타난다 = 사용자 관점의 RIGHT eye.
  // 따라서 사용자 관점 highlight 매핑은 다음과 같이 뒤집어서 그린다:
  //   사용자 LEFT eye highlight  → FACEMESH_RIGHT_EYE 에 노란색
  //   사용자 RIGHT eye highlight → FACEMESH_LEFT_EYE  에 노란색
  drawConnectors(ctx, landmarks, FACEMESH_FACE_OVAL, FACE_OPTS);
  drawConnectors(ctx, landmarks, FACEMESH_LEFT_EYE, isBlinkRight ? HIGHLIGHT_OPTS : BLUE_OPTS);
  drawConnectors(ctx, landmarks, FACEMESH_RIGHT_EYE, isBlinkLeft ? HIGHLIGHT_OPTS : BLUE_OPTS);
  drawConnectors(ctx, landmarks, FACEMESH_LIPS, isSmile ? HIGHLIGHT_OPTS : BLUE_OPTS);

  // 코끝 점
  const nose = landmarks[1];
  if (nose) {
    ctx.beginPath();
    ctx.arc(nose.x * ctx.canvas.width, nose.y * ctx.canvas.height, 4, 0, Math.PI * 2);
    ctx.fillStyle = isHeadAction ? COLOR_YELLOW : COLOR_BLUE;
    ctx.fill();
  }
}
