// =============================================================================
// MediaPipe Face Mesh 랜드마크 → 안면 동작 판정 유틸
// -----------------------------------------------------------------------------
// 모든 함수는 normalized landmarks (배열, 각 원소 {x:0..1, y:0..1, z:number}) 를
// 받아 boolean / float 을 반환하는 순수 함수.
//
// **좌표 컨벤션 — 매우 중요**
//   - MediaPipe 는 un-mirrored 원본 프레임을 분석하므로, 사용자의 physical LEFT
//     eye 는 이미지의 RIGHT 쪽(high X)에 위치한다 (카메라 시점 기준).
//   - 화면에 보여주는 영상/캔버스는 CSS scaleX(-1) 로 거울 모드.
//   - 본 모듈의 모든 "left/right" 명명은 **사용자(주체) 관점** 으로 통일한다.
//     즉 `LEFT_EYE_EAR_INDICES` 는 "사용자의 왼쪽 눈" 을 가리키고, MediaPipe 의
//     FACEMESH_LEFT_EYE 와는 정반대다.
//   - `getHeadYaw` 의 부호도 사용자 관점: 양수 = 사용자가 본인의 오른쪽으로 회전,
//     음수 = 사용자가 본인의 왼쪽으로 회전.
// =============================================================================

// 눈 EAR 6점 — 사용자 관점 좌/우.
// LEFT  = 사용자의 physical left eye  = 이미지의 우측 눈 (MediaPipe FACEMESH_RIGHT_EYE 영역)
// RIGHT = 사용자의 physical right eye = 이미지의 좌측 눈 (MediaPipe FACEMESH_LEFT_EYE 영역)
export const LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380];
export const RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144];

// 머리 자세 추정용 — 광대 인덱스는 이미지 좌/우 (un-mirrored) 기준.
const NOSE_TIP = 1;
const FOREHEAD = 10;
const CHIN = 152;
const IMG_LEFT_CHEEK = 234;   // 이미지 좌측 광대 = 사용자의 우측 광대
const IMG_RIGHT_CHEEK = 454;  // 이미지 우측 광대 = 사용자의 좌측 광대

// 입
const MOUTH_LEFT = 61;
const MOUTH_RIGHT = 291;
const MOUTH_TOP = 13;
const MOUTH_BOTTOM = 14;

// EAR 임계치
const EAR_THRESHOLD = 0.2;

// 머리 회전 임계치 (도)
const YAW_THRESHOLD_DEG = 15;

// 미소 임계치 — 입 가로/세로 비율
const SMILE_RATIO_THRESHOLD = 4.0;

// NOD 검출용
const NOD_WINDOW_MS = 500;       // 0.5초 윈도우
const NOD_RANGE_THRESHOLD = 0.02; // 정규화 좌표(0..1) 기준 노이즈 컷오프


// ---------------------------------------------------------------------------
// 기본 거리 / 보조 함수
// ---------------------------------------------------------------------------

function dist2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function ear(landmarks, [p1, p2, p3, p4, p5, p6]) {
  // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
  const vert1 = dist2D(landmarks[p2], landmarks[p6]);
  const vert2 = dist2D(landmarks[p3], landmarks[p5]);
  const horz = dist2D(landmarks[p1], landmarks[p4]);
  if (horz < 1e-6) return 0;
  return (vert1 + vert2) / (2 * horz);
}


// ---------------------------------------------------------------------------
// 동작 판정 (boolean / 수치)
// ---------------------------------------------------------------------------

export function leftEyeEAR(landmarks) {
  return ear(landmarks, LEFT_EYE_EAR_INDICES);
}

export function rightEyeEAR(landmarks) {
  return ear(landmarks, RIGHT_EYE_EAR_INDICES);
}

export function isLeftEyeClosed(landmarks) {
  return leftEyeEAR(landmarks) < EAR_THRESHOLD;
}

export function isRightEyeClosed(landmarks) {
  return rightEyeEAR(landmarks) < EAR_THRESHOLD;
}

/**
 * 머리 좌우 회전(yaw, 도) 추정 — **사용자 관점** 부호.
 * - 양수: 사용자가 본인의 오른쪽으로 머리를 돌림
 * - 음수: 사용자가 본인의 왼쪽으로 머리를 돌림
 *
 * 구현: 코끝 X 가 양쪽 광대 중점 대비 어느 쪽으로 치우쳤는지 측정한 뒤
 *      이미지 좌표계(좌→오 X 증가) 와 사용자 관점이 거울 반전이므로 부호를 뒤집는다.
 *      → 사용자가 본인의 왼쪽으로 돌리면 코가 이미지 우측으로 이동(positive offset),
 *         부호 반전 후 negative yaw 로 보고됨.
 *
 * 정확한 3D pose 가 아닌 근사치. ±15° 임계치만 의미 있게 본다.
 */
export function getHeadYaw(landmarks) {
  const nose = landmarks[NOSE_TIP];
  const leftCheek = landmarks[IMG_LEFT_CHEEK];
  const rightCheek = landmarks[IMG_RIGHT_CHEEK];
  const midX = (leftCheek.x + rightCheek.x) / 2;
  const cheekWidth = Math.abs(rightCheek.x - leftCheek.x);
  if (cheekWidth < 1e-6) return 0;
  const offsetRatio = (nose.x - midX) / cheekWidth; // -0.5 ~ 0.5 정도 (이미지 좌표계 기준)
  // 정면 0, 완전 옆모습 ±0.5 → 약 ±90°. 거울 반전을 사용자 관점으로 보정 (부호 뒤집기).
  return -offsetRatio * 180;
}

/**
 * 머리 위/아래 (pitch) 근사값. 코끝 Y 가 이마-턱 중점에서 얼마나 떨어졌는지 비율.
 * 양수: 고개가 아래로 (코가 중점보다 아래)
 * 음수: 고개가 위로
 */
export function getHeadPitch(landmarks) {
  const nose = landmarks[NOSE_TIP];
  const forehead = landmarks[FOREHEAD];
  const chin = landmarks[CHIN];
  const midY = (forehead.y + chin.y) / 2;
  const faceHeight = Math.abs(chin.y - forehead.y);
  if (faceHeight < 1e-6) return 0;
  return (nose.y - midY) / faceHeight; // -0.5 ~ 0.5
}

/**
 * 미소 판정. 입 가로/세로 비율이 임계치 이상이면 true.
 * 평상시 ~2.5, 미소 ~4+, 큰 웃음 ~5+.
 */
export function isSmiling(landmarks) {
  const width = dist2D(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]);
  const height = dist2D(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]);
  if (height < 1e-6) return false;
  return width / height > SMILE_RATIO_THRESHOLD;
}

/**
 * 끄덕임 검출. 최근 NOD_WINDOW_MS 안의 코끝 Y 변화량이 임계치 이상이고
 * 방향 전환이 1회 이상 있으면 true.
 *
 * @param {Array<{y:number,t:number}>} noseHistory FaceMissionCaptcha 가 매 프레임 push.
 */
export function detectNod(noseHistory) {
  const now = Date.now();
  const recent = noseHistory.filter((p) => now - p.t < NOD_WINDOW_MS);
  if (recent.length < 5) return false;

  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of recent) {
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  if (maxY - minY < NOD_RANGE_THRESHOLD) return false;

  let directionChanges = 0;
  let prevDir = 0;
  for (let i = 1; i < recent.length; i += 1) {
    const dy = recent[i].y - recent[i - 1].y;
    if (Math.abs(dy) < 1e-4) continue;
    const dir = dy > 0 ? 1 : -1;
    if (prevDir !== 0 && dir !== prevDir) directionChanges += 1;
    prevDir = dir;
  }
  return directionChanges >= 1;
}


// ---------------------------------------------------------------------------
// 단일 진입점 — 지시 타입 → 판정 결과
// ---------------------------------------------------------------------------

/**
 * 백엔드 FaceInstructionType 과 1:1 대응.
 * @param {string} type
 * @param {Array} landmarks 468 normalized landmarks
 * @param {Array} noseHistory NOD 검출용 누적 기록
 */
export function detectInstruction(type, landmarks, noseHistory) {
  switch (type) {
    case 'blink_left':
      return isLeftEyeClosed(landmarks);
    case 'blink_right':
      return isRightEyeClosed(landmarks);
    case 'turn_left':
      return getHeadYaw(landmarks) < -YAW_THRESHOLD_DEG;
    case 'turn_right':
      return getHeadYaw(landmarks) > YAW_THRESHOLD_DEG;
    case 'smile':
      return isSmiling(landmarks);
    case 'nod':
      return detectNod(noseHistory);
    default:
      return false;
  }
}
