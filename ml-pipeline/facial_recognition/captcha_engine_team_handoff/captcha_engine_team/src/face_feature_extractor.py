from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# v3(상대좌표 + 시간정규화) 모델이 학습한 피처 순서/네이밍.
# preprocessing/extract_features_time_norm.py 의 SEQ_FEATURE_NAMES와 동일해야 한다 —
# 순서가 다르면 ONNX 모델 입력이 어긋난다.
SELECTED_FEATURES = [
    "ear",
    "mar",
    "smile_w",
    "nose_x_rel",
    "nose_y_rel",
    "cx_rel",
    "cy_rel",
    "roll",
    "yaw",
    "pitch",
    "nose_dx_tn",
    "nose_dy_tn",
    "center_dx_tn",
    "center_dy_tn",
    "nose_speed_tn",
    "ear_vel_tn",
    "mar_vel_tn",
    "yaw_vel_tn",
    "pitch_vel_tn",
    "roll_vel_tn",
]

# 학습 코드(preprocessing/extract_features_time_norm.py)와 동일한 landmark 인덱스.
LEFT_EYE            = [33, 160, 158, 133, 153, 144]
RIGHT_EYE           = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT_IDX       = 61
MOUTH_RIGHT_IDX      = 291
MOUTH_TOP_IDX        = 13
MOUTH_BOTTOM_IDX     = 14
NOSE_TIP_IDX         = 4
FACE_LEFT_IDX        = 234
FACE_RIGHT_IDX       = 454
EYE_LEFT_OUTER_IDX   = 33
EYE_RIGHT_OUTER_IDX  = 263

# ─────────────────────────────────────────────────────────────────────────────
# 2026-06-24 수정: "실시간 캡처는 항상 R_live_clip과 같은 30fps(frame_interval=1)"
# 라는 기존 가정이 캡챠위젯팀 실측(프레임 간격 67~119ms, 평균 ~83ms - setInterval
# 지터 + MediaPipe 추론 부하)으로 반증됨. frame_interval=1로 고정하면 velocity_tn이
# 학습분포보다 평균 ~2.5배 과대해지고, 클립 내에서도 프레임마다 스케일이 들쑥날쑥해진다
# (R_live_clip vs S_dataset_sequence 시간정규화 버그와 같은 종류의 문제가 더 작은
# 규모로 재현됨 - RETROSPECTIVE 참고).
#
# 학습 코드(preprocessing/extract_features_time_norm.py)의 frame_interval은 실측
# ms가 아니라 "파일명 인덱스 간격"이며, R_live_clip은 항상 frame_interval=1로
# 추정됐고 이는 DEFAULT_FPS=30(33.33ms/frame) 네이티브 캡처를 가정한 것이다.
# 즉 학습이 쓴 "1 frame_interval 단위" = 33.33ms. 실시간 timestamp_ms가 있으면
# 이 기준 단위에 맞춰 실측 Δt를 보정해야 학습분포와 스케일이 맞는다:
#
#   base_interval_ms = 1000 / DEFAULT_FPS        (= 33.33ms)
#   fi_eff           = Δt_ms / base_interval_ms
#   feature_tn       = raw_diff / fi_eff
#                    = raw_diff * base_interval_ms / Δt_ms
#
# timestamp_ms가 없는 호출자는 여전히 FRAME_INTERVAL=1 fallback을 쓴다 - 아래
# _build_seq_array()의 분기와 로그 경고 참고.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FPS = 30
BASE_INTERVAL_MS = 1000.0 / DEFAULT_FPS  # 학습이 가정한 "frame_interval=1 단위" (≈33.33ms)
FRAME_INTERVAL = 1  # timestamp_ms 미제공 시 fallback (기존 동작 유지)

# Δt_ms 방어용 clamp. 둘 다 base_interval_ms 기준 배수로 잡아 30fps 가정과 일관되게 한다.
MIN_DT_MS = 1.0                     # 0/음수(timestamp 역전, 중복 timestamp) 방어
MAX_DT_MS = 10.0 * BASE_INTERVAL_MS  # ≈333ms — 이보다 크면 프레임 드롭/일시정지로 간주, median Δt로 대체

MIN_VALID_FRAMES = 3


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _ear(points: np.ndarray, idx: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [points[i] for i in idx]
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * max(_dist(p1, p4), 1e-6)
    return float(vertical / horizontal)


def _mar(points: np.ndarray) -> float:
    vert = _dist(points[MOUTH_TOP_IDX], points[MOUTH_BOTTOM_IDX])
    horiz = _dist(points[MOUTH_LEFT_IDX], points[MOUTH_RIGHT_IDX])
    return float(vert / horiz) if horiz else 0.0


def _head_roll(points: np.ndarray) -> float:
    dx = points[EYE_RIGHT_OUTER_IDX][0] - points[EYE_LEFT_OUTER_IDX][0]
    dy = points[EYE_RIGHT_OUTER_IDX][1] - points[EYE_LEFT_OUTER_IDX][1]
    return float(np.degrees(np.arctan2(dy, dx)))


def _head_yaw(points: np.ndarray) -> float:
    face_cx = (points[FACE_LEFT_IDX][0] + points[FACE_RIGHT_IDX][0]) / 2.0
    half_w = abs(points[FACE_RIGHT_IDX][0] - points[FACE_LEFT_IDX][0]) / 2.0 + 1e-6
    return float((points[NOSE_TIP_IDX][0] - face_cx) / half_w)


def _head_pitch(points: np.ndarray) -> float:
    eye_y = (points[EYE_LEFT_OUTER_IDX][1] + points[EYE_RIGHT_OUTER_IDX][1]) / 2.0
    mouth_y = (points[MOUTH_LEFT_IDX][1] + points[MOUTH_RIGHT_IDX][1]) / 2.0
    mid_y = (eye_y + mouth_y) / 2.0
    half_h = abs(mouth_y - eye_y) / 2.0 + 1e-6
    return float((points[NOSE_TIP_IDX][1] - mid_y) / half_h)


def _extract_frame_raw(points: np.ndarray) -> dict[str, float]:
    """학습 코드의 extract_frame_raw()와 동일한 bbox 정규화 피처.

    nose_x/y, cx/cy는 face_w(좌우 얼굴 폭) 기준으로 정규화해 카메라 거리·얼굴
    크기에 무관하게 만든다. _rel/_tn 변환은 클립 단위(여러 프레임)로 별도 처리한다.
    """
    face_w = _dist(points[FACE_LEFT_IDX], points[FACE_RIGHT_IDX]) + 1e-6
    face_cx = (points[FACE_LEFT_IDX][0] + points[FACE_RIGHT_IDX][0]) / 2.0
    face_cy = (points[FACE_LEFT_IDX][1] + points[FACE_RIGHT_IDX][1]) / 2.0

    ear_l = _ear(points, LEFT_EYE)
    ear_r = _ear(points, RIGHT_EYE)

    cx_raw = (points[33][0] + points[133][0] + points[362][0] + points[263][0]) / 4.0
    cy_raw = (points[33][1] + points[133][1] + points[362][1] + points[263][1]) / 4.0

    smile_w = _dist(points[MOUTH_LEFT_IDX], points[MOUTH_RIGHT_IDX]) / face_w

    return {
        "ear":    (ear_l + ear_r) / 2.0,
        "mar":    _mar(points),
        "smile_w": smile_w,
        "nose_x": (points[NOSE_TIP_IDX][0] - face_cx) / face_w,
        "nose_y": (points[NOSE_TIP_IDX][1] - face_cy) / face_w,
        "cx":     (cx_raw - face_cx) / face_w,
        "cy":     (cy_raw - face_cy) / face_w,
        "roll":   _head_roll(points),
        "yaw":    _head_yaw(points),
        "pitch":  _head_pitch(points),
    }


def _interpolate(frames: list[dict[str, float] | None]) -> list[dict[str, float] | None]:
    """학습 코드의 interpolate_frames()와 동일한 forward-fill → backward-fill."""
    result = list(frames)
    n = len(result)

    last = None
    for i in range(n):
        if result[i] is not None:
            last = result[i]
        elif last is not None:
            result[i] = last

    last = None
    for i in range(n - 1, -1, -1):
        if result[i] is not None:
            last = result[i]
        elif last is not None:
            result[i] = last

    return result


def _compute_fi_eff(timestamps_ms: list[float] | None, n: int) -> list[float] | None:
    """clip의 각 프레임 i(i>=1)에 대해 i-1→i 구간의 fi_eff(=Δt_ms/BASE_INTERVAL_MS)를
    계산한다. index 0은 velocity 계산에 쓰이지 않으므로 placeholder(None)로 둔다.

    timestamps_ms가 없거나 clip 길이와 안 맞으면 None을 반환해 호출부가
    FRAME_INTERVAL=1 fallback을 쓰게 한다.

    방어 로직:
    - Δt_ms <= 0 (timestamp 역전/중복) 또는 Δt_ms > MAX_DT_MS(프레임 드롭/일시정지로
      간주) → 그 구간만 median Δt로 대체. 한 구간의 이상치가 velocity를 한쪽으로
      튀게 만드는 것을 막는다.
    - median 계산 자체에 쓸 유효 Δt가 하나도 없으면(전부 비정상) BASE_INTERVAL_MS로
      대체 - 학습이 가정한 30fps 기준으로 안전하게 떨어진다.
    - 최종 Δt는 MIN_DT_MS로 하한 clamp (분모가 0에 가까워 fi_eff가 폭발하는 것 방지).
    """
    if not timestamps_ms or len(timestamps_ms) != n:
        return None

    raw_dts = [float(timestamps_ms[i]) - float(timestamps_ms[i - 1]) for i in range(1, n)]
    valid_dts = [dt for dt in raw_dts if 0 < dt <= MAX_DT_MS]
    median_dt = float(np.median(valid_dts)) if valid_dts else BASE_INTERVAL_MS

    fi_eff: list[float | None] = [None]
    for dt in raw_dts:
        if dt <= 0 or dt > MAX_DT_MS:
            logger.warning(
                "비정상 Δt 감지(%.1fms) - median(%.1fms)으로 대체합니다. "
                "프레임 드롭/타임스탬프 역전 가능성을 확인하세요.",
                dt, median_dt,
            )
            dt = median_dt
        dt = max(dt, MIN_DT_MS)
        fi_eff.append(dt / BASE_INTERVAL_MS)
    return fi_eff


def _build_seq_array(
    clip: list[dict[str, float]],
    fi_eff: list[float] | None = None,
) -> np.ndarray:
    """학습 코드의 build_seq_array()와 동일한 클립 단위 상대좌표 + 시간정규화 velocity.

    fi_eff가 주어지면(= 실측 timestamp_ms 기반) 프레임마다 다른 fi_eff[i]로
    나눠 실제 캡처 간격을 반영한다. fi_eff가 없으면 기존처럼 FRAME_INTERVAL(=1)
    고정값을 쓴다 - 이 경우 캡처 간격이 학습이 가정한 33.33ms(30fps)와 다르면
    velocity_tn 스케일이 학습분포와 어긋날 수 있다는 점에 주의해야 한다.
    """
    n = len(clip)
    using_real_dt = fi_eff is not None
    if not using_real_dt:
        logger.warning(
            "timestamp_ms 미제공 - frame_interval=1 fallback 사용. "
            "실제 캡처 간격이 33.33ms(30fps)와 다르면 velocity_tn이 학습분포와 "
            "어긋날 수 있습니다 (정확도 리스크)."
        )

    mean_nose_x = float(np.mean([f["nose_x"] for f in clip]))
    mean_nose_y = float(np.mean([f["nose_y"] for f in clip]))
    mean_cx     = float(np.mean([f["cx"]     for f in clip]))
    mean_cy     = float(np.mean([f["cy"]     for f in clip]))

    out = np.zeros((n, len(SELECTED_FEATURES)), dtype=np.float32)

    for i, frame in enumerate(clip):
        nose_x_rel = frame["nose_x"] - mean_nose_x
        nose_y_rel = frame["nose_y"] - mean_nose_y
        cx_rel     = frame["cx"]     - mean_cx
        cy_rel     = frame["cy"]     - mean_cy

        abs_feats = np.array([
            frame["ear"], frame["mar"], frame["smile_w"],
            nose_x_rel, nose_y_rel, cx_rel, cy_rel,
            frame["roll"], frame["yaw"], frame["pitch"],
        ], dtype=np.float32)

        if i == 0:
            vel_feats = np.zeros(10, dtype=np.float32)
        else:
            fi = fi_eff[i] if using_real_dt else FRAME_INTERVAL

            prev = clip[i - 1]
            nose_dx   = nose_x_rel - (prev["nose_x"] - mean_nose_x)
            nose_dy   = nose_y_rel - (prev["nose_y"] - mean_nose_y)
            center_dx = cx_rel     - (prev["cx"]     - mean_cx)
            center_dy = cy_rel     - (prev["cy"]     - mean_cy)

            nose_dx_tn    = nose_dx   / fi
            nose_dy_tn    = nose_dy   / fi
            center_dx_tn  = center_dx / fi
            center_dy_tn  = center_dy / fi
            nose_speed_tn = math.hypot(nose_dx, nose_dy) / fi

            ear_vel_tn   = (frame["ear"]   - prev["ear"])   / fi
            mar_vel_tn   = (frame["mar"]   - prev["mar"])   / fi
            yaw_vel_tn   = (frame["yaw"]   - prev["yaw"])   / fi
            pitch_vel_tn = (frame["pitch"] - prev["pitch"]) / fi
            roll_vel_tn  = (frame["roll"]  - prev["roll"])  / fi

            vel_feats = np.array([
                nose_dx_tn, nose_dy_tn,
                center_dx_tn, center_dy_tn,
                nose_speed_tn,
                ear_vel_tn, mar_vel_tn,
                yaw_vel_tn, pitch_vel_tn, roll_vel_tn,
            ], dtype=np.float32)

        out[i] = np.concatenate([abs_feats, vel_feats])

    return out


class FaceFeatureExtractor:
    """MediaPipe FaceMesh 기반 v3(상대좌표+시간정규화) 피처 추출기.

    preprocessing/extract_features_time_norm.py와 동일한 공식을 실시간 프레임
    버퍼에 적용한다. 출력 shape: (target_frames, 20), 피처 순서는 SELECTED_FEATURES.

    extract_from_frames()에 timestamps_ms(프레임별 절대 timestamp, ms)를 같이
    넘기면 실제 캡처 간격을 반영해 velocity_tn을 계산한다(권장) - 캡처 간격이
    불균일하거나 30fps와 다른 환경(예: 실측 67~119ms)에서는 반드시 넘겨야
    학습분포와 스케일이 맞는다. 생략하면 frame_interval=1 fallback을 쓰며
    로그 경고가 출력된다.
    """

    def __init__(self, target_frames: int = 16, max_num_faces: int = 1):
        self.target_frames = int(target_frames)
        self.max_num_faces = int(max_num_faces)

        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._face_mesh.close()

    def extract_from_frames(
        self,
        frames: list[np.ndarray],
        timestamps_ms: list[float] | None = None,
    ) -> tuple[np.ndarray, int, dict]:
        """frames: BGR 이미지 리스트. timestamps_ms: frames와 같은 길이의 프레임별
        절대 타임스탬프(ms, 예: performance.now()/Date.now()). 제공하면 프레임 간
        실제 Δt로 velocity_tn을 보정하고(권장), 없으면 frame_interval=1 fallback을
        쓴다(정확도 리스크 - 로그 경고 참고)."""
        if timestamps_ms is not None and len(timestamps_ms) != len(frames):
            raise ValueError(
                f"timestamps_ms 길이({len(timestamps_ms)})가 frames 길이({len(frames)})와 다릅니다."
            )

        indices = self._sample_indices(len(frames))
        selected_frames = [frames[i] for i in indices]
        selected_timestamps = (
            [float(timestamps_ms[i]) for i in indices] if timestamps_ms is not None else None
        )
        n = len(selected_frames)

        raw_frames: list[dict[str, float] | None] = []
        face_detected_frames = 0
        for frame in selected_frames:
            feat = self._extract_one_raw(frame)
            if feat is not None:
                face_detected_frames += 1
            raw_frames.append(feat)

        valid_count = face_detected_frames
        x_seq = np.zeros((self.target_frames, len(SELECTED_FEATURES)), dtype=np.float32)
        face_detected = valid_count >= MIN_VALID_FRAMES

        if face_detected:
            filled = _interpolate(raw_frames)
            # 보간 후에도 양 끝에 None이 남을 수 있다(얼굴이 한 번도 검출되지 않은
            # 경우는 위에서 걸러지지만, 만약을 위해 0-feature로 방어).
            clip = [f if f is not None else _zero_raw_feature() for f in filled]
            fi_eff = _compute_fi_eff(selected_timestamps, n)
            seq = _build_seq_array(clip, fi_eff=fi_eff)
            use_n = min(n, self.target_frames)
            x_seq[:use_n] = seq[:use_n]

        seq_length = max(1, min(self.target_frames, n, valid_count or n))
        info = {
            "target_frames": self.target_frames,
            "input_frames": len(frames),
            "used_frames": n,
            "valid_frames": valid_count,
            "face_detected_frames": face_detected_frames,
            "face_detected": face_detected,
            "face_detect_rate": face_detected_frames / max(1, n),
            "selected_features": SELECTED_FEATURES,
            "used_real_timestamps": selected_timestamps is not None,
        }
        return x_seq, seq_length, info

    def _sample_indices(self, count: int) -> list[int]:
        """frames 원본 인덱스를 target_frames 길이로 다운/업샘플링한다. frames와
        timestamps_ms에 동일하게 적용해야 둘의 정렬이 깨지지 않는다."""
        if count == 0:
            return []
        if count >= self.target_frames:
            return np.linspace(0, count - 1, self.target_frames).astype(int).tolist()

        indices = list(range(count))
        while len(indices) < self.target_frames:
            indices.append(count - 1)
        return indices

    def _extract_one_raw(self, frame_bgr: np.ndarray) -> dict[str, float] | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        result = self._face_mesh.process(image_rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        # MediaPipe landmark.x/y는 각각 프레임 너비/높이로 독립 정규화된다.
        # 프레임이 정사각형이 아니면(폰 카메라는 보통 16:9 또는 9:16) y축이
        # 체계적으로 왜곡되어 EAR 등 거리 기반 피처가 학습 데이터의 촬영 비율과
        # 다른 사용자 카메라 비율만으로 달라진다 (RETROSPECTIVE 참고). y를
        # 너비 기준 단위로 환산해 이후 모든 거리 계산이 단일 단위를 쓰도록 만든다.
        h, w = frame_bgr.shape[:2]
        aspect_ratio = (w / h) if h else 1.0
        points = np.asarray(
            [[lm.x, lm.y / aspect_ratio, lm.z] for lm in landmarks], dtype=np.float32
        )

        return _extract_frame_raw(points)

    @staticmethod
    def _zero_feature() -> dict[str, float]:
        return {name: 0.0 for name in SELECTED_FEATURES}


def _zero_raw_feature() -> dict[str, float]:
    return {
        "ear": 0.0, "mar": 0.0, "smile_w": 0.0,
        "nose_x": 0.0, "nose_y": 0.0, "cx": 0.0, "cy": 0.0,
        "roll": 0.0, "yaw": 0.0, "pitch": 0.0,
    }


class FrameBuffer:
    """frame_bgr와 timestamp_ms(선택)를 함께 보관한다. timestamp_ms를 매번 넘기면
    as_timestamps_list()로 꺼내 FaceFeatureExtractor.extract_from_frames()에
    그대로 전달할 수 있다 - 일부만 timestamp_ms를 넘기고 일부는 생략하면 정렬이
    깨지므로 섞어 쓰지 않는다(둘 다 None이거나 둘 다 값이 있어야 함)."""

    def __init__(self, maxlen: int = 16):
        self.frames: deque[np.ndarray] = deque(maxlen=maxlen)
        self.timestamps: deque[float | None] = deque(maxlen=maxlen)

    def append(self, frame: np.ndarray, timestamp_ms: float | None = None) -> None:
        self.frames.append(frame.copy())
        self.timestamps.append(float(timestamp_ms) if timestamp_ms is not None else None)

    def ready(self) -> bool:
        return len(self.frames) == self.frames.maxlen

    def as_list(self) -> list[np.ndarray]:
        return list(self.frames)

    def as_timestamps_list(self) -> list[float] | None:
        """timestamp가 하나라도 빠져 있으면 None을 반환한다 - extract_from_frames()가
        부분적인 timestamp 배열로 잘못 정렬된 Δt를 계산하지 않도록 방어."""
        ts = list(self.timestamps)
        if not ts or any(t is None for t in ts):
            return None
        return ts
