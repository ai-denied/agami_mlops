from __future__ import annotations

import math
from collections import deque
from typing import Any

import cv2
import numpy as np


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

# 실시간 캡처는 항상 연속 프레임(frame_interval=1, R_live_clip/ATK_external_clip과
# 동일한 temporal density)이므로 시간정규화(_tn = raw / frame_interval)는 분모가
# 1이라 raw와 동일하다. 학습 코드의 일반 공식을 그대로 따르되 fi=1로 고정한다.
FRAME_INTERVAL = 1

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


def _build_seq_array(clip: list[dict[str, float]]) -> np.ndarray:
    """학습 코드의 build_seq_array()와 동일한 클립 단위 상대좌표 + 시간정규화 velocity.

    클립 평균 기준 상대좌표(nose_x_rel 등)와 frame_interval(=1, 실시간 연속 프레임)
    로 나눈 velocity(_tn)를 계산한다.
    """
    n = len(clip)
    fi = FRAME_INTERVAL

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

    def extract_from_frames(self, frames: list[np.ndarray]) -> tuple[np.ndarray, int, dict]:
        selected_frames = self._sample_frames(frames)
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
            seq = _build_seq_array(clip)
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
        }
        return x_seq, seq_length, info

    def _sample_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not frames:
            return []
        if len(frames) >= self.target_frames:
            indices = np.linspace(0, len(frames) - 1, self.target_frames).astype(int)
            return [frames[int(i)] for i in indices]

        padded = list(frames)
        while len(padded) < self.target_frames:
            padded.append(frames[-1])
        return padded

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
    def __init__(self, maxlen: int = 16):
        self.frames: deque[np.ndarray] = deque(maxlen=maxlen)

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def ready(self) -> bool:
        return len(self.frames) == self.frames.maxlen

    def as_list(self) -> list[np.ndarray]:
        return list(self.frames)
