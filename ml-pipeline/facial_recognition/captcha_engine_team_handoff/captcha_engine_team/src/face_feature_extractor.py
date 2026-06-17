from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np


SELECTED_FEATURES = [
    "ear",
    "mar",
    "smile_w",
    "nose_x",
    "nose_y",
    "cx",
    "cy",
    "roll",
    "yaw",
    "pitch",
    "nose_dx",
    "nose_dy",
    "center_dx",
    "center_dy",
    "nose_speed",
    "ear_velocity",
    "mar_velocity",
    "yaw_velocity",
    "pitch_velocity",
    "roll_velocity",
]


class FaceFeatureExtractor:
    """MediaPipe FaceMesh based demo feature extractor.

    The exact training-time feature implementation is not available in this
    runtime module. This class prioritizes demo stability while preserving the
    model-required feature order and shape: (target_frames, 20).
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
        x_seq = np.zeros((self.target_frames, len(SELECTED_FEATURES)), dtype=np.float32)
        valid_count = 0
        face_detected_frames = 0
        last_feature: dict[str, float] | None = None
        prev_feature: dict[str, float] | None = None

        selected_frames = self._sample_frames(frames)
        for idx, frame in enumerate(selected_frames):
            feature = self._extract_one(frame, prev_feature)
            if feature is None:
                feature = last_feature or self._zero_feature()
            else:
                face_detected_frames += 1
                valid_count += 1
                last_feature = feature

            x_seq[idx] = np.asarray([feature[name] for name in SELECTED_FEATURES], dtype=np.float32)
            prev_feature = feature

        seq_length = max(1, min(self.target_frames, len(selected_frames), valid_count or len(selected_frames)))
        info = {
            "target_frames": self.target_frames,
            "input_frames": len(frames),
            "used_frames": len(selected_frames),
            "valid_frames": valid_count,
            "face_detected_frames": face_detected_frames,
            "face_detected": face_detected_frames > 0,
            "face_detect_rate": face_detected_frames / max(1, len(selected_frames)),
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

    def _extract_one(
        self,
        frame_bgr: np.ndarray,
        prev_feature: dict[str, float] | None,
    ) -> dict[str, float] | None:
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

        left_eye = self._eye_aspect_ratio(points, [33, 160, 158, 133, 153, 144])
        right_eye = self._eye_aspect_ratio(points, [362, 385, 387, 263, 373, 380])
        ear = float((left_eye + right_eye) / 2.0)
        mar = float(self._distance(points[13], points[14]) / max(self._distance(points[61], points[291]), 1e-6))
        smile_w = float(self._distance(points[61], points[291]))

        nose = points[1]
        center = points[:, :2].mean(axis=0)
        roll = float(np.degrees(np.arctan2(points[263][1] - points[33][1], points[263][0] - points[33][0])))
        yaw = float((points[1][0] - center[0]) * 100.0)
        pitch = float((points[1][1] - center[1]) * 100.0)

        feature = {
            "ear": ear,
            "mar": mar,
            "smile_w": smile_w,
            "nose_x": float(nose[0]),
            "nose_y": float(nose[1]),
            "cx": float(center[0]),
            "cy": float(center[1]),
            "roll": roll,
            "yaw": yaw,
            "pitch": pitch,
            "nose_dx": 0.0,
            "nose_dy": 0.0,
            "center_dx": 0.0,
            "center_dy": 0.0,
            "nose_speed": 0.0,
            "ear_velocity": 0.0,
            "mar_velocity": 0.0,
            "yaw_velocity": 0.0,
            "pitch_velocity": 0.0,
            "roll_velocity": 0.0,
        }

        if prev_feature is not None:
            feature["nose_dx"] = feature["nose_x"] - prev_feature["nose_x"]
            feature["nose_dy"] = feature["nose_y"] - prev_feature["nose_y"]
            feature["center_dx"] = feature["cx"] - prev_feature["cx"]
            feature["center_dy"] = feature["cy"] - prev_feature["cy"]
            feature["nose_speed"] = float(np.hypot(feature["nose_dx"], feature["nose_dy"]))
            feature["ear_velocity"] = feature["ear"] - prev_feature["ear"]
            feature["mar_velocity"] = feature["mar"] - prev_feature["mar"]
            feature["yaw_velocity"] = feature["yaw"] - prev_feature["yaw"]
            feature["pitch_velocity"] = feature["pitch"] - prev_feature["pitch"]
            feature["roll_velocity"] = feature["roll"] - prev_feature["roll"]

        return feature

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _eye_aspect_ratio(self, points: np.ndarray, idx: list[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [points[i] for i in idx]
        vertical = self._distance(p2, p6) + self._distance(p3, p5)
        horizontal = 2.0 * max(self._distance(p1, p4), 1e-6)
        return float(vertical / horizontal)

    @staticmethod
    def _zero_feature() -> dict[str, float]:
        return {name: 0.0 for name in SELECTED_FEATURES}


class FrameBuffer:
    def __init__(self, maxlen: int = 16):
        self.frames: deque[np.ndarray] = deque(maxlen=maxlen)

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def ready(self) -> bool:
        return len(self.frames) == self.frames.maxlen

    def as_list(self) -> list[np.ndarray]:
        return list(self.frames)
