from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import cv2
import numpy as np


SupportedMission = Literal["index_up", "two_fingers", "open_palm", "fist"]
SUPPORTED_MISSIONS: tuple[str, ...] = ("index_up", "two_fingers", "open_palm", "fist")


@dataclass
class HandGestureResult:
    mission: str
    mission_pass: bool
    detected: bool
    gesture: str
    confidence: float
    finger_states: dict[str, bool]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class HandGestureDetector:
    """OpenCV webcam frame + MediaPipe Hands rule-based mission detector."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
    ):
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray, mission: str) -> dict:
        if mission not in SUPPORTED_MISSIONS:
            return HandGestureResult(
                mission=mission,
                mission_pass=False,
                detected=False,
                gesture="unknown",
                confidence=0.0,
                finger_states={},
                reason=f"Unsupported mission: {mission}",
            ).to_dict()

        try:
            result = self._detect_result(frame_bgr, mission)
        except Exception as exc:  # Demo should fail closed rather than crash.
            result = HandGestureResult(
                mission=mission,
                mission_pass=False,
                detected=False,
                gesture="error",
                confidence=0.0,
                finger_states={},
                reason=f"Hand detection error: {exc}",
            )
        return result.to_dict()

    def _detect_result(self, frame_bgr: np.ndarray, mission: str) -> HandGestureResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return HandGestureResult(mission, False, False, "none", 0.0, {}, "Empty frame")

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self._hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return HandGestureResult(mission, False, False, "none", 0.0, {}, "No hand detected")

        landmarks = results.multi_hand_landmarks[0].landmark
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        states = self._finger_states(landmarks, handedness)
        gesture = self._classify(states)
        mission_pass = gesture == mission
        confidence = self._confidence(states, mission)
        reason = "Mission matched" if mission_pass else f"Detected {gesture}, expected {mission}"

        return HandGestureResult(
            mission=mission,
            mission_pass=mission_pass,
            detected=True,
            gesture=gesture,
            confidence=confidence,
            finger_states=states,
            reason=reason,
        )

    def _finger_states(self, landmarks, handedness: str) -> dict[str, bool]:
        # MediaPipe normalized coordinates: y decreases as a finger goes upward.
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]

        if handedness == "Right":
            thumb_up = thumb_tip.x < thumb_ip.x
        else:
            thumb_up = thumb_tip.x > thumb_ip.x

        return {
            "thumb": bool(thumb_up),
            "index": bool(index_tip.y < index_pip.y),
            "middle": bool(middle_tip.y < middle_pip.y),
            "ring": bool(ring_tip.y < ring_pip.y),
            "pinky": bool(pinky_tip.y < pinky_pip.y),
        }

    @staticmethod
    def _classify(states: dict[str, bool]) -> str:
        extended = {name for name, up in states.items() if up}
        non_thumb = {name for name in extended if name != "thumb"}

        if len(extended) >= 4:
            return "open_palm"
        if non_thumb == {"index", "middle"}:
            return "two_fingers"
        if non_thumb == {"index"}:
            return "index_up"
        if len(extended) == 0 or extended == {"thumb"}:
            return "fist"
        return "unknown"

    @staticmethod
    def _confidence(states: dict[str, bool], mission: str) -> float:
        expected = {
            "index_up": {"index": True, "middle": False, "ring": False, "pinky": False},
            "two_fingers": {"index": True, "middle": True, "ring": False, "pinky": False},
            "open_palm": {"index": True, "middle": True, "ring": True, "pinky": True},
            "fist": {"index": False, "middle": False, "ring": False, "pinky": False},
        }.get(mission, {})
        if not expected:
            return 0.0
        matches = sum(states.get(name) == value for name, value in expected.items())
        return float(matches / len(expected))
