from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.captcha_decision import MissionRound, decide_three_round_captcha
from src.face_feature_extractor import FaceFeatureExtractor, FrameBuffer
from src.face_liveness_predictor import FaceLivenessPredictor
from src.hand_gesture_detector import HandGestureDetector, SUPPORTED_MISSIONS


FACE_MODEL_PATH = "runs/gru_h32_lr0005_v1/best_gru.pt"
ROUND_SECONDS = 5.0
HAND_REQUIRED_HITS = 8


def build_round_plan() -> list[tuple[str, str]]:
    hand_mission = random.choice(SUPPORTED_MISSIONS)
    plan = [("face", "face_live"), ("hand", hand_mission)]
    third_type = random.choice(["face", "hand"])
    third_name = "face_live" if third_type == "face" else random.choice(SUPPORTED_MISSIONS)
    plan.append((third_type, third_name))
    random.shuffle(plan)
    return plan


def draw_status(frame, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30


def run_round(
    cap,
    round_id: int,
    mission_type: str,
    mission_name: str,
    predictor: FaceLivenessPredictor,
    face_extractor: FaceFeatureExtractor,
    hand_detector: HandGestureDetector,
) -> MissionRound:
    frame_buffer = FrameBuffer(maxlen=16)
    start = time.monotonic()
    hand_hits = 0
    hand_seen = False
    last_hand = {"gesture": "none", "mission_pass": False, "detected": False, "confidence": 0.0}
    show_window = True

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            return MissionRound(
                round_id=round_id,
                mission_type=mission_type,  # type: ignore[arg-type]
                mission_name=mission_name,
                spoof_score=1.0,
                mission_pass=False,
                face_detected=False,
                timeout=True,
                detail="Cannot read webcam frame.",
            )

        elapsed = time.monotonic() - start
        remaining = max(0.0, ROUND_SECONDS - elapsed)
        frame_buffer.append(frame)

        if mission_type == "hand":
            last_hand = hand_detector.detect(frame, mission_name)
            hand_seen = hand_seen or bool(last_hand.get("detected"))
            if last_hand.get("mission_pass"):
                hand_hits += 1

        draw_status(
            frame,
            [
                f"Round {round_id}/3 | {mission_type}: {mission_name}",
                f"Time left: {remaining:0.1f}s",
                f"Hand: {last_hand.get('gesture', 'none')} ({hand_hits}/{HAND_REQUIRED_HITS})",
                "Press q to abort",
            ],
        )

        if show_window:
            try:
                cv2.imshow("AGAMI full CAPTCHA demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            except cv2.error:
                show_window = False
                print("OpenCV GUI window is not available. Continuing without imshow.")

        if elapsed >= ROUND_SECONDS and frame_buffer.ready():
            break

    try:
        x_seq, seq_length, face_info = face_extractor.extract_from_frames(frame_buffer.as_list())
        pred = predictor.predict_dict(x_seq, seq_length=seq_length)
        spoof_score = float(pred["spoof_score"])
        threshold = float(pred["threshold"])
        face_detected = bool(face_info.get("face_detected"))
    except Exception as exc:
        spoof_score = 1.0
        threshold = getattr(predictor, "threshold", 0.5)
        face_detected = False
        face_info = {"error": str(exc)}

    if mission_type == "face":
        # TODO: Replace this presentation-friendly placeholder with blink,
        # head-turn, or prompted motion missions when those labels/features exist.
        mission_pass = bool(face_detected and spoof_score <= threshold)
        detail = f"face_detected={face_detected}, spoof_score={spoof_score:.3f}, threshold={threshold:.3f}"
    else:
        mission_pass = hand_hits >= HAND_REQUIRED_HITS
        detail = (
            f"hand_detected={hand_seen}, hand_hits={hand_hits}, "
            f"gesture={last_hand.get('gesture')}, spoof_score={spoof_score:.3f}"
        )

    return MissionRound(
        round_id=round_id,
        mission_type=mission_type,  # type: ignore[arg-type]
        mission_name=mission_name,
        spoof_score=spoof_score,
        mission_pass=mission_pass,
        face_detected=face_detected,
        timeout=False,
        hand_detected=hand_seen,
        detail=detail,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-path", default=FACE_MODEL_PATH)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            "Cannot open webcam. If running on remote server, run this demo locally "
            "or connect webcam frames from frontend."
        )
        return 2

    predictor = FaceLivenessPredictor(model_path=str(model_path), device=args.device)
    face_extractor = FaceFeatureExtractor(target_frames=16)
    hand_detector = HandGestureDetector()

    rounds: list[MissionRound] = []
    try:
        for round_id, (mission_type, mission_name) in enumerate(build_round_plan(), start=1):
            print(f"Round {round_id}: {mission_type} / {mission_name}")
            result = run_round(
                cap,
                round_id,
                mission_type,
                mission_name,
                predictor,
                face_extractor,
                hand_detector,
            )
            rounds.append(result)
            print(result)
            time.sleep(0.7)

        final = decide_three_round_captcha(rounds)
        print("\nFinal decision:", final.decision)
        print("Reason:", final.reason)
        print("Total risk:", f"{final.total_risk:.3f}")

        end = time.monotonic() + 4.0
        while time.monotonic() < end:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            draw_status(
                frame,
                [
                    f"FINAL: {final.decision}",
                    f"Risk: {final.total_risk:.3f}",
                    final.reason,
                ],
            )
            try:
                cv2.imshow("AGAMI full CAPTCHA demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                break

        return 0
    except KeyboardInterrupt:
        print("Demo aborted by user.")
        return 130
    finally:
        cap.release()
        face_extractor.close()
        hand_detector.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
