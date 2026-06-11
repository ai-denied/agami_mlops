"""
얼굴 이미지 특징 추출 스크립트  (manifest 기반 클립 단위)

face_images 폴더의 이미지는 영상을 프레임으로 잘라 놓은 시계열 데이터이므로,
samples_manifest.jsonl의 clip 단위로 묶어서 temporal feature를 집계한다.
출력: features/face_clip_features.csv  — 1행 = 1클립 (영상 스크립트와 동일 스키마)

Features per clip (기획서 기준):
  blink_count         : EAR < 0.2 구간 진입 횟수
  eye_open_ratio      : EAR 평균
  mouth_open_ratio    : MAR 평균
  smile_ratio         : 입꼬리 너비 비율 std (입꼬리 변화량)
  head_yaw            : 좌우 회전 std
  head_pitch          : 상하 회전 std
  head_roll           : 기울기 std
  face_movement       : 얼굴 중심 누적 이동량
  face_stability      : 얼굴 중심 위치 std

메타:
  sample_id, subject_id, split, label, attack_type,
  frame_count, valid_frame_count, face_detect_rate,
  session, illumination, device

Usage:
  cd facial_recognition
  python preprocessing/extract_image_features.py
  python preprocessing/extract_image_features.py \\
    --manifest dataset/samples_manifest.jsonl \\
    --img-dir  dataset/face_images \\
    --out      features
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE            = [33, 160, 158, 133, 153, 144]
RIGHT_EYE           = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT_IDX      = 61
MOUTH_RIGHT_IDX     = 291
MOUTH_TOP_IDX       = 13
MOUTH_BOTTOM_IDX    = 14
NOSE_TIP_IDX        = 4
FACE_LEFT_IDX       = 234
FACE_RIGHT_IDX      = 454
EYE_LEFT_OUTER_IDX  = 33
EYE_RIGHT_OUTER_IDX = 263

BLINK_THRESH = 0.20


# ── 기본 계산 ─────────────────────────────────────────────────────────────────

def _dist(lm, i, j):
    a, b = lm[i], lm[j]
    return math.hypot(a.x - b.x, a.y - b.y)


def _ear(lm, eye):
    p1, p2, p3, p4, p5, p6 = eye
    num = _dist(lm, p2, p6) + _dist(lm, p3, p5)
    den = 2.0 * _dist(lm, p1, p4)
    return num / den if den else 0.0


def _mar(lm):
    vert  = _dist(lm, MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX)
    horiz = _dist(lm, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX)
    return vert / horiz if horiz else 0.0


def _smile_width_ratio(lm):
    mouth_w = _dist(lm, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX)
    face_w  = _dist(lm, FACE_LEFT_IDX, FACE_RIGHT_IDX)
    return mouth_w / face_w if face_w else 0.0


def _head_roll(lm):
    dx = lm[EYE_RIGHT_OUTER_IDX].x - lm[EYE_LEFT_OUTER_IDX].x
    dy = lm[EYE_RIGHT_OUTER_IDX].y - lm[EYE_LEFT_OUTER_IDX].y
    return math.degrees(math.atan2(dy, dx))


def _head_yaw(lm):
    face_cx = (lm[FACE_LEFT_IDX].x + lm[FACE_RIGHT_IDX].x) / 2.0
    half_w  = abs(lm[FACE_RIGHT_IDX].x - lm[FACE_LEFT_IDX].x) / 2.0 + 1e-6
    return (lm[NOSE_TIP_IDX].x - face_cx) / half_w


def _head_pitch(lm):
    eye_y   = (lm[EYE_LEFT_OUTER_IDX].y + lm[EYE_RIGHT_OUTER_IDX].y) / 2.0
    mouth_y = (lm[MOUTH_LEFT_IDX].y     + lm[MOUTH_RIGHT_IDX].y)     / 2.0
    mid_y   = (eye_y + mouth_y) / 2.0
    half_h  = abs(mouth_y - eye_y) / 2.0 + 1e-6
    return (lm[NOSE_TIP_IDX].y - mid_y) / half_h


# ── 프레임 단위 특징 ──────────────────────────────────────────────────────────

def extract_frame_features(lm) -> dict:
    ear_l = _ear(lm, LEFT_EYE)
    ear_r = _ear(lm, RIGHT_EYE)
    cx = (lm[33].x + lm[133].x + lm[362].x + lm[263].x) / 4.0
    cy = (lm[33].y + lm[133].y + lm[362].y + lm[263].y) / 4.0
    return {
        "ear":     (ear_l + ear_r) / 2.0,
        "mar":     _mar(lm),
        "smile_w": _smile_width_ratio(lm),
        "nose_x":  lm[NOSE_TIP_IDX].x,
        "nose_y":  lm[NOSE_TIP_IDX].y,
        "cx":      cx,
        "cy":      cy,
        "roll":    _head_roll(lm),
        "yaw":     _head_yaw(lm),
        "pitch":   _head_pitch(lm),
    }


# ── 클립 단위 집계 ─────────────────────────────────────────────────────────────

def count_blinks(ear_series: list) -> int:
    below = [e < BLINK_THRESH for e in ear_series]
    return sum(1 for i in range(1, len(below)) if below[i] and not below[i - 1])


def aggregate(frames: list) -> dict:
    ears     = [f["ear"]     for f in frames]
    mars     = [f["mar"]     for f in frames]
    smile_ws = [f["smile_w"] for f in frames]
    nose_x   = [f["nose_x"]  for f in frames]
    nose_y   = [f["nose_y"]  for f in frames]
    cx       = [f["cx"]      for f in frames]
    cy       = [f["cy"]      for f in frames]
    rolls    = [f["roll"]    for f in frames]
    yaws     = [f["yaw"]     for f in frames]
    pitches  = [f["pitch"]   for f in frames]

    n = len(frames)

    nose_movement = sum(
        math.hypot(nose_x[i] - nose_x[i-1], nose_y[i] - nose_y[i-1])
        for i in range(1, n)
    )
    face_movement = sum(
        math.hypot(cx[i] - cx[i-1], cy[i] - cy[i-1])
        for i in range(1, n)
    )
    std_cx = float(np.std(cx)) if n > 1 else 0.0
    std_cy = float(np.std(cy)) if n > 1 else 0.0

    def _std(a):  return float(np.std(a))  if len(a) > 1 else 0.0
    def _mean(a): return float(np.mean(a))

    return {
        # 기획서 feature 이름
        "blink_count":      count_blinks(ears),
        "eye_open_ratio":   _mean(ears),
        "mouth_open_ratio": _mean(mars),
        "smile_ratio":      _std(smile_ws),
        "head_yaw":         _std(yaws),
        "head_pitch":       _std(pitches),
        "head_roll":        _std(rolls),
        "face_movement":    face_movement,
        "face_stability":   math.hypot(std_cx, std_cy),
        # 보조
        "ear_std":          _std(ears),
        "mar_mean":         _mean(mars),
        "mar_std":          _std(mars),
        "nose_movement":    nose_movement,
        "head_yaw_mean":    _mean(yaws),
        "head_pitch_mean":  _mean(pitches),
        "head_roll_mean":   _mean(rolls),
    }


# ── 클립 처리 ─────────────────────────────────────────────────────────────────

def process_clip(sample: dict, img_base: Path, face_mesh, seq_length: int = 16) -> tuple:
    """
    manifest sample 1개 처리.
    Returns (aggregated_dict | None, frame_seq_array | None, total_frames, valid_count)
    """
    valid_frames = []

    for rel_path in sample["frames"]:
        img = cv2.imread(str(img_base / rel_path))
        if img is None:
            continue
        result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            valid_frames.append(extract_frame_features(lm))

    total = len(sample["frames"])
    valid = len(valid_frames)

    if not valid_frames:
        return None, None, total, 0

    # frame_seq: (seq_length, 20) = 절대값 10개 + velocity 10개
    frame_seq = np.zeros((seq_length, 20), dtype=np.float32)

    for i, frame in enumerate(valid_frames[:seq_length]):
        # 절대값 10개
        abs_features = np.array([
            frame["ear"], frame["mar"], frame["smile_w"],
            frame["nose_x"], frame["nose_y"],
            frame["cx"], frame["cy"],
            frame["roll"], frame["yaw"], frame["pitch"]
        ], dtype=np.float32)

        # velocity 10개 (첫 프레임은 0)
        if i == 0:
            vel_features = np.zeros(10, dtype=np.float32)
        else:
            prev_frame = valid_frames[i - 1]
            nose_dx = frame["nose_x"] - prev_frame["nose_x"]
            nose_dy = frame["nose_y"] - prev_frame["nose_y"]
            center_dx = frame["cx"] - prev_frame["cx"]
            center_dy = frame["cy"] - prev_frame["cy"]
            nose_speed = math.sqrt(nose_dx**2 + nose_dy**2)

            vel_features = np.array([
                nose_dx, nose_dy,
                center_dx, center_dy,
                nose_speed,
                frame["ear"] - prev_frame["ear"],
                frame["mar"] - prev_frame["mar"],
                frame["yaw"] - prev_frame["yaw"],
                frame["pitch"] - prev_frame["pitch"],
                frame["roll"] - prev_frame["roll"]
            ], dtype=np.float32)

        frame_seq[i] = np.concatenate([abs_features, vel_features])

    return aggregate(valid_frames), frame_seq, total, valid


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(manifest_path: Path, img_base: Path, out_dir: Path, seq_length: int = 16):
    out_dir.mkdir(exist_ok=True)

    with open(manifest_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"총 클립: {len(samples)}개  (시퀀스 길이: {seq_length}프레임)")

    rows, skipped = [], []
    x_seqs, x_statics, ys = [], [], []
    sample_ids, subject_ids, splits, attack_types = [], [], [], []
    valid_frame_counts, seq_lengths, face_detect_rates = [], [], []
    sessions, illuminations, devices = [], [], []

    agg_feature_names = [
        "blink_count", "eye_open_ratio", "mouth_open_ratio", "smile_ratio",
        "head_yaw", "head_pitch", "head_roll", "face_movement", "face_stability",
        "ear_std", "mar_mean", "mar_std", "nose_movement",
        "head_yaw_mean", "head_pitch_mean", "head_roll_mean"
    ]

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,   # 프레임별 독립 처리
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        for sample in tqdm(samples, desc="클립 특징 추출"):
            agg, frame_seq, total, valid = process_clip(sample, img_base, face_mesh, seq_length)

            if agg is None:
                skipped.append({
                    "sample_id":  sample["sample_id"],
                    "label":      sample["label"],
                    "reason":     "no_face_detected",
                    "frame_count": total,
                })
                continue

            x_seqs.append(frame_seq)
            x_static = np.array([agg[name] for name in agg_feature_names], dtype=np.float32)
            x_statics.append(x_static)
            ys.append(sample["label"])

            sample_ids.append(sample["sample_id"])
            subject_ids.append(sample["subject_id"])
            splits.append(sample["split"])
            attack_types.append(sample["attack_type"])
            sessions.append(sample.get("session"))
            illuminations.append(sample.get("illumination"))
            devices.append(sample.get("device"))
            valid_frame_counts.append(valid)
            seq_lengths.append(min(valid, seq_length))
            face_detect_rates.append(round(valid / total, 4) if total else 0.0)

            row = {
                # 메타
                "sample_id":         sample["sample_id"],
                "subject_id":        sample["subject_id"],
                "split":             sample["split"],
                "label":             sample["label"],
                "attack_type":       sample["attack_type"],
                "session":           sample.get("session"),
                "illumination":      sample.get("illumination"),
                "device":            sample.get("device"),
                # 특징
                **agg,
                # 품질
                "frame_count":       total,
                "valid_frame_count": valid,
                "face_detect_rate":  round(valid / total, 4) if total else 0.0,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "face_clip_features.csv"
    df.to_csv(csv_path, index=False)

    if x_seqs:
        npz_data = {
            "x_seq": np.array(x_seqs, dtype=np.float32),
            "x_static": np.array(x_statics, dtype=np.float32),
            "y": np.array(ys, dtype=np.int64),
            "sample_ids": np.array(sample_ids, dtype=object),
            "subject_ids": np.array(subject_ids, dtype=object),
            "splits": np.array(splits, dtype=object),
            "attack_types": np.array(attack_types, dtype=object),
            "sessions": np.array(sessions, dtype=object),
            "illuminations": np.array(illuminations, dtype=object),
            "devices": np.array(devices, dtype=object),
            "valid_frame_counts": np.array(valid_frame_counts, dtype=np.int32),  # 원본 프레임 개수
            "seq_lengths": np.array(seq_lengths, dtype=np.int32),                # x_seq의 실제 유효 프레임 개수
            "face_detect_rates": np.array(face_detect_rates, dtype=np.float32),
            "agg_feature_names": np.array(agg_feature_names, dtype=object),
            "seq_feature_names": np.array([
                "ear", "mar", "smile_w", "nose_x", "nose_y", "cx", "cy", "roll", "yaw", "pitch",
                "nose_dx", "nose_dy", "center_dx", "center_dy", "nose_speed", "ear_velocity", "mar_velocity", "yaw_velocity", "pitch_velocity", "roll_velocity"
            ], dtype=object),
        }
        npz_path = out_dir / "face_clip_data.npz"
        np.savez_compressed(npz_path, **npz_data)

    skip_df = pd.DataFrame(skipped)
    skip_df.to_csv(out_dir / "skipped_clips.csv", index=False)

    total_clips = len(samples)
    success     = len(rows)
    fail        = len(skipped)
    print(f"\n=== 클립 특징 추출 완료 ===")
    print(f"  전체: {total_clips}  |  성공: {success}  |  실패: {fail}  |  성공률: {success/total_clips*100:.1f}%")
    print(f"  CSV 저장: {csv_path}")
    if x_seqs:
        print(f"  NPZ 저장: {npz_path}")
        print(f"    x_seq shape: {np.array(x_seqs).shape}")
        print(f"    x_static shape: {np.array(x_statics).shape}")

    if success > 0:
        n0 = (df["label"] == 0).sum()
        n1 = (df["label"] == 1).sum()
        print(f"  레이블: real={n0}, spoof={n1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="dataset/samples_manifest.jsonl")
    parser.add_argument("--img-dir",  default="dataset/face_images")
    parser.add_argument("--out",      default="features")
    parser.add_argument("--seq-length", type=int, default=16)
    args = parser.parse_args()
    main(Path(args.manifest), Path(args.img_dir), Path(args.out), args.seq_length)