"""
얼굴 Anti-Spoofing 전처리 v2 — face_clip_data_rel.npz 생성

v1(extract_image_features.py) 대비 변경사항:
  1. landmark 좌표를 face_w 기준 bbox 정규화 → 카메라 거리·얼굴 크기 불변
  2. nose_x/y, cx/cy → 클립 평균 기준 상대좌표 (nose_x_rel 등)
     "얼굴이 화면 어디 있냐"가 아니라 "클립 안에서 얼마나 움직였냐"
  3. velocity는 정규화된 상대좌표 기준 계산 (mean 소거로 velocity 값은 동일)
  4. face_movement / nose_movement → 프레임 수로 정규화 (클립 길이 편차 제거)
  5. 얼굴 검출 실패 프레임 → 이웃 프레임으로 보간 후 사용
  6. 유효 프레임 < 3개 클립 제거
  7. train set 1~99% percentile clipping
  8. FaceMesh video mode (static_image_mode=False): 클립 내 프레임 간 tracking

출력 구조:
  face_clip_data_rel.npz
    x_seq:    float32 (N, 16, 20)   — GRU 입력
    x_static: float32 (N, 16)       — SVM/RF/ET 비교용
    y:        int64   (N,)
    seq_feature_names / static_feature_names
    sample_ids, subject_ids, splits, source_groups,
    attack_types, devices, seq_lengths, face_detect_rates

Usage (ml-pipeline/facial_recognition/ 기준):
  python preprocessing/extract_features_rel.py
  python preprocessing/extract_features_rel.py \\
    --manifest ../../data/facial_recognition/samples_manifest.jsonl \\
    --img-dir  ../../data/facial_recognition/face_images \\
    --out      features/face_clip_data_rel.npz
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

# ── 상수 ──────────────────────────────────────────────────────────────────────

SEQ_LENGTH       = 16
MIN_VALID_FRAMES = 3
BLINK_THRESH     = 0.20

SEQ_FEATURE_NAMES = [
    "ear", "mar", "smile_w",
    "nose_x_rel", "nose_y_rel", "cx_rel", "cy_rel",
    "roll", "yaw", "pitch",
    "nose_dx", "nose_dy", "center_dx", "center_dy", "nose_speed",
    "ear_velocity", "mar_velocity", "yaw_velocity", "pitch_velocity", "roll_velocity",
]

STATIC_FEATURE_NAMES = [
    "blink_count", "eye_open_ratio", "mouth_open_ratio", "smile_ratio",
    "head_yaw", "head_pitch", "head_roll", "face_movement", "face_stability",
    "ear_std", "mar_mean", "mar_std", "nose_movement",
    "head_yaw_mean", "head_pitch_mean", "head_roll_mean",
]

# ── Landmark 인덱스 ───────────────────────────────────────────────────────────

LEFT_EYE             = [33, 160, 158, 133, 153, 144]
RIGHT_EYE            = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT_IDX       = 61
MOUTH_RIGHT_IDX      = 291
MOUTH_TOP_IDX        = 13
MOUTH_BOTTOM_IDX     = 14
NOSE_TIP_IDX         = 4
FACE_LEFT_IDX        = 234
FACE_RIGHT_IDX       = 454
EYE_LEFT_OUTER_IDX   = 33
EYE_RIGHT_OUTER_IDX  = 263

# ── 유틸 ──────────────────────────────────────────────────────────────────────

def infer_source_group(sample_id: str) -> str:
    s = str(sample_id)
    if s.startswith("ATK"):
        return "ATK_external_clip"
    if s.startswith("R"):
        return "R_live_clip"
    if s.startswith("S"):
        return "S_dataset_sequence"
    return "unknown"


def _dist(lm, i: int, j: int) -> float:
    a, b = lm[i], lm[j]
    return math.hypot(a.x - b.x, a.y - b.y)


def _ear(lm, eye: list) -> float:
    p1, p2, p3, p4, p5, p6 = eye
    num = _dist(lm, p2, p6) + _dist(lm, p3, p5)
    den = 2.0 * _dist(lm, p1, p4)
    return num / den if den else 0.0


def _mar(lm) -> float:
    vert  = _dist(lm, MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX)
    horiz = _dist(lm, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX)
    return vert / horiz if horiz else 0.0


def _head_roll(lm) -> float:
    dx = lm[EYE_RIGHT_OUTER_IDX].x - lm[EYE_LEFT_OUTER_IDX].x
    dy = lm[EYE_RIGHT_OUTER_IDX].y - lm[EYE_LEFT_OUTER_IDX].y
    return math.degrees(math.atan2(dy, dx))


def _head_yaw(lm) -> float:
    face_cx = (lm[FACE_LEFT_IDX].x + lm[FACE_RIGHT_IDX].x) / 2.0
    half_w  = abs(lm[FACE_RIGHT_IDX].x - lm[FACE_LEFT_IDX].x) / 2.0 + 1e-6
    return (lm[NOSE_TIP_IDX].x - face_cx) / half_w


def _head_pitch(lm) -> float:
    eye_y   = (lm[EYE_LEFT_OUTER_IDX].y + lm[EYE_RIGHT_OUTER_IDX].y) / 2.0
    mouth_y = (lm[MOUTH_LEFT_IDX].y     + lm[MOUTH_RIGHT_IDX].y)     / 2.0
    mid_y   = (eye_y + mouth_y) / 2.0
    half_h  = abs(mouth_y - eye_y) / 2.0 + 1e-6
    return (lm[NOSE_TIP_IDX].y - mid_y) / half_h


# ── 프레임 단위 피처 ──────────────────────────────────────────────────────────

def extract_frame_raw(lm) -> dict:
    """
    프레임 단위 피처 추출.
    nose_x/y, cx/cy는 face_w 기준 bbox 정규화 좌표.
      x_norm = (x_raw - face_cx) / face_w  → 얼굴 폭 단위, 얼굴 중심 기준
    EAR/MAR/smile_w/yaw/pitch/roll은 이미 스케일 불변이므로 그대로.
    """
    face_w  = _dist(lm, FACE_LEFT_IDX, FACE_RIGHT_IDX) + 1e-6
    face_cx = (lm[FACE_LEFT_IDX].x + lm[FACE_RIGHT_IDX].x) / 2.0
    face_cy = (lm[FACE_LEFT_IDX].y + lm[FACE_RIGHT_IDX].y) / 2.0

    ear_l = _ear(lm, LEFT_EYE)
    ear_r = _ear(lm, RIGHT_EYE)

    # 얼굴 중심 (눈 4코너 평균) — bbox 정규화
    cx_raw = (lm[33].x + lm[133].x + lm[362].x + lm[263].x) / 4.0
    cy_raw = (lm[33].y + lm[133].y + lm[362].y + lm[263].y) / 4.0

    # smile_w = mouth_w / face_w 는 이미 스케일 불변
    smile_w = _dist(lm, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX) / face_w

    return {
        "ear":    (ear_l + ear_r) / 2.0,
        "mar":    _mar(lm),
        "smile_w": smile_w,
        "nose_x": (lm[NOSE_TIP_IDX].x - face_cx) / face_w,
        "nose_y": (lm[NOSE_TIP_IDX].y - face_cy) / face_w,
        "cx":     (cx_raw - face_cx) / face_w,
        "cy":     (cy_raw - face_cy) / face_w,
        "roll":   _head_roll(lm),
        "yaw":    _head_yaw(lm),
        "pitch":  _head_pitch(lm),
    }


# ── 보간 ──────────────────────────────────────────────────────────────────────

def interpolate_frames(frames: list) -> list:
    """
    None 위치를 nearest-neighbor → forward/backward fill 순서로 채움.
    (완전 선형 보간은 dict 필드 수가 많아 nearest fill로 대체)
    """
    result = list(frames)
    n = len(result)

    # forward fill
    last = None
    for i in range(n):
        if result[i] is not None:
            last = result[i]
        elif last is not None:
            result[i] = last

    # backward fill (앞부분 None 처리)
    last = None
    for i in range(n - 1, -1, -1):
        if result[i] is not None:
            last = result[i]
        elif last is not None:
            result[i] = last

    return result


# ── 클립 단위 시퀀스 배열 ─────────────────────────────────────────────────────

def build_seq_array(frames: list, seq_length: int) -> tuple[np.ndarray, int]:
    """
    (seq_length, 20) float32 배열 생성.
    위치 좌표: 클립 평균 기준 상대좌표 변환.
    velocity: 정규화 좌표 차분 (mean 소거로 절대 차분과 동일).
    Returns: (frame_seq, actual_len)
    """
    use_n = min(len(frames), seq_length)
    clip  = frames[:use_n]

    # 클립 평균 (bbox 정규화 좌표 기준)
    mean_nose_x = float(np.mean([f["nose_x"] for f in clip]))
    mean_nose_y = float(np.mean([f["nose_y"] for f in clip]))
    mean_cx     = float(np.mean([f["cx"]     for f in clip]))
    mean_cy     = float(np.mean([f["cy"]     for f in clip]))

    frame_seq = np.zeros((seq_length, 20), dtype=np.float32)

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

            vel_feats = np.array([
                nose_dx, nose_dy,
                center_dx, center_dy,
                math.hypot(nose_dx, nose_dy),
                frame["ear"]   - prev["ear"],
                frame["mar"]   - prev["mar"],
                frame["yaw"]   - prev["yaw"],
                frame["pitch"] - prev["pitch"],
                frame["roll"]  - prev["roll"],
            ], dtype=np.float32)

        frame_seq[i] = np.concatenate([abs_feats, vel_feats])

    return frame_seq, use_n


# ── 클립 단위 static 집계 ─────────────────────────────────────────────────────

def count_blinks(ear_series: list) -> int:
    below = [e < BLINK_THRESH for e in ear_series]
    return sum(1 for i in range(1, len(below)) if below[i] and not below[i - 1])


def aggregate_static(frames: list) -> dict:
    """
    16개 static 피처.
    face_movement / nose_movement: bbox 정규화 좌표 기준 + 프레임 수 정규화.
    """
    ears    = [f["ear"]    for f in frames]
    mars    = [f["mar"]    for f in frames]
    smiles  = [f["smile_w"] for f in frames]
    nose_x  = [f["nose_x"] for f in frames]
    nose_y  = [f["nose_y"] for f in frames]
    cx      = [f["cx"]     for f in frames]
    cy      = [f["cy"]     for f in frames]
    yaws    = [f["yaw"]    for f in frames]
    pitches = [f["pitch"]  for f in frames]
    rolls   = [f["roll"]   for f in frames]

    n = len(frames)
    denom = max(n - 1, 1)

    nose_movement = sum(
        math.hypot(nose_x[i] - nose_x[i-1], nose_y[i] - nose_y[i-1])
        for i in range(1, n)
    ) / denom

    face_movement = sum(
        math.hypot(cx[i] - cx[i-1], cy[i] - cy[i-1])
        for i in range(1, n)
    ) / denom

    def _std(a):  return float(np.std(a))  if len(a) > 1 else 0.0
    def _mean(a): return float(np.mean(a))

    std_cx = float(np.std(cx)) if n > 1 else 0.0
    std_cy = float(np.std(cy)) if n > 1 else 0.0

    return {
        "blink_count":      count_blinks(ears),
        "eye_open_ratio":   _mean(ears),
        "mouth_open_ratio": _mean(mars),
        "smile_ratio":      _std(smiles),
        "head_yaw":         _std(yaws),
        "head_pitch":       _std(pitches),
        "head_roll":        _std(rolls),
        "face_movement":    face_movement,
        "face_stability":   math.hypot(std_cx, std_cy),
        "ear_std":          _std(ears),
        "mar_mean":         _mean(mars),
        "mar_std":          _std(mars),
        "nose_movement":    nose_movement,
        "head_yaw_mean":    _mean(yaws),
        "head_pitch_mean":  _mean(pitches),
        "head_roll_mean":   _mean(rolls),
    }


# ── 클립 처리 ─────────────────────────────────────────────────────────────────

def process_clip(
    sample: dict, img_base: Path, face_mesh, seq_length: int
) -> tuple:
    """
    manifest sample 1개 처리.
    Returns (static_dict, frame_seq, total_frames, valid_count, actual_seq_len)
    실패 시 static_dict = None.
    """
    raw_frames = []
    for rel_path in sample["frames"]:
        img = cv2.imread(str(img_base / rel_path))
        if img is None:
            raw_frames.append(None)
            continue
        result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            raw_frames.append(extract_frame_raw(result.multi_face_landmarks[0].landmark))
        else:
            raw_frames.append(None)

    total       = len(raw_frames)
    valid_count = sum(1 for f in raw_frames if f is not None)

    if valid_count < MIN_VALID_FRAMES:
        return None, None, total, valid_count, 0

    filled = interpolate_frames(raw_frames)
    frame_seq, actual_len = build_seq_array(filled, seq_length)
    static = aggregate_static(filled[:actual_len])

    return static, frame_seq, total, valid_count, actual_len


# ── Clipping ──────────────────────────────────────────────────────────────────

def apply_clipping(
    x_seq: np.ndarray,
    seq_lengths: np.ndarray,
    x_static: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """train set 기준 1~99% percentile clipping."""
    # seq: train set의 유효 프레임 수집
    seq_frames = []
    for i in train_idx:
        l = int(seq_lengths[i])
        if l > 0:
            seq_frames.append(x_seq[i, :l, :])
    seq_all = np.concatenate(seq_frames, axis=0)  # (total_frames, 20)
    seq_lo = np.percentile(seq_all, 1, axis=0)
    seq_hi = np.percentile(seq_all, 99, axis=0)

    static_train = x_static[train_idx]
    static_lo = np.percentile(static_train, 1, axis=0)
    static_hi = np.percentile(static_train, 99, axis=0)

    x_seq_out = x_seq.copy()
    for i in range(len(x_seq)):
        l = int(seq_lengths[i])
        if l > 0:
            x_seq_out[i, :l, :] = np.clip(x_seq[i, :l, :], seq_lo, seq_hi)

    return x_seq_out, np.clip(x_static, static_lo, static_hi)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(manifest_path: Path, img_base: Path, out_path: Path, seq_length: int = SEQ_LENGTH):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"총 클립: {len(samples)}개  (SEQ_LENGTH={seq_length}, MIN_VALID_FRAMES={MIN_VALID_FRAMES})")

    x_seqs, x_statics, ys = [], [], []
    sample_ids, subject_ids, splits = [], [], []
    source_groups, attack_types, devices = [], [], []
    seq_lengths_list, face_detect_rates = [], []
    skipped = []

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for sample in tqdm(samples, desc="클립 피처 추출"):
            static, frame_seq, total, valid, actual_len = process_clip(
                sample, img_base, face_mesh, seq_length
            )

            if static is None:
                reason = "no_face_detected" if valid == 0 else "insufficient_frames"
                skipped.append({
                    "sample_id":   sample["sample_id"],
                    "label":       sample["label"],
                    "reason":      reason,
                    "frame_count": total,
                    "valid_count": valid,
                })
                continue

            x_seqs.append(frame_seq)
            x_statics.append([static[name] for name in STATIC_FEATURE_NAMES])
            ys.append(sample["label"])

            sample_ids.append(sample["sample_id"])
            subject_ids.append(str(sample.get("subject_id") or ""))
            splits.append(sample["split"])
            source_groups.append(infer_source_group(sample["sample_id"]))
            attack_types.append(sample["attack_type"])
            devices.append(str(sample.get("device") or ""))
            seq_lengths_list.append(actual_len)
            face_detect_rates.append(round(valid / total, 4) if total else 0.0)

    if not x_seqs:
        print("ERROR: 처리된 클립 없음")
        return

    x_seq_arr    = np.array(x_seqs,     dtype=np.float32)
    x_static_arr = np.array(x_statics,  dtype=np.float32)
    y_arr        = np.array(ys,         dtype=np.int64)
    splits_arr   = np.array(splits,     dtype=object)
    seq_len_arr  = np.array(seq_lengths_list, dtype=np.int32)

    # train set 기준 1~99% clipping
    train_idx = np.where(splits_arr == "train")[0]
    if len(train_idx) > 0:
        print(f"\ntrain {len(train_idx)}개 기준 1~99% clipping 적용...")
        x_seq_arr, x_static_arr = apply_clipping(
            x_seq_arr, seq_len_arr, x_static_arr, train_idx
        )
    else:
        print("[WARN] train split 없음 — clipping 생략")

    np.savez_compressed(
        str(out_path),
        x_seq=x_seq_arr,
        x_static=x_static_arr,
        y=y_arr,
        seq_feature_names=np.array(SEQ_FEATURE_NAMES,    dtype=object),
        static_feature_names=np.array(STATIC_FEATURE_NAMES, dtype=object),
        sample_ids=np.array(sample_ids,    dtype=object),
        subject_ids=np.array(subject_ids,  dtype=object),
        splits=splits_arr,
        source_groups=np.array(source_groups, dtype=object),
        attack_types=np.array(attack_types,   dtype=object),
        devices=np.array(devices,             dtype=object),
        seq_lengths=seq_len_arr,
        face_detect_rates=np.array(face_detect_rates, dtype=np.float32),
    )

    skip_df = pd.DataFrame(skipped)
    skip_df.to_csv(out_path.parent / "skipped_clips_rel.csv", index=False)

    total_input = len(samples)
    success     = len(x_seqs)
    fail        = len(skipped)
    print(f"\n=== 전처리 완료 ===")
    print(f"  전체: {total_input}  |  성공: {success}  |  실패: {fail}  |  성공률: {success/total_input*100:.1f}%")
    print(f"  x_seq    : {x_seq_arr.shape}")
    print(f"  x_static : {x_static_arr.shape}")
    print(f"  저장     : {out_path}")

    df_summary = pd.DataFrame({
        "split":        splits_arr,
        "label":        y_arr,
        "source_group": source_groups,
        "attack_type":  attack_types,
    })
    print()
    for s in ["train", "valid", "test"]:
        sub = df_summary[df_summary.split == s]
        n0  = (sub.label == 0).sum()
        n1  = (sub.label == 1).sum()
        src = sub.source_group.value_counts().to_dict()
        print(f"  {s:5s}: {len(sub):4d}  (real={n0}, spoof={n1})  {src}")

    if fail > 0:
        print(f"\n  실패 상세: {out_path.parent / 'skipped_clips_rel.csv'}")
        print(skip_df["reason"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing 전처리 v2 (상대좌표 버전)")
    parser.add_argument("--manifest",   default="../../data/facial_recognition/samples_manifest.jsonl")
    parser.add_argument("--img-dir",    default="../../data/facial_recognition/face_images")
    parser.add_argument("--out",        default="features/face_clip_data_rel.npz")
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    args = parser.parse_args()

    main(
        Path(args.manifest),
        Path(args.img_dir),
        Path(args.out),
        args.seq_length,
    )
