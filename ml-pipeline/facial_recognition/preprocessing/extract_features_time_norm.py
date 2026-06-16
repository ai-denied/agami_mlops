"""
얼굴 Anti-Spoofing 전처리 v3 — face_clip_data_time_norm.npz

# 배경: R_live_clip 오탐 원인 분석
─────────────────────────────────────────────────────────────────────────
S_dataset_sequence: 파일명 패턴 _60.jpg, _120.jpg → frame_interval=60
  원본 30fps 기준: 프레임 간 실제 시간 = 60/30 = 2.0초
  16프레임 커버 시간: 15 × 2.0s = 30초

R_live_clip / ATK_external_clip: 파일명 001.jpg, 002.jpg → frame_interval=1
  원본 30fps 기준: 프레임 간 실제 시간 = 1/30 ≈ 0.033초
  16프레임 커버 시간: 15 × 0.033s = 0.5초

결과: velocity/displacement feature의 크기가 source별로 최대 60배 차이
  nose_dx   : S_live=0.063, R_live=0.019, ATK_spoof=0.018 → R_live≈ATK_spoof
  nose_speed: S_live=0.088, R_live=0.020, ATK_spoof=0.022 → R_live≈ATK_spoof
  pitch_vel : S_live=0.129, R_live=0.033, ATK_spoof=0.033 → R_live≡ATK_spoof

→ 모델이 "작은 velocity = spoof"로 학습하여 R_live_clip을 오탐

# v3 전처리 변경사항
─────────────────────────────────────────────────────────────────────────
v2(extract_features_rel.py) 대비 변경사항:

  1. frame_interval 추정 (파일명 숫자 패턴 기반)
     - S_dataset_sequence: _60.jpg → frame_interval=60
     - R_live_clip / ATK_external_clip: 001.jpg → frame_interval=1

  2. displacement/velocity feature를 frame_interval로 나눠 시간 정규화
     nose_dx_tn    = nose_dx    / frame_interval
     nose_dy_tn    = nose_dy    / frame_interval
     center_dx_tn  = center_dx  / frame_interval
     center_dy_tn  = center_dy  / frame_interval
     nose_speed_tn = nose_speed / frame_interval
     ear_vel_tn    = ear_velocity   / frame_interval
     mar_vel_tn    = mar_velocity   / frame_interval
     yaw_vel_tn    = yaw_velocity   / frame_interval
     pitch_vel_tn  = pitch_velocity / frame_interval
     roll_vel_tn   = roll_velocity  / frame_interval

  3. static feature face_movement / nose_movement도 frame_interval 정규화
     (v2의 /n 정규화에 추가로 /frame_interval)

  4. frame_interval을 clip 메타데이터로 NPZ에 저장

  v2에서 유지된 부분:
  - bbox 기준 nose/cx/cy 정규화 (카메라 거리 불변)
  - 클립 평균 기준 상대좌표 (nose_x_rel 등)
  - forward/backward fill 보간
  - MIN_VALID_FRAMES=3 필터
  - train set 1~99% percentile clipping
  - FaceMesh video mode (static_image_mode=False)

# 한계
─────────────────────────────────────────────────────────────────────────
시간 정규화 후에도 R_live(frame_interval=1)와 ATK_spoof(frame_interval=1)는
동일한 temporal scale이므로, velocity feature만으로는 구분이 어려움.
R_live를 ATK_spoof와 구별하는 데 더 유효한 feature:
  - blink_count: R_live=1.63 > ATK_spoof=0.93 (연속 프레임에서 실제 눈깜박임 포착)
  - ear_std: R_live=0.145 >> ATK_spoof=0.095 (실제 눈동작 변화)
  - ear_velocity_tn: R_live > ATK_spoof (blink에 의한 EAR 변동)
근본 해결을 위해서는 R_live 데이터를 S와 동일한 temporal density로 수집하거나
source-weighted 학습과 병행이 필요합니다.

출력:
  face_clip_data_time_norm.npz
    x_seq:    float32 (N, 16, 20)   — GRU 입력 (velocity는 시간 정규화)
    x_static: float32 (N, 16)       — SVM/RF 비교용
    y:        int64   (N,)
    seq_feature_names / static_feature_names
    sample_ids, subject_ids, splits, source_groups,
    attack_types, devices, seq_lengths, face_detect_rates, frame_intervals

Usage (ml-pipeline/facial_recognition/ 기준):
  python preprocessing/extract_features_time_norm.py
  python preprocessing/extract_features_time_norm.py \\
    --manifest ../../data/facial_recognition/samples_manifest.jsonl \\
    --img-dir  ../../data/facial_recognition/face_images \\
    --out      features/face_clip_data_time_norm.npz
"""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mode

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── 상수 ──────────────────────────────────────────────────────────────────────

SEQ_LENGTH       = 16
MIN_VALID_FRAMES = 3
BLINK_THRESH     = 0.20
DEFAULT_FPS      = 30   # 실제 fps 미지정 시 기본값 (frame_interval을 초로 변환하는 데 사용)

SEQ_FEATURE_NAMES = [
    "ear", "mar", "smile_w",
    "nose_x_rel", "nose_y_rel", "cx_rel", "cy_rel",
    "roll", "yaw", "pitch",
    # velocity — 아래 _tn 접미사: frame_interval로 나눈 시간 정규화 버전
    "nose_dx_tn", "nose_dy_tn",
    "center_dx_tn", "center_dy_tn",
    "nose_speed_tn",
    "ear_vel_tn", "mar_vel_tn",
    "yaw_vel_tn", "pitch_vel_tn", "roll_vel_tn",
]

STATIC_FEATURE_NAMES = [
    "blink_count", "eye_open_ratio", "mouth_open_ratio", "smile_ratio",
    "head_yaw", "head_pitch", "head_roll", "face_movement_tn", "face_stability",
    "ear_std", "mar_mean", "mar_std", "nose_movement_tn",
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
    if s.startswith("ATK"): return "ATK_external_clip"
    if s.startswith("VL"):  return "VL_face_video"
    if s.startswith("VS"):  return "VS_face_video"
    if s.startswith("R"):   return "R_live_clip"
    if s.startswith("S"):   return "S_dataset_sequence"
    return "unknown"


def estimate_frame_interval(frames: list) -> int:
    """
    파일명 숫자 패턴에서 frame index 간격 추정.

    - S_dataset_sequence: real/live_001/live_001-1-1-1-1_60.jpg → 숫자=60, 간격=60
    - R_live_clip / ATK_external_clip: real/real_01_phone/001.jpg → 숫자=1,2,3, 간격=1

    최빈값(mode)을 사용해 이상치에 강건하게 추정.
    숫자 패턴을 찾지 못하면 1 반환 (연속 프레임 가정).
    """
    nums = []
    for f in frames:
        # 파일명 마지막 숫자 블록 추출: _60.jpg 또는 /001.jpg 형태 모두 처리
        m = re.search(r'[_/](\d+)\.jpg$', f)
        if m:
            nums.append(int(m.group(1)))

    if len(nums) >= 2:
        diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
        pos_diffs = [d for d in diffs if d > 0]
        if pos_diffs:
            try:
                return mode(pos_diffs)   # 최빈값
            except Exception:
                return round(sum(pos_diffs) / len(pos_diffs))

    return 1  # 연속 프레임 기본값


# ── 기하학적 계산 ─────────────────────────────────────────────────────────────

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
    v2와 동일: bbox 기준 정규화 좌표.
    nose_x/y, cx/cy = (raw - face_cx) / face_w
    """
    face_w  = _dist(lm, FACE_LEFT_IDX, FACE_RIGHT_IDX) + 1e-6
    face_cx = (lm[FACE_LEFT_IDX].x + lm[FACE_RIGHT_IDX].x) / 2.0
    face_cy = (lm[FACE_LEFT_IDX].y + lm[FACE_RIGHT_IDX].y) / 2.0

    ear_l = _ear(lm, LEFT_EYE)
    ear_r = _ear(lm, RIGHT_EYE)

    cx_raw = (lm[33].x + lm[133].x + lm[362].x + lm[263].x) / 4.0
    cy_raw = (lm[33].y + lm[133].y + lm[362].y + lm[263].y) / 4.0

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
    """v2와 동일: forward fill → backward fill."""
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


# ── 클립 단위 시퀀스 배열 ─────────────────────────────────────────────────────

def build_seq_array(
    frames: list,
    seq_length: int,
    frame_interval: int,
) -> tuple:
    """
    (seq_length, 20) float32 배열 생성.

    v2 대비 변경:
    - velocity 10개를 frame_interval로 나눠 시간 정규화
      → 서로 다른 temporal density source에서 velocity 크기를 비교 가능하게 만듦
    - 절대좌표(nose_x_rel 등)는 v2와 동일 (클립 평균 기준 상대좌표)

    Returns: (frame_seq, actual_len)
    """
    fi = max(frame_interval, 1)  # 0 방지
    use_n = min(len(frames), seq_length)
    clip  = frames[:use_n]

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
            # 원시 차분
            nose_dx   = nose_x_rel - (prev["nose_x"] - mean_nose_x)
            nose_dy   = nose_y_rel - (prev["nose_y"] - mean_nose_y)
            center_dx = cx_rel     - (prev["cx"]     - mean_cx)
            center_dy = cy_rel     - (prev["cy"]     - mean_cy)

            # ── 시간 정규화: frame_interval로 나눔 ───────────────────────────
            # frame_interval=60 이면 /60, frame_interval=1 이면 /1
            # → "frame index 단위당 변화량"으로 스케일 통일
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

        frame_seq[i] = np.concatenate([abs_feats, vel_feats])

    return frame_seq, use_n


# ── 클립 단위 static 집계 ─────────────────────────────────────────────────────

def count_blinks(ear_series: list) -> int:
    below = [e < BLINK_THRESH for e in ear_series]
    return sum(1 for i in range(1, len(below)) if below[i] and not below[i - 1])


def aggregate_static(frames: list, frame_interval: int) -> dict:
    """
    16개 static 피처.

    v2 대비 변경:
    - face_movement_tn, nose_movement_tn: frame_interval 추가 정규화
      → 프레임당 이동량을 "frame index 단위당"으로 재스케일
    """
    fi = max(frame_interval, 1)

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

    # face_movement_tn: (누적합 / (n-1)) / frame_interval
    nose_movement_tn = sum(
        math.hypot(nose_x[i] - nose_x[i-1], nose_y[i] - nose_y[i-1])
        for i in range(1, n)
    ) / (denom * fi)

    face_movement_tn = sum(
        math.hypot(cx[i] - cx[i-1], cy[i] - cy[i-1])
        for i in range(1, n)
    ) / (denom * fi)

    def _std(a):  return float(np.std(a))  if len(a) > 1 else 0.0
    def _mean(a): return float(np.mean(a))

    std_cx = float(np.std(cx)) if n > 1 else 0.0
    std_cy = float(np.std(cy)) if n > 1 else 0.0

    return {
        "blink_count":       count_blinks(ears),
        "eye_open_ratio":    _mean(ears),
        "mouth_open_ratio":  _mean(mars),
        "smile_ratio":       _std(smiles),
        "head_yaw":          _std(yaws),
        "head_pitch":        _std(pitches),
        "head_roll":         _std(rolls),
        "face_movement_tn":  face_movement_tn,
        "face_stability":    math.hypot(std_cx, std_cy),
        "ear_std":           _std(ears),
        "mar_mean":          _mean(mars),
        "mar_std":           _std(mars),
        "nose_movement_tn":  nose_movement_tn,
        "head_yaw_mean":     _mean(yaws),
        "head_pitch_mean":   _mean(pitches),
        "head_roll_mean":    _mean(rolls),
    }


# ── 클립 처리 ─────────────────────────────────────────────────────────────────

def process_clip(
    sample: dict, img_base: Path, face_mesh, seq_length: int
) -> tuple:
    """
    manifest sample 1개 처리.
    Returns (static_dict, frame_seq, frame_interval, total_frames, valid_count, actual_seq_len)
    실패 시 static_dict = None.
    """
    frame_interval = estimate_frame_interval(sample["frames"])

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
        return None, None, frame_interval, total, valid_count, 0

    filled = interpolate_frames(raw_frames)
    frame_seq, actual_len = build_seq_array(filled, seq_length, frame_interval)
    static = aggregate_static(filled[:actual_len], frame_interval)

    return static, frame_seq, frame_interval, total, valid_count, actual_len


# ── Percentile Clipping ───────────────────────────────────────────────────────

def apply_clipping(
    x_seq: np.ndarray,
    seq_lengths: np.ndarray,
    x_static: np.ndarray,
    train_idx: np.ndarray,
) -> tuple:
    """train set 기준 1~99% percentile clipping (v2와 동일)."""
    seq_frames = []
    for i in train_idx:
        l = int(seq_lengths[i])
        if l > 0:
            seq_frames.append(x_seq[i, :l, :])
    seq_all = np.concatenate(seq_frames, axis=0)
    seq_lo  = np.percentile(seq_all, 1,  axis=0)
    seq_hi  = np.percentile(seq_all, 99, axis=0)

    static_train = x_static[train_idx]
    static_lo = np.percentile(static_train, 1,  axis=0)
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
    frame_intervals_list = []
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
            static, frame_seq, fi, total, valid, actual_len = process_clip(
                sample, img_base, face_mesh, seq_length
            )

            if static is None:
                reason = "no_face_detected" if valid == 0 else "insufficient_frames"
                skipped.append({
                    "sample_id":      sample["sample_id"],
                    "label":          sample["label"],
                    "reason":         reason,
                    "frame_count":    total,
                    "valid_count":    valid,
                    "frame_interval": fi,
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
            frame_intervals_list.append(fi)

    if not x_seqs:
        print("ERROR: 처리된 클립 없음")
        return

    x_seq_arr    = np.array(x_seqs,     dtype=np.float32)
    x_static_arr = np.array(x_statics,  dtype=np.float32)
    y_arr        = np.array(ys,         dtype=np.int64)
    splits_arr   = np.array(splits,     dtype=object)
    seq_len_arr  = np.array(seq_lengths_list, dtype=np.int32)
    fi_arr       = np.array(frame_intervals_list, dtype=np.int32)

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
        frame_intervals=fi_arr,
    )

    skip_df = pd.DataFrame(skipped)
    skip_df.to_csv(out_path.parent / "skipped_clips_time_norm.csv", index=False)

    total_input = len(samples)
    success     = len(x_seqs)
    fail        = len(skipped)
    print(f"\n=== 전처리 완료 (v3 time-norm) ===")
    print(f"  전체: {total_input}  |  성공: {success}  |  실패: {fail}  |  성공률: {success/total_input*100:.1f}%")
    print(f"  x_seq    : {x_seq_arr.shape}")
    print(f"  x_static : {x_static_arr.shape}")
    print(f"  저장     : {out_path}")

    # source별 frame_interval 분포 출력
    sg_arr = np.array(source_groups)
    print("\n  source별 frame_interval 분포:")
    for sg in sorted(set(source_groups)):
        fi_vals = fi_arr[sg_arr == sg]
        unique_fis = sorted(set(fi_vals.tolist()))
        print(f"    {sg}: {unique_fis} (n={len(fi_vals)})")

    # split/label/source 요약
    df_summary = pd.DataFrame({
        "split":        splits_arr,
        "label":        y_arr,
        "source_group": source_groups,
        "frame_interval": fi_arr,
    })
    print()
    for s in ["train", "valid", "test"]:
        sub = df_summary[df_summary.split == s]
        n0  = (sub.label == 0).sum()
        n1  = (sub.label == 1).sum()
        src = sub.source_group.value_counts().to_dict()
        print(f"  {s:5s}: {len(sub):4d}  (real={n0}, spoof={n1})  {src}")

    if fail > 0:
        print(f"\n  실패 상세: {out_path.parent / 'skipped_clips_time_norm.csv'}")
        print(skip_df["reason"].value_counts().to_string())

    # ── report 저장 ────────────────────────────────────────────────────────────
    report_path = out_path.parent / "preprocessing_report_v3.json"
    report = {
        "version": "v3_time_norm",
        "file": str(out_path),
        "n_samples": success,
        "seq_length": seq_length,
        "min_valid_frames": MIN_VALID_FRAMES,
        "frame_interval_by_source": {},
        "time_normalized_seq_features": [
            "nose_dx_tn", "nose_dy_tn", "center_dx_tn", "center_dy_tn", "nose_speed_tn",
            "ear_vel_tn", "mar_vel_tn", "yaw_vel_tn", "pitch_vel_tn", "roll_vel_tn",
        ],
        "time_normalized_static_features": ["face_movement_tn", "nose_movement_tn"],
        "normalization_formula": "feature_tn = raw_feature / frame_interval",
        "changes_from_v2": [
            "velocity/displacement features divided by estimated frame_interval",
            "face_movement and nose_movement also divided by frame_interval",
            "frame_intervals saved as NPZ metadata",
            "feature names use _tn suffix for time-normalized features",
        ],
        "known_limitations": [
            "R_live_clip (frame_interval=1) and ATK_external_clip (frame_interval=1) remain at same temporal scale",
            "time normalization does not make R_live ≈ S_live in velocity features",
            "R_live and ATK_spoof micro-motion at 1/30s scale remains similar",
            "discriminative features at dense sampling: blink_count, ear_std, ear_vel_tn",
        ],
    }
    for sg in sorted(set(source_groups)):
        fi_vals = fi_arr[sg_arr == sg].tolist()
        report["frame_interval_by_source"][sg] = {
            "unique": sorted(set(fi_vals)),
            "count": len(fi_vals),
        }

    import json as _json
    with open(report_path, "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  전처리 리포트: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing 전처리 v3 (시간 정규화 버전)")
    parser.add_argument("--manifest",   default="../../data/facial_recognition/samples_manifest.jsonl")
    parser.add_argument("--img-dir",    default="../../data/facial_recognition/face_images")
    parser.add_argument("--out",        default="features/face_clip_data_time_norm.npz")
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    args = parser.parse_args()

    main(
        Path(args.manifest),
        Path(args.img_dir),
        Path(args.out),
        args.seq_length,
    )
