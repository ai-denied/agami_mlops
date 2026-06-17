"""
얼굴 영상 전처리 스크립트
dataset/face_videos/{real,spoof} → features/face_video_features.csv

샘플링: 5프레임마다 1프레임 처리 → 영상 1개 = CSV 1행

Features per video (기획서 기준):
  blink_count         : EAR < 0.2 구간 진입 횟수
  eye_open_ratio      : EAR 평균 (눈이 열린 정도)
  mouth_open_ratio    : MAR 평균 (입 벌림 정도)
  smile_ratio         : 입꼬리 너비 비율의 std (입꼬리 변화량)
  head_yaw            : 고개 좌우 회전 std (변화량)
  head_pitch          : 고개 상하 회전 std (변화량)
  head_roll           : 고개 기울기 std (변화량)
  face_movement       : 얼굴 중심점 누적 이동량
  face_stability      : 얼굴 중심 위치의 표준편차 √(σx² + σy²)
  reaction_time       : (미지원 — 서버 타임스탬프 필요)

추가 보조 feature (모델 선택 폭 확보):
  ear_std             : EAR 표준편차
  mar_mean, mar_std   : MAR 평균·표준편차
  nose_movement       : 코 끝 누적 이동량
  head_yaw_mean       : yaw 평균 (고개 방향 기준)
  head_pitch_mean     : pitch 평균
  head_roll_mean      : roll 평균 (기울기 방향)

Usage:
  cd facial_recognition
  python preprocessing/extract_video_features.py
  python preprocessing/extract_video_features.py --dataset dataset/face_videos --out features --step 5
"""

import argparse
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE             = [33, 160, 158, 133, 153, 144]
RIGHT_EYE            = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT_IDX       = 61
MOUTH_RIGHT_IDX      = 291
MOUTH_TOP_IDX        = 13
MOUTH_BOTTOM_IDX     = 14
NOSE_TIP_IDX         = 4
FACE_LEFT_IDX        = 234   # 얼굴 왼쪽 끝
FACE_RIGHT_IDX       = 454   # 얼굴 오른쪽 끝
EYE_LEFT_OUTER_IDX   = 33    # 왼쪽 눈 바깥 코너
EYE_RIGHT_OUTER_IDX  = 263   # 오른쪽 눈 바깥 코너

VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}
LABEL_MAP    = {"real": 0, "spoof": 1}
BLINK_THRESH = 0.20


class _Landmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


def aspect_corrected_landmarks(landmarks, width: int, height: int):
    """
    MediaPipe landmark.x/y는 각각 이미지 너비/높이로 독립 정규화된다. 영상이
    정사각형이 아니면(대부분의 영상이 그렇다) y축이 체계적으로 왜곡되어
    _dist() 기반의 거리/각도가 source별 촬영 비율 차이만으로 달라진다
    (RETROSPECTIVE 참고).
    """
    aspect_ratio = (width / height) if height else 1.0
    return [_Landmark(lm.x, lm.y / aspect_ratio, lm.z) for lm in landmarks]


# ── 기본 계산 ────────────────────────────────────────────────────────────────

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
    """입꼬리 너비 / 얼굴 너비 — 값의 std가 smile_ratio"""
    mouth_w = _dist(lm, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX)
    face_w  = _dist(lm, FACE_LEFT_IDX, FACE_RIGHT_IDX)
    return mouth_w / face_w if face_w else 0.0


def _face_center(lm):
    """눈 양쪽 코너 4점 중심 → 얼굴 안정성 및 이동량 기준"""
    xs = [lm[i].x for i in [33, 133, 362, 263]]
    ys = [lm[i].y for i in [33, 133, 362, 263]]
    return sum(xs) / 4.0, sum(ys) / 4.0


# ── 헤드 포즈 (기하학적 근사) ─────────────────────────────────────────────────

def _head_roll(lm):
    """
    눈 라인 기울기 → roll (degrees).
    0° = 수평, 양수 = 오른쪽으로 기울어짐.
    """
    dx = lm[EYE_RIGHT_OUTER_IDX].x - lm[EYE_LEFT_OUTER_IDX].x
    dy = lm[EYE_RIGHT_OUTER_IDX].y - lm[EYE_LEFT_OUTER_IDX].y
    return math.degrees(math.atan2(dy, dx))


def _head_yaw(lm):
    """
    코 끝 x 좌표와 얼굴 중심 x 사이의 편차 → yaw proxy.
    0 = 정면, 양수 = 오른쪽으로 돌아감, 음수 = 왼쪽.
    범위: 대략 -1 ~ +1.
    """
    face_cx  = (lm[FACE_LEFT_IDX].x + lm[FACE_RIGHT_IDX].x) / 2.0
    half_w   = abs(lm[FACE_RIGHT_IDX].x - lm[FACE_LEFT_IDX].x) / 2.0 + 1e-6
    return (lm[NOSE_TIP_IDX].x - face_cx) / half_w


def _head_pitch(lm):
    """
    코 끝 y 좌표와 눈-입 중간점 사이의 편차 → pitch proxy.
    0 = 정면, 양수 = 위로 들어올림(코가 아래), 음수 = 아래로 숙임(코가 위).
    범위: 대략 -1 ~ +1.
    """
    eye_y    = (lm[EYE_LEFT_OUTER_IDX].y + lm[EYE_RIGHT_OUTER_IDX].y) / 2.0
    mouth_y  = (lm[MOUTH_LEFT_IDX].y    + lm[MOUTH_RIGHT_IDX].y)    / 2.0
    mid_y    = (eye_y + mouth_y) / 2.0
    half_h   = abs(mouth_y - eye_y) / 2.0 + 1e-6
    return (lm[NOSE_TIP_IDX].y - mid_y) / half_h


# ── 프레임 단위 특징 추출 ─────────────────────────────────────────────────────

def extract_frame_features(lm) -> dict:
    ear_l = _ear(lm, LEFT_EYE)
    ear_r = _ear(lm, RIGHT_EYE)
    cx, cy = _face_center(lm)
    return {
        "ear":         (ear_l + ear_r) / 2.0,
        "mar":         _mar(lm),
        "smile_w":     _smile_width_ratio(lm),
        "nose_x":      lm[NOSE_TIP_IDX].x,
        "nose_y":      lm[NOSE_TIP_IDX].y,
        "cx":          cx,
        "cy":          cy,
        "roll":        _head_roll(lm),
        "yaw":         _head_yaw(lm),
        "pitch":       _head_pitch(lm),
    }


# ── 영상 단위 집계 ────────────────────────────────────────────────────────────

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

    # 이동량 (누적 거리)
    nose_movement = sum(
        math.hypot(nose_x[i] - nose_x[i-1], nose_y[i] - nose_y[i-1])
        for i in range(1, n)
    )
    face_movement = sum(
        math.hypot(cx[i] - cx[i-1], cy[i] - cy[i-1])
        for i in range(1, n)
    )

    # 얼굴 안정성
    std_cx = float(np.std(cx)) if n > 1 else 0.0
    std_cy = float(np.std(cy)) if n > 1 else 0.0
    face_stability = math.hypot(std_cx, std_cy)

    def _std(arr): return float(np.std(arr)) if len(arr) > 1 else 0.0
    def _mean(arr): return float(np.mean(arr))

    return {
        # ── 기획서 feature 이름 ────────────────────────────────────────────
        "blink_count":       count_blinks(ears),
        "eye_open_ratio":    _mean(ears),          # EAR 평균
        "mouth_open_ratio":  _mean(mars),           # MAR 평균
        "smile_ratio":       _std(smile_ws),        # 입꼬리 너비 변화량 (std)
        "head_yaw":          _std(yaws),            # 좌우 회전 변화량 (std)
        "head_pitch":        _std(pitches),         # 상하 회전 변화량 (std)
        "head_roll":         _std(rolls),           # 기울기 변화량 (std)
        "face_movement":     face_movement,         # 얼굴 중심 이동량
        "face_stability":    face_stability,        # 얼굴 흔들림 std

        # ── 보조 feature (모델 선택 폭) ────────────────────────────────────
        "ear_std":           _std(ears),
        "mar_mean":          _mean(mars),
        "mar_std":           _std(mars),
        "nose_movement":     nose_movement,
        "head_yaw_mean":     _mean(yaws),
        "head_pitch_mean":   _mean(pitches),
        "head_roll_mean":    _mean(rolls),
    }


# ── 영상 처리 ─────────────────────────────────────────────────────────────────

def process_video(path: Path, face_mesh, step: int):
    """영상 1개 처리 → (집계 dict | None, total_sampled, valid_count)"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0, 0

    frame_idx     = 0
    total_sampled = 0
    valid_frames  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            total_sampled += 1
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = aspect_corrected_landmarks(result.multi_face_landmarks[0].landmark, w, h)
                valid_frames.append(extract_frame_features(lm))
        frame_idx += 1

    cap.release()

    if not valid_frames:
        return None, total_sampled, 0

    return aggregate(valid_frames), total_sampled, len(valid_frames)


def collect_paths(dataset_dir: Path):
    paths = []
    for label_name, label_id in LABEL_MAP.items():
        label_dir = dataset_dir / label_name
        if not label_dir.exists():
            print(f"[WARN] 폴더 없음: {label_dir}")
            continue
        for p in sorted(label_dir.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS:
                paths.append((p, label_name, label_id))
    return paths


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(dataset_dir: Path, out_dir: Path, step: int):
    out_dir.mkdir(exist_ok=True)
    paths = collect_paths(dataset_dir)
    print(f"총 영상: {len(paths)}개  (샘플링 간격: {step}프레임)")

    rows, skipped = [], []

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for path, label_name, label_id in tqdm(paths, desc="영상 특징 추출"):
            agg, total_sampled, valid_count = process_video(path, face_mesh, step)

            if agg is None:
                reason = "open_fail" if total_sampled == 0 else "no_face_detected"
                skipped.append({"file_path": str(path), "label": label_name, "reason": reason})
                continue

            face_detect_rate = valid_count / total_sampled if total_sampled else 0.0
            row = {
                "file_path":         str(path),
                "label":             label_id,
                **agg,
                "valid_frame_count": valid_count,
                "face_detect_rate":  round(face_detect_rate, 4),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "face_video_features.csv", index=False)

    skip_df   = pd.DataFrame(skipped)
    skip_path = out_dir / "skipped_files.csv"
    if skip_path.exists() and skip_path.stat().st_size > 0:
        existing = pd.read_csv(skip_path)
        skip_df  = pd.concat([existing, skip_df], ignore_index=True).drop_duplicates("file_path")
    skip_df.to_csv(skip_path, index=False)

    total   = len(paths)
    success = len(rows)
    fail    = len(skipped)
    print(f"\n=== 영상 특징 추출 완료 ===")
    print(f"  전체: {total}  |  성공: {success}  |  실패: {fail}  |  성공률: {success/total*100:.1f}%")
    print(f"  저장: {out_dir / 'face_video_features.csv'}")

    if success > 0:
        print(f"\n컬럼 목록 ({len(df.columns)}개):")
        for c in df.columns:
            print(f"  {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/face_videos")
    parser.add_argument("--out",     default="features")
    parser.add_argument("--step",    type=int, default=5)
    args = parser.parse_args()
    main(Path(args.dataset), Path(args.out), args.step)