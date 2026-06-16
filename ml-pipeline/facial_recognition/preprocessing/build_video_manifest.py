"""
face_videos → samples_manifest.jsonl 추가 스크립트

face_videos/real/*.mp4   → VL_XXXX_clip001 (label=0, attack_type=live)
face_videos/spoof/*.mp4  → VS_XXXX_clip001 (label=1, attack_type=print|replay)

각 영상에서 FRAMES_PER_CLIP 장을 균등 샘플링해 face_images에 저장하고
manifest 항목을 생성한 뒤 samples_manifest.jsonl에 추가한다.

Usage (ml-pipeline/facial_recognition/ 기준):
  python preprocessing/build_video_manifest.py
  python preprocessing/build_video_manifest.py \\
    --video-dir  ../../data/facial_recognition/face_videos \\
    --img-dir    ../../data/facial_recognition/face_images \\
    --manifest   ../../data/facial_recognition/samples_manifest.jsonl \\
    --frames     30
"""

import argparse
import json
import re
from pathlib import Path

import cv2
from tqdm import tqdm

FRAMES_PER_CLIP = 30

# 영상 파일명 → attack_type 매핑
def infer_attack_type(filename: str) -> str:
    name = filename.lower()
    if "replay" in name:
        return "replay"
    if "print" in name:
        return "print"
    return "live"

# 영상 파일명 → 짧은 ID 생성 (live_video10.mp4 → live_video10)
def video_stem(path: Path) -> str:
    return path.stem

# split 할당 (video 단위, 비율 70/15/15)
def assign_split(idx: int, total: int) -> str:
    r = idx / total
    if r < 0.70:
        return "train"
    if r < 0.85:
        return "valid"
    return "test"


def extract_frames(video_path: Path, out_dir: Path, n_frames: int) -> list[str]:
    """
    영상에서 n_frames장을 균등 샘플링해 out_dir에 저장.
    Returns: face_images/ 기준 상대경로 리스트 (manifest에 기록할 경로)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for seq_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"{seq_idx+1:03d}.jpg"
        fpath = out_dir / fname
        cv2.imwrite(str(fpath), frame)
        # face_images/ 기준 상대경로
        saved.append(str(fpath.relative_to(fpath.parents[2])))

    cap.release()
    return saved


def main(video_dir: Path, img_dir: Path, manifest_path: Path, n_frames: int):
    # 기존 manifest 로드 (중복 방지용 sample_id 집합)
    existing_ids = set()
    existing_entries = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    existing_ids.add(s["sample_id"])
                    existing_entries.append(s)

    new_entries = []

    for label_name, label_id in [("real", 0), ("spoof", 1)]:
        vid_dir = video_dir / label_name
        videos = sorted(vid_dir.glob("*.mp4"))
        total = len(videos)

        for idx, vpath in enumerate(tqdm(videos, desc=f"{label_name} 영상 처리")):
            stem = video_stem(vpath)
            attack_type = infer_attack_type(stem)

            # sample_id: VL_live_video10_clip001 / VS_replay_video3_clip001
            prefix = "VL" if label_id == 0 else "VS"
            sample_id = f"{prefix}_{stem}_clip001"

            if sample_id in existing_ids:
                print(f"  [SKIP] 이미 존재: {sample_id}")
                continue

            # 프레임 저장 경로
            sub = "real" if label_id == 0 else "spoof"
            out_dir = img_dir / sub / f"vid_{stem}"

            frames = extract_frames(vpath, out_dir, n_frames)
            if len(frames) < 3:
                print(f"  [SKIP] 프레임 부족: {vpath.name} ({len(frames)}장)")
                continue

            split = assign_split(idx, total)

            entry = {
                "sample_id":   sample_id,
                "subject_id":  sample_id,   # 영상 단위로 subject 구분
                "split":       split,
                "label":       label_id,
                "attack_type": attack_type,
                "session":     None,
                "illumination": None,
                "device":      None,
                "frames":      frames,
            }
            new_entries.append(entry)
            existing_ids.add(sample_id)

    if not new_entries:
        print("추가할 항목 없음.")
        return

    # manifest에 추가
    with open(manifest_path, "a", encoding="utf-8") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    live  = sum(1 for e in new_entries if e["label"] == 0)
    spoof = sum(1 for e in new_entries if e["label"] == 1)
    print(f"\n완료: {len(new_entries)}개 추가 (live={live}, spoof={spoof})")

    from collections import Counter
    splits = Counter(e["split"] for e in new_entries)
    print(f"split: {dict(splits)}")
    print(f"manifest 저장: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir",  default="../../data/facial_recognition/face_videos")
    parser.add_argument("--img-dir",    default="../../data/facial_recognition/face_images")
    parser.add_argument("--manifest",   default="../../data/facial_recognition/samples_manifest.jsonl")
    parser.add_argument("--frames",     type=int, default=FRAMES_PER_CLIP)
    args = parser.parse_args()

    main(
        Path(args.video_dir),
        Path(args.img_dir),
        Path(args.manifest),
        args.frames,
    )
