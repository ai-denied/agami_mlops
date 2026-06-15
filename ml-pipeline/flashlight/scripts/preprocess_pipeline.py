#!/usr/bin/env python3
"""
flashlight 전처리 파이프라인 CLI 래퍼

extract_all_features.py와 merge_dynamic_json.py를 순서대로 실행한다.
두 스크립트가 CWD 기준 하드코딩 경로를 사용하므로 --data-dir 로 CWD를 지정한다.

사용법:
  python -m flashlight.scripts.preprocess_pipeline \
    --data-dir /mnt/flashlight-data/data/flashlight

  실행 후 생성 파일:
    {data-dir}/processed/dynamic/human/*.json     ← 인간 세션 동적 피처
    {data-dir}/processed/dynamic/bot/{type}/*.json ← 봇 세션 동적 피처
    {data-dir}/merged_dynamic_features_sampled.json ← 학습 입력 파일
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="flashlight 전처리 파이프라인 (extract + merge)"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="mouse_logs/ 가 위치한 데이터 루트 (e.g. /mnt/flashlight-data/data/flashlight)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"[SKIP] --data-dir 가 존재하지 않습니다: {data_dir}")
        print("[SKIP] 원시 데이터가 아직 수집되지 않았습니다. 전처리를 건너뜁니다.")
        sys.exit(0)

    human_dir = os.path.join(data_dir, "mouse_logs", "human")
    if not os.path.isdir(human_dir):
        print(f"[SKIP] mouse_logs/human 디렉토리 없음: {human_dir}")
        print("[SKIP] 수집된 원시 로그가 없습니다. 전처리를 건너뜁니다.")
        sys.exit(0)

    # preprocessing 스크립트 디렉토리: flashlight/preprocessing/
    preprocessing_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing")
    )

    print(f"[preprocess] data-dir    : {data_dir}")
    print(f"[preprocess] scripts-dir : {preprocessing_dir}")
    print()

    for script_name in ("extract_all_features.py", "merge_dynamic_json.py"):
        script_path = os.path.join(preprocessing_dir, script_name)
        if not os.path.isfile(script_path):
            print(f"[ERROR] 스크립트를 찾을 수 없습니다: {script_path}", file=sys.stderr)
            sys.exit(1)

        print(f"[preprocess] === {script_name} ===")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=data_dir,
        )
        if result.returncode != 0:
            print(
                f"[ERROR] {script_name} 실패 (exit code {result.returncode})",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
        print(f"[preprocess] {script_name} 완료")
        print()

    merged_path = os.path.join(data_dir, "merged_dynamic_features_sampled.json")
    if os.path.exists(merged_path):
        size = os.path.getsize(merged_path)
        print(f"[preprocess] 병합 파일: {merged_path} ({size:,} bytes)")
    else:
        print(f"[WARNING] 병합 파일이 생성되지 않았습니다: {merged_path}", file=sys.stderr)

    print("[preprocess] 전처리 파이프라인 완료")


if __name__ == "__main__":
    main()
