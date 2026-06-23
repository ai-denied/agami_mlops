#!/usr/bin/env python3
"""모델팀이 패키징 전에 자기 산출물 폴더를 미리 점검할 수 있는 독립 도구.

package_emotion_model.py가 내부적으로 쓰는 deployment/model_store.py의
contract 검증을 그대로 노출한다 - candidates/{version}/이나 current/뿐만
아니라 아무 디렉터리에나 돌릴 수 있다 (예: 아직 패키징하지 않은 학습
run 출력 디렉터리에 6개 파일을 모아놓고 미리 점검).

사용법:
    python -m context_emotion.scripts.validate_model_artifacts --dir runs/emotion_classifier_v1/packaged
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.deployment import model_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="contract 검증: 필수 파일 + metadata + label_schema + preprocessing_config")
    ap.add_argument("--dir", required=True, help="검증할 디렉터리 (model.onnx, metadata.json, ... 6개 파일이 있어야 함)")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[FAILED] 디렉터리를 찾을 수 없습니다: {args.dir}", file=sys.stderr)
        sys.exit(1)

    problems = model_store.validate_artifact_dir(args.dir)
    flat = [p for plist in problems.values() for p in plist]

    print(f"검증 대상: {args.dir}")
    print(f"필수 파일: {model_store.REQUIRED_FILES}")
    print()
    for category, plist in problems.items():
        status = "PASS" if not plist else "FAIL"
        print(f"[{status}] {category}")
        for p in plist:
            print(f"    - {p}")

    print()
    if flat:
        print(f"[FAILED] {len(flat)}개 문제 발견 - 이 디렉터리는 패키징/승격에 쓸 수 없습니다.")
        sys.exit(1)
    print("[OK] contract 검증 통과")


if __name__ == "__main__":
    main()
