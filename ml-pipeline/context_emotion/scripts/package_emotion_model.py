#!/usr/bin/env python3
"""학습 산출물 -> model-store/context_emotion/candidates/{version}/ 패키징.

flashlight/scripts/package_for_captcha_engine.py와 같은 역할이지만,
검증을 통과하기 전에는 candidates/{version}/이 존재하지 않는다 - 임시
staging 디렉터리에 복사 + contract 검증을 마친 뒤에만 최종 위치로
옮긴다. 검증 실패 시 후보 생성 자체가 안 남는다 (요구사항: "실패하면
후보 생성 자체를 막아야 한다").

입력 5종은 모두 모델팀이 만들어서 넘기는 산출물이다:
    --onnx                 model.onnx
    --metadata              metadata.json (contracts/model_metadata.schema.json)
    --label-schema           label_schema.json (contracts/label_schema_contract.md)
    --preprocessing-config    preprocessing_config.json
    --evaluation-result       evaluation_result.json (evaluate_candidate.py 출력)
checkpoint(.pt)는 패키징하지 않는다 - metadata.json의 checkpoint_source
필드에 경로만 기록해 추적성만 남긴다 (model-store에는 서빙에 필요한
ONNX만 둔다는 flashlight 관례를 그대로 따름).

사용법:
    python -m context_emotion.scripts.package_emotion_model \\
        --onnx runs/emotion_classifier_v1/model.onnx \\
        --metadata runs/emotion_classifier_v1/metadata.json \\
        --label-schema runs/emotion_classifier_v1/label_schema.json \\
        --preprocessing-config runs/emotion_classifier_v1/preprocessing_config.json \\
        --evaluation-result runs/emotion_classifier_v1/evaluation_result.json \\
        --version v1_20260701
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.deployment import model_store  # noqa: E402

_INPUT_TO_OUTPUT_NAME = {
    "onnx": "model.onnx",
    "metadata": "metadata.json",
    "label_schema": "label_schema.json",
    "preprocessing_config": "preprocessing_config.json",
    "evaluation_result": "evaluation_result.json",
}


def _build_manifest(staging_dir: str, version: str, inputs: dict) -> dict:
    manifest = {
        "version": version,
        "packaged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "files": {},
        "source_artifacts": inputs,
    }
    for output_name in _INPUT_TO_OUTPUT_NAME.values():
        manifest["files"][output_name] = model_store.sha256_of(os.path.join(staging_dir, output_name))
    return manifest


def package(
    onnx: str,
    metadata: str,
    label_schema: str,
    preprocessing_config: str,
    evaluation_result: str,
    version: str,
    checkpoint: str = None,
    output_dir: str = None,
) -> str:
    """검증까지 통과한 candidate 디렉터리 경로를 반환한다. 실패 시 예외
    (FileNotFoundError/ValueError) - 이 경우 candidates/{version}/는 생성되지 않는다."""
    inputs = {
        "onnx": onnx, "metadata": metadata, "label_schema": label_schema,
        "preprocessing_config": preprocessing_config, "evaluation_result": evaluation_result,
    }
    if checkpoint:
        inputs["checkpoint"] = checkpoint

    missing = [(k, v) for k, v in inputs.items() if k != "checkpoint" and not os.path.isfile(v)]
    if missing:
        raise FileNotFoundError("입력 파일을 찾을 수 없습니다: " + ", ".join(f"--{k}={v}" for k, v in missing))

    final_dir = output_dir or model_store.candidate_dir(version)
    if model_store.is_current_dir(final_dir):
        raise ValueError("output-dir이 model-store/context_emotion/current/입니다 - "
                          "current는 package로 직접 수정할 수 없습니다. promote_model.py를 쓰세요.")
    if os.path.isdir(final_dir):
        raise ValueError(f"candidates/{version}/이 이미 존재합니다: {final_dir} (다른 버전명을 쓰세요)")

    with tempfile.TemporaryDirectory(prefix=f"context_emotion_pkg_{version}_") as staging_dir:
        for key, output_name in _INPUT_TO_OUTPUT_NAME.items():
            shutil.copy2(inputs[key], os.path.join(staging_dir, output_name))

        manifest = _build_manifest(staging_dir, version, inputs)
        with open(os.path.join(staging_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        problems = model_store.validate_artifact_dir(staging_dir)
        flat_problems = [p for plist in problems.values() for p in plist]
        if flat_problems:
            raise ValueError(
                f"candidate {version} contract 검증 실패 - 후보를 생성하지 않습니다:\n  - "
                + "\n  - ".join(flat_problems)
            )

        os.makedirs(os.path.dirname(final_dir), exist_ok=True)
        shutil.copytree(staging_dir, final_dir)

    return final_dir


def _parse_args():
    ap = argparse.ArgumentParser(description="학습 산출물을 model-store/context_emotion/candidates/{version}/로 패키징")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--label-schema", required=True)
    ap.add_argument("--preprocessing-config", required=True)
    ap.add_argument("--evaluation-result", required=True)
    ap.add_argument("--checkpoint", default=None, help="추적성 기록용 - 패키지에는 복사되지 않음")
    ap.add_argument("--version", required=True)
    ap.add_argument("--output-dir", default=None, help="기본값: model-store/context_emotion/candidates/{version}/")
    return ap.parse_args()


def main():
    args = _parse_args()
    try:
        final_dir = package(
            onnx=args.onnx, metadata=args.metadata, label_schema=args.label_schema,
            preprocessing_config=args.preprocessing_config, evaluation_result=args.evaluation_result,
            version=args.version, checkpoint=args.checkpoint, output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[FAILED] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] candidate 패키징 완료: {final_dir}")
    print("\n다음 단계:")
    print(f"  python -m context_emotion.evaluation.compare_candidate --version {args.version}")


if __name__ == "__main__":
    main()
