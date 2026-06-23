#!/usr/bin/env python3
"""승격 직후(또는 아무 candidate/current에 대해서나) 실제로 ONNX가
로딩되고 한 번이라도 추론이 도는지 확인하는 마지막 안전망.

promote_model.py의 ONNX 검증과 비슷하지만 별도 스크립트로 분리한 이유:
이건 "승격해도 되는가"가 아니라 "방금 승격된 게 실제로 떠 있는가"를
보는, 배포 직후 헬스체크다 - run_model_update_pipeline.py에서 promote
다음 단계로 호출한다. onnxruntime이 없거나 모델이 없으면 그냥 종료하지
않고 명확한 실패로 처리한다 (가짜 성공 금지).

사용법:
    python -m context_emotion.scripts.smoke_test_model --dir model-store/context_emotion/current
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.deployment import model_store  # noqa: E402


def smoke_test(artifact_dir: str) -> dict:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit(f"[FAILED] smoke test 의존성 미설치 ({e}) - 가짜 통과를 출력하지 않습니다")

    onnx_path = os.path.join(artifact_dir, "model.onnx")
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    label_schema_path = os.path.join(artifact_dir, "label_schema.json")
    for p, label in [(onnx_path, "model.onnx"), (metadata_path, "metadata.json"), (label_schema_path, "label_schema.json")]:
        if not os.path.isfile(p):
            raise SystemExit(f"[FAILED] {label}을 찾을 수 없습니다: {p}")

    metadata = model_store.load_json(metadata_path)
    label_schema = model_store.load_json(label_schema_path)
    input_spec = metadata["input_spec"]
    output_spec = metadata["output_spec"]
    num_classes = len(label_schema["emotion_classes"])

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}
    if input_spec["name"] not in input_names:
        raise SystemExit(f"[FAILED] ONNX 입력 이름 불일치: expected '{input_spec['name']}', got {input_names}")

    shape = [1 if d == "batch" else int(d) for d in input_spec["shape"]]
    dummy = np.random.default_rng(42).standard_normal(shape).astype("float32")
    outputs = session.run([output_spec["name"]], {input_spec["name"]: dummy})[0]

    if outputs.shape[-1] != num_classes:
        raise SystemExit(f"[FAILED] 출력 차원 {outputs.shape[-1]} != label_schema 클래스 수 {num_classes}")
    if np.isnan(outputs).any():
        raise SystemExit("[FAILED] 출력에 NaN이 있습니다")

    print(f"[OK] {onnx_path}")
    print(f"[OK] 입력 {input_spec['name']} shape={shape} -> 출력 {output_spec['name']} shape={outputs.shape}")
    print(f"[OK] argmax class = {label_schema['emotion_classes'][int(outputs[0].argmax())]}")
    return {"smoke_test_passed": True, "output_shape": list(outputs.shape)}


def main():
    ap = argparse.ArgumentParser(description="ONNX 모델이 실제로 로딩/추론되는지 확인하는 배포 직후 헬스체크")
    ap.add_argument("--dir", required=True, help="model-store/context_emotion/current/ 또는 candidates/{version}/")
    args = ap.parse_args()
    smoke_test(args.dir)


if __name__ == "__main__":
    main()
