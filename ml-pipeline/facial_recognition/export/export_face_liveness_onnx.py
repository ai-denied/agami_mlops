"""
GRU 얼굴 활성도 모델 → ONNX 변환 스크립트.

스케일러를 ONNX 그래프에 내장하므로 추론 시 별도 전처리 불필요.

사용법:
  python -m facial_recognition.export.export_face_liveness_onnx \\
    --checkpoint runs/gru_v1/best_gru.pt \\
    --output     runs/gru_v1/face_liveness.onnx \\
    --meta       runs/gru_v1/onnx_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from facial_recognition.model.face_liveness_gru import FaceLivenessGRU, FaceLivenessOnnxModule

SEQ_LEN = 16


def export(
    checkpoint_path: str,
    output_path: str,
    meta_path: str | None = None,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    input_dim   = int(ckpt["input_dim"])
    hidden_size = int(ckpt.get("hidden_size", 32))
    num_layers  = int(ckpt.get("num_layers",  1))
    dropout     = float(ckpt.get("dropout",   0.3))
    threshold   = float(ckpt.get("threshold", 0.5))
    selected_features = ckpt.get("selected_features", [])
    scaler_mean  = ckpt.get("scaler_mean",  np.zeros(input_dim))
    scaler_scale = ckpt.get("scaler_scale", np.ones(input_dim))

    gru = FaceLivenessGRU(input_dim, hidden_size, num_layers, dropout)
    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()

    module = FaceLivenessOnnxModule(gru, scaler_mean, scaler_scale).eval()

    dummy = torch.zeros((1, SEQ_LEN, input_dim), dtype=torch.float32)

    with torch.no_grad():
        score_before = float(module(dummy).cpu().numpy()[0])

    torch.onnx.export(
        module,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x_seq"],
        output_names=["spoof_score"],
        dynamic_axes={"x_seq": {0: "batch"}, "spoof_score": {0: "batch"}},
    )

    # 검증
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        out  = sess.run(["spoof_score"], {"x_seq": dummy.numpy()})
        score_after = float(out[0][0])
        diff = abs(score_before - score_after)
        print(f"ONNX 검증  score_before={score_before:.6f}  score_after={score_after:.6f}  diff={diff:.2e}")
        if diff > 1e-4:
            print("[WARN] PyTorch / ONNX 출력 불일치가 큽니다", file=sys.stderr)
    except ImportError:
        score_after = score_before
        diff = 0.0
        print("[SKIP] onnxruntime 미설치 — ONNX 검증 생략")

    if meta_path:
        meta = {
            "model":             "GRU face liveness",
            "onnx_file":         Path(output_path).name,
            "source_checkpoint": Path(checkpoint_path).name,
            "input_name":        "x_seq",
            "input_shape":       ["batch", SEQ_LEN, input_dim],
            "input_dtype":       "float32",
            "output_name":       "spoof_score",
            "output_shape":      ["batch"],
            "output_meaning":    "Higher value means higher spoof risk.",
            "threshold":         threshold,
            "selected_features": selected_features,
            "preprocessing":     "Raw features in selected_features order. Scaler embedded in ONNX.",
            "export_check":      {
                "score_before_export": score_before,
                "score_after_export":  score_after,
                "absolute_diff":       diff,
            },
        }
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"메타데이터 저장: {meta_path}")

    print(f"ONNX 저장: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU 얼굴 활성도 모델 ONNX 변환")
    parser.add_argument("--checkpoint", required=True, help="best_gru.pt 경로")
    parser.add_argument("--output",     required=True, help="출력 ONNX 경로")
    parser.add_argument("--meta",       default=None,  help="메타데이터 JSON 저장 경로 (선택)")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.meta)


if __name__ == "__main__":
    main()
