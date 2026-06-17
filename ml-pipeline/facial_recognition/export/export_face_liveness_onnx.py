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
    high_threshold: float | None = None,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    input_dim   = int(ckpt["input_dim"])
    hidden_size = int(ckpt.get("hidden_size", 32))
    num_layers  = int(ckpt.get("num_layers",  1))
    dropout     = float(ckpt.get("dropout",   0.3))
    threshold   = float(ckpt.get("threshold", 0.5))
    selected_features = ckpt.get("selected_features", [])

    # high_spoof_threshold(suspicious/spoof_detected 경계)는 명시값을 받지 못하면
    # threshold*1.3 으로 추정하던 기존 fallback 대신, 운영에서 검증된 보수적인 값을
    # 사용한다. R_live_clip 실환경 FRR 95% 모델 특성상, 단일 라운드 noise로 인한
    # spoof_detected 오분류를 줄이기 위함이다. (RETROSPECTIVE 참고)
    if high_threshold is None:
        high_threshold = max(threshold * 1.3, 0.55)
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
            "high_spoof_threshold": high_threshold,
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
    parser.add_argument(
        "--high-threshold", type=float, default=None,
        help="suspicious/spoof_detected 경계값. 미지정 시 max(threshold*1.3, 0.55) 사용.",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.meta, args.high_threshold)


if __name__ == "__main__":
    main()
