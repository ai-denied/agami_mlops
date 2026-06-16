from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_liveness_predictor import GRUClassifier, FaceLivenessPredictor


class FaceLivenessOnnxModule(nn.Module):
    """Fixed-length ONNX inference module for website/runtime integration.

    Input `x_seq` is raw, unscaled feature sequence with shape `(batch, 16, 20)`.
    The training scaler is embedded into the graph, then the GRU returns
    `spoof_score` after sigmoid. For the website demo we use full 16-frame
    windows, so no packed sequence or variable length input is exported.
    """

    def __init__(self, predictor: FaceLivenessPredictor):
        super().__init__()
        self.gru = predictor.model.gru
        self.head = predictor.model.head

        mean = predictor.scaler_mean
        scale = predictor.scaler_scale
        if mean is None or scale is None:
            input_dim = len(predictor.selected_features)
            mean = np.zeros((input_dim,), dtype=np.float32)
            scale = np.ones((input_dim,), dtype=np.float32)

        mean_t = torch.tensor(np.asarray(mean, dtype=np.float32)).view(1, 1, -1)
        scale_t = torch.tensor(np.asarray(scale, dtype=np.float32)).view(1, 1, -1)
        scale_t = torch.where(scale_t == 0, torch.ones_like(scale_t), scale_t)
        self.register_buffer("scaler_mean", mean_t)
        self.register_buffer("scaler_scale", scale_t)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_scaled = (x_seq - self.scaler_mean) / self.scaler_scale
        _, h_n = self.gru(x_scaled)
        h_last = h_n[-1]
        logits = self.head(h_last).squeeze(-1)
        return torch.sigmoid(logits)


def main() -> None:
    model_path = PROJECT_ROOT / "runs/gru_h32_lr0005_v1/best_gru.pt"
    out_path = PROJECT_ROOT / "runs/gru_h32_lr0005_v1/best_gru.onnx"
    meta_path = PROJECT_ROOT / "runs/gru_h32_lr0005_v1/best_gru_onnx_meta.json"

    predictor = FaceLivenessPredictor(model_path=str(model_path), device="cpu")
    module = FaceLivenessOnnxModule(predictor).eval()

    dummy = torch.zeros((1, 16, len(predictor.selected_features)), dtype=torch.float32)

    with torch.no_grad():
        onnx_like_score = float(module(dummy).cpu().numpy()[0])
        predictor_score = predictor.predict_score(dummy.numpy()[0], seq_length=16)

    torch.onnx.export(
        module,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x_seq"],
        output_names=["spoof_score"],
        dynamic_axes={"x_seq": {0: "batch"}, "spoof_score": {0: "batch"}},
    )

    meta = {
        "model": "GRU face liveness",
        "onnx_file": "best_gru.onnx",
        "source_checkpoint": "best_gru.pt",
        "input_name": "x_seq",
        "input_shape": ["batch", 16, len(predictor.selected_features)],
        "input_dtype": "float32",
        "output_name": "spoof_score",
        "output_shape": ["batch"],
        "output_meaning": "Higher value means higher spoof risk.",
        "threshold": predictor.threshold,
        "selected_features": predictor.selected_features,
        "preprocessing": "Raw 20 face features in selected_features order. Training scaler is embedded in ONNX graph.",
        "export_check": {
            "zero_input_pytorch_predictor_score": predictor_score,
            "zero_input_onnx_module_score_before_export": onnx_like_score,
            "absolute_diff": abs(predictor_score - onnx_like_score),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("exported:", out_path)
    print("metadata:", meta_path)
    print("predictor_score:", predictor_score)
    print("onnx_module_score:", onnx_like_score)
    print("abs_diff:", abs(predictor_score - onnx_like_score))


if __name__ == "__main__":
    main()
