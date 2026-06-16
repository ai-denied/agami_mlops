#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agami / Sentient-CAPTCHA Mouse GRU 모델 ONNX Export 스크립트

목적
- PyTorch 학습 결과(.pth)를 CAPTCHA 엔진에서 가볍게 추론할 수 있도록 ONNX로 변환한다.
- 학습 모델은 pack_padded_sequence를 사용하지만, ONNX 추론용 모델은 padded sequence에서 lengths 기준 마지막 유효 timestep을 gather하는 방식으로 구성한다.

실행 예시
python export_mouse_gru_to_onnx.py \
  --model ./runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.pth \
  --metadata ./runs/mouse_gru_final_v3_policy_tuned/mouse_metadata_server_final_v2.json \
  --output ./runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.onnx
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn as nn

SEQ_FEATURES = [
    "dx",
    "dy",
    "dt",
    "distance",
    "velocity",
    "acceleration",
    "angle_change",
]

STATIC_FEATURES = [
    "duration",
    "log_count",
    "total_distance",
    "straight_distance",
    "distance_ratio",
    "avg_speed",
    "max_speed",
    "speed_std",
    "direction_changes",
    "pauses",
]


class OnnxMouseGRUModel(nn.Module):
    """
    ONNX Export 전용 모델.

    학습 코드의 MouseGRUModelV2와 동일한 파라미터 이름을 유지한다.
    단, pack_padded_sequence는 ONNX에서 다루기 까다로우므로 사용하지 않고,
    GRU 전체 출력을 만든 뒤 lengths - 1 위치의 마지막 유효 timestep을 gather한다.
    """

    def __init__(self, seq_size: int = 7, static_size: int = 10, hidden: int = 32, layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(static_size, 32),
            nn.LayerNorm(32),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.fc_final = nn.Sequential(
            nn.Linear(hidden + 32, 64),
            nn.LayerNorm(64),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_seq: [batch, seq_len, 7]
        # lengths: [batch]
        # x_static: [batch, 10]
        gru_all, _ = self.gru(x_seq)

        batch_size = x_seq.size(0)
        last_indices = torch.clamp(lengths.long() - 1, min=0)
        batch_indices = torch.arange(batch_size, device=x_seq.device)
        gru_out = gru_all[batch_indices, last_indices, :]

        static_out = self.static_mlp(x_static)
        combined = torch.cat([gru_out, static_out], dim=1)
        logits = self.fc_final(combined).view(-1)
        bot_risk_score = torch.sigmoid(logits)
        return bot_risk_score


def load_metadata(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_onnx(model_path: str, metadata_path: str, output_path: str, opset: int = 17):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata not found: {metadata_path}")

    metadata = load_metadata(metadata_path)
    hidden = int(metadata.get("hidden", 32))
    layers = int(metadata.get("layers", 1))
    dropout = float(metadata.get("dropout", 0.4))

    model = OnnxMouseGRUModel(
        seq_size=len(SEQ_FEATURES),
        static_size=len(STATIC_FEATURES),
        hidden=hidden,
        layers=layers,
        dropout=dropout,
    )

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # dummy input. seq_len은 dynamic_axes로 동적 처리됨.
    dummy_batch = 1
    dummy_seq_len = 32
    x_seq = torch.randn(dummy_batch, dummy_seq_len, len(SEQ_FEATURES), dtype=torch.float32)
    lengths = torch.tensor([dummy_seq_len], dtype=torch.long)
    x_static = torch.randn(dummy_batch, len(STATIC_FEATURES), dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (x_seq, lengths, x_static),
        output_path,
        input_names=["x_seq", "lengths", "x_static"],
        output_names=["bot_risk_score"],
        dynamic_axes={
            "x_seq": {0: "batch", 1: "seq_len"},
            "lengths": {0: "batch"},
            "x_static": {0: "batch"},
            "bot_risk_score": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    print("ONNX export 완료")
    print(f"ONNX: {output_path}")
    print("입력: x_seq[batch, seq_len, 7], lengths[batch], x_static[batch, 10]")
    print("출력: bot_risk_score[batch]")


def main():
    parser = argparse.ArgumentParser(description="Export Mouse GRU model to ONNX")
    parser.add_argument("--model", required=True, help="PyTorch .pth path")
    parser.add_argument("--metadata", required=True, help="metadata json path")
    parser.add_argument("--output", required=True, help="output onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_onnx(args.model, args.metadata, args.output, args.opset)


if __name__ == "__main__":
    main()
