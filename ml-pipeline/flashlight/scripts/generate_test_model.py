#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트용 model-store 파일 생성 스크립트

실제 학습된 ONNX 파일이 없을 때 API 동작 검증 목적으로만 사용한다.
실제 운영에서는 package_for_captcha_engine.py → promote_model.py 사용.

출력:
  model-store/flashlight/current/mouse_gru.onnx
  model-store/flashlight/current/normalizer.json
  model-store/flashlight/current/metadata.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OUT_DIR = os.path.join(_ML_PIPELINE_ROOT, "model-store", "flashlight", "current")

SEQ_FEATURES = ["dx", "dy", "dt", "distance", "velocity", "acceleration", "angle_change"]
STATIC_FEATURES = [
    "duration", "log_count", "total_distance", "straight_distance",
    "distance_ratio", "avg_speed", "max_speed", "speed_std",
    "direction_changes", "pauses",
]
SEQ_DIM = len(SEQ_FEATURES)      # 7
STATIC_DIM = len(STATIC_FEATURES)  # 10
HIDDEN = 32


# ── 모델 정의 (mouse_gru.py와 동일) ────────────────────────────────────────────

class MouseGRUModelV2(nn.Module):
    def __init__(self, seq_size=SEQ_DIM, static_size=STATIC_DIM, hidden=HIDDEN, layers=1, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(seq_size, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_size, 32), nn.LayerNorm(32), nn.PReLU(), nn.Dropout(dropout),
        )
        self.fc_final = nn.Sequential(
            nn.Linear(hidden + 32, 64), nn.LayerNorm(64), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, lengths, x_static):
        packed = pack_padded_sequence(x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.gru(packed)
        gru_out = hn[-1]
        static_out = self.static_mlp(x_static)
        combined = torch.cat([gru_out, static_out], dim=1)
        logits = self.fc_final(combined).view(-1)
        return torch.sigmoid(logits)


# ── ONNX Export ───────────────────────────────────────────────────────────────

class ExportWrapper(nn.Module):
    """pack_padded_sequence는 dynamic shape ONNX export와 호환되지 않아 GRU를 직접 처리한다."""

    def __init__(self, model: MouseGRUModelV2):
        super().__init__()
        self.gru = model.gru
        self.static_mlp = model.static_mlp
        self.fc_final = model.fc_final

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor, x_static: torch.Tensor):
        # lengths는 ONNX export 시 시그니처 유지를 위해 받되, 여기선 전체 시퀀스를 그대로 사용
        gru_out, _ = self.gru(x_seq)
        # lengths로 마지막 유효 스텝만 선택
        idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, gru_out.size(2))
        last_out = gru_out.gather(1, idx).squeeze(1)
        static_out = self.static_mlp(x_static)
        combined = torch.cat([last_out, static_out], dim=1)
        score = torch.sigmoid(self.fc_final(combined).view(-1))
        return score


def export_onnx(out_path: str) -> None:
    model = MouseGRUModelV2()
    model.eval()
    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy_seq = torch.randn(1, 32, SEQ_DIM)
    dummy_lengths = torch.tensor([32], dtype=torch.int64)
    dummy_static = torch.randn(1, STATIC_DIM)

    torch.onnx.export(
        wrapper,
        (dummy_seq, dummy_lengths, dummy_static),
        out_path,
        input_names=["x_seq", "lengths", "x_static"],
        output_names=["bot_risk_score"],
        dynamic_axes={
            "x_seq":    {0: "batch", 1: "seq_len"},
            "lengths":  {0: "batch"},
            "x_static": {0: "batch"},
            "bot_risk_score": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[OK] ONNX 저장: {out_path}")


# ── normalizer.json ───────────────────────────────────────────────────────────

def build_normalizer_json(out_path: str) -> None:
    rng = np.random.default_rng(42)
    normalizer = {
        "seq_features": SEQ_FEATURES,
        "static_features": STATIC_FEATURES,
        "seq_scaler": {
            "mean":  rng.uniform(-1, 1, SEQ_DIM).tolist(),
            "scale": rng.uniform(0.5, 2.0, SEQ_DIM).tolist(),
            "var":   rng.uniform(0.1, 1.0, SEQ_DIM).tolist(),
        },
        "static_scaler": {
            "mean":  rng.uniform(-1, 1, STATIC_DIM).tolist(),
            "scale": rng.uniform(0.5, 2.0, STATIC_DIM).tolist(),
            "var":   rng.uniform(0.1, 1.0, STATIC_DIM).tolist(),
        },
        "threshold_policy": {
            "low_risk_threshold":     0.05,
            "high_risk_threshold":    0.6,
            "block_suspicious_count": 2,
            "block_high_risk_count":  1,
            "block_total_score":      0.25,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalizer, f, indent=2)
    print(f"[OK] normalizer.json 저장: {out_path}")


# ── metadata.json ─────────────────────────────────────────────────────────────

def build_metadata_json(out_path: str) -> None:
    metadata = {
        "model_name": "MouseGRUModelV2",
        "version": "test_v1",
        "packaged_at": "2026-06-12",
        "promoted_at": "2026-06-12 00:00:00",
        "source_run": "generate_test_model.py",
        "model_arch": {
            "type": "GRU",
            "seq_input_dim": SEQ_DIM,
            "static_input_dim": STATIC_DIM,
            "hidden": HIDDEN,
            "layers": 1,
            "dropout": 0.4,
        },
        "onnx_spec": {
            "opset": 17,
            "inputs": [
                "x_seq [batch, seq_len, 7]",
                "lengths [batch]",
                "x_static [batch, 10]",
            ],
            "output": "bot_risk_score [batch]",
            "score_name": "bot_risk_score",
            "label_rule": "0=human, 1=bot",
        },
        "threshold_policy": {
            "low_risk_threshold":     0.05,
            "high_risk_threshold":    0.6,
            "block_suspicious_count": 2,
            "block_high_risk_count":  1,
            "block_total_score":      0.25,
        },
        "performance": {
            "note": "테스트용 랜덤 가중치 — 성능 수치 없음"
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] metadata.json 저장: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_OUT_DIR, exist_ok=True)
    print(f"출력 경로: {_OUT_DIR}\n")
    export_onnx(os.path.join(_OUT_DIR, "mouse_gru.onnx"))
    build_normalizer_json(os.path.join(_OUT_DIR, "normalizer.json"))
    build_metadata_json(os.path.join(_OUT_DIR, "metadata.json"))
    print("\n[완료] model-store/flashlight/current/ 준비됨")


if __name__ == "__main__":
    main()
