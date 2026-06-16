"""
GRU 기반 얼굴 활성도(liveness) 분류 모델.

Input : (batch, seq_len, input_dim) — 패딩된 프레임 시퀀스
Lengths: (batch,) int64 — 실제 유효 프레임 수
Output : (batch,) float — 로짓 (sigmoid 전)

score = sigmoid(logit): 0에 가까울수록 live, 1에 가까울수록 spoof.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class FaceLivenessGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_last = h_n[-1]
        return self.head(h_last).squeeze(-1)


class FaceLivenessOnnxModule(nn.Module):
    """ONNX 익스포트용 래퍼. 스케일러를 그래프에 내장하고 sigmoid를 적용한다.

    Input  : x_seq  (batch, 16, 20)  — raw unscaled features
    Output : spoof_score (batch,)    — 0~1 float
    """

    def __init__(self, gru: FaceLivenessGRU, scaler_mean, scaler_scale) -> None:
        super().__init__()
        self.gru  = gru.gru
        self.head = gru.head

        import numpy as np
        mean_t  = torch.tensor(np.asarray(scaler_mean,  dtype="float32")).view(1, 1, -1)
        scale_t = torch.tensor(np.asarray(scaler_scale, dtype="float32")).view(1, 1, -1)
        scale_t = torch.where(scale_t == 0, torch.ones_like(scale_t), scale_t)
        self.register_buffer("scaler_mean",  mean_t)
        self.register_buffer("scaler_scale", scale_t)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_scaled = (x_seq - self.scaler_mean) / self.scaler_scale
        _, h_n   = self.gru(x_scaled)
        h_last   = h_n[-1]
        logits   = self.head(h_last).squeeze(-1)
        return torch.sigmoid(logits)
