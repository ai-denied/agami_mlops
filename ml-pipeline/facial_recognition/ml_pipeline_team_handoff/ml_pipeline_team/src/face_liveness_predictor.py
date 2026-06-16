from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_layers=1, dropout=0.3):
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

    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()

        packed = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        _, h_n = self.gru(packed)
        h_last = h_n[-1]
        logits = self.head(h_last).squeeze(-1)

        return logits


class FaceLivenessPredictor:
    """
    GRU 얼굴 liveness 모델 추론용 wrapper.

    입력:
    - x_seq: shape (16, feature_dim) 또는 (1, 16, feature_dim)
    - seq_length: 실제 유효 프레임 수

    출력:
    - spoof_score: 0~1 사이 값
      높을수록 spoof 가능성이 높음
    """

    def __init__(
        self,
        model_path: str = "runs/gru_h32_lr0005_v1/best_gru.pt",
        device: str = "auto",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {self.model_path}")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.threshold = float(checkpoint.get("threshold", 0.5))
        self.selected_features = checkpoint.get("selected_features", None)
        self.selected_idx = checkpoint.get("selected_idx", None)

        self.scaler_mean = checkpoint.get("scaler_mean", None)
        self.scaler_scale = checkpoint.get("scaler_scale", None)

        input_dim = int(checkpoint["input_dim"])
        hidden_size = int(checkpoint.get("hidden_size", 32))
        num_layers = int(checkpoint.get("num_layers", 1))
        dropout = float(checkpoint.get("dropout", 0.3))

        self.model = GRUClassifier(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _scale_sequence(self, x_seq: np.ndarray, seq_length: int) -> np.ndarray:
        x_seq = x_seq.astype(np.float32).copy()

        if self.scaler_mean is None or self.scaler_scale is None:
            return x_seq

        mean = np.asarray(self.scaler_mean, dtype=np.float32)
        scale = np.asarray(self.scaler_scale, dtype=np.float32)

        scale = np.where(scale == 0, 1.0, scale)

        x_seq[:seq_length] = (x_seq[:seq_length] - mean) / scale

        return x_seq

    @torch.no_grad()
    def predict_score(
        self,
        x_seq: np.ndarray,
        seq_length: Optional[int] = None,
    ) -> float:
        if x_seq.ndim == 3:
            if x_seq.shape[0] != 1:
                raise ValueError("현재 predict_score는 batch size 1만 지원합니다.")
            x_seq = x_seq[0]

        if x_seq.ndim != 2:
            raise ValueError(f"x_seq shape가 잘못되었습니다: {x_seq.shape}")

        if seq_length is None:
            seq_length = x_seq.shape[0]

        seq_length = int(seq_length)

        if seq_length <= 0:
            raise ValueError("seq_length는 1 이상이어야 합니다.")

        x_seq = self._scale_sequence(x_seq, seq_length)

        x_tensor = torch.tensor(
            x_seq[None, :, :],
            dtype=torch.float32,
            device=self.device,
        )

        length_tensor = torch.tensor(
            [seq_length],
            dtype=torch.long,
            device=self.device,
        )

        logits = self.model(x_tensor, length_tensor)
        prob = torch.sigmoid(logits).detach().cpu().numpy()[0]

        return float(prob)

    def predict_label(
        self,
        x_seq: np.ndarray,
        seq_length: Optional[int] = None,
    ) -> int:
        spoof_score = self.predict_score(x_seq, seq_length)
        return int(spoof_score >= self.threshold)

    def predict_dict(
        self,
        x_seq: np.ndarray,
        seq_length: Optional[int] = None,
    ) -> dict:
        spoof_score = self.predict_score(x_seq, seq_length)
        pred_label = int(spoof_score >= self.threshold)

        return {
            "spoof_score": spoof_score,
            "threshold": self.threshold,
            "pred_label": pred_label,
            "pred_name": "spoof" if pred_label == 1 else "live",
            "device": self.device,
            "selected_features": self.selected_features,
        }