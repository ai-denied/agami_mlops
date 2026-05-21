#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mouse_inference.py

Sentient-CAPTCHA 손전등 CAPTCHA 모델 추론 전용 파일

역할
- 학습된 GRU 모델(.pth), 정규화기(.joblib), 메타데이터(.json)를 로드한다.
- 손전등 CAPTCHA 1회 수행 로그를 입력받아 bot_risk_score를 산출한다.
- 3회 수행 점수를 누적해 allow / challenge_again / block 최종 판정을 반환한다.

필수 파일
- mouse_gru_server_final_v2.pth
- mouse_normalizer_server_final_v2.joblib
- mouse_metadata_server_final_v2.json

입력 shape
- GRU 입력(dynamic_features): (batch=1, seq_len=N, feat=7)
- MLP 입력(static_features): (batch=1, feat=10)
- lengths: (batch=1), 예: [N]

단일 샘플 추론 예시
python mouse_inference.py \
  --model "./runs/mouse_gru_final_v3_policy_tuned/mouse_gru_server_final_v2.pth" \
  --normalizer "./runs/mouse_gru_final_v3_policy_tuned/mouse_normalizer_server_final_v2.joblib" \
  --metadata "./runs/mouse_gru_final_v3_policy_tuned/mouse_metadata_server_final_v2.json" \
  --sample "./sample_captcha_log.json"

3회 누적 점수 판정 예시
python mouse_inference.py \
  --metadata "./runs/mouse_gru_final_v3_policy_tuned/mouse_metadata_server_final_v2.json" \
  --scores 0.03 0.08 0.11
"""

import argparse
import json
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence


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


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda로 지정했지만 CUDA를 사용할 수 없습니다.")
        return torch.device("cuda")

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(f"Unknown device option: {device_arg}")


class MouseFeatureNormalizer:
    """
    학습 코드에서 joblib로 저장된 normalizer를 로드하기 위해 필요한 클래스 정의다.
    학습 시 train 데이터 기준으로 fit된 StandardScaler를 그대로 사용한다.
    """

    def __init__(self):
        self.seq_scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        self.is_fitted = False

    def _get_raw_seq(self, sample: Dict) -> np.ndarray:
        seq = []

        for feat in sample.get("dynamic_features", []):
            row = [float(feat.get(k, 0.0)) for k in SEQ_FEATURES]
            seq.append(row)

        if len(seq) == 0:
            seq = [[0.0] * len(SEQ_FEATURES)]

        return np.array(seq, dtype=np.float32)

    def _get_raw_static(self, sample: Dict) -> np.ndarray:
        stat = sample.get("static_features", {})

        row = [
            float(stat.get("duration", 0.0)),
            float(stat.get("log_count", 0.0)),
            float(stat.get("total_distance", 0.0)),
            float(stat.get("straight_distance", 0.0)),
            float(stat.get("distance_ratio", 0.0)),
            float(stat.get("avg_speed", 0.0)),
            float(stat.get("max_speed", 0.0)),
            float(stat.get("speed_std", 0.0)),
            float(stat.get("direction_changes", 0.0)),
            float(stat.get("pauses", 0.0)),
        ]

        return np.array(row, dtype=np.float32)

    def transform_seq(self, sample: Dict) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Normalizer is not fitted yet.")

        seq = self._get_raw_seq(sample)
        return self.seq_scaler.transform(seq).astype(np.float32)

    def transform_static(self, sample: Dict) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Normalizer is not fitted yet.")

        static = self._get_raw_static(sample).reshape(1, -1)
        return self.static_scaler.transform(static).reshape(-1).astype(np.float32)


class MouseGRUModel(nn.Module):
    """
    학습 당시 사용한 모델 구조와 동일해야 한다.
    dynamic_features는 GRU로 처리하고, static_features는 MLP로 처리한 뒤 concat한다.
    """

    def __init__(
        self,
        seq_size: int = 7,
        static_size: int = 10,
        hidden: int = 32,
        layers: int = 1,
        dropout: float = 0.4,
    ):
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
        packed = pack_padded_sequence(
            x_seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hn = self.gru(packed)
        gru_out = hn[-1]

        static_out = self.static_mlp(x_static)
        combined = torch.cat([gru_out, static_out], dim=1)

        logits = self.fc_final(combined).view(-1)
        return logits


# 학습 코드에서 MouseGRUModelV2 이름으로 저장된 경우를 대비한 alias
MouseGRUModelV2 = MouseGRUModel


def classify_single_attempt_risk(
    bot_risk_score: float,
    low_risk_threshold: float,
    high_risk_threshold: float,
) -> str:
    if bot_risk_score < low_risk_threshold:
        return "low_risk"
    if bot_risk_score < high_risk_threshold:
        return "suspicious"
    return "high_risk"


def decide_three_attempts(
    scores: List[float],
    low_risk_threshold: float,
    high_risk_threshold: float,
    block_suspicious_count: int = 2,
    block_high_risk_count: int = 1,
    block_total_score: float = 0.25,
) -> Dict[str, Any]:
    """
    3회 손전등 CAPTCHA 수행 결과를 누적해 최종 판정한다.

    기본 정책
    - high_risk가 1회 이상이면 block
    - suspicious가 2회 이상이면 block
    - total_score가 0.25 이상이면 block
    - suspicious가 1회만 있으면 challenge_again
    - 모두 low_risk면 allow
    """

    if not scores:
        raise ValueError("scores must not be empty.")

    scores = [float(s) for s in scores]

    total_score = float(sum(scores))
    avg_score = float(total_score / len(scores))
    max_score = float(max(scores))
    min_score = float(min(scores))

    suspicious_count = int(sum(s >= low_risk_threshold for s in scores))
    high_risk_count = int(sum(s >= high_risk_threshold for s in scores))

    if (
        high_risk_count >= block_high_risk_count
        or suspicious_count >= block_suspicious_count
        or total_score >= block_total_score
    ):
        decision = "block"
    elif suspicious_count >= 1:
        decision = "challenge_again"
    else:
        decision = "allow"

    return {
        "scores": [round(float(s), 6) for s in scores],
        "total_score": round(total_score, 6),
        "avg_score": round(avg_score, 6),
        "max_score": round(max_score, 6),
        "min_score": round(min_score, 6),
        "suspicious_count": suspicious_count,
        "high_risk_count": high_risk_count,
        "low_risk_threshold": float(low_risk_threshold),
        "high_risk_threshold": float(high_risk_threshold),
        "block_suspicious_count": int(block_suspicious_count),
        "block_high_risk_count": int(block_high_risk_count),
        "block_total_score": float(block_total_score),
        "decision": decision,
    }


class MouseBotRiskDetector:
    """
    캡챠 엔진 연동용 추론 클래스.

    사용 흐름
    1. detector = MouseBotRiskDetector(model_path, normalizer_path, metadata_path)
    2. result = detector.predict_one(sample)
    3. 3회 score를 모아서 detector.decide_three_attempts(scores)
    """

    def __init__(
        self,
        model_path: str,
        normalizer_path: str,
        metadata_path: str,
        device: str = "auto",
    ):
        self.device = get_device(device)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.normalizer = joblib.load(normalizer_path)

        self.hidden = int(self.metadata.get("hidden", 32))
        self.layers = int(self.metadata.get("layers", 1))
        self.dropout = float(self.metadata.get("dropout", 0.4))

        self.low_risk_threshold = float(self.metadata.get("low_risk_threshold", 0.05))
        self.high_risk_threshold = float(self.metadata.get("high_risk_threshold", 0.60))

        policy = self.metadata.get("three_attempt_policy", {})
        self.three_attempt_policy = {
            "attempts": int(policy.get("attempts", 3)),
            "block_suspicious_count": int(policy.get("block_suspicious_count", 2)),
            "block_high_risk_count": int(policy.get("block_high_risk_count", 1)),
            "block_total_score": float(policy.get("block_total_score", 0.25)),
        }

        self.model = MouseGRUModel(
            seq_size=len(SEQ_FEATURES),
            static_size=len(STATIC_FEATURES),
            hidden=self.hidden,
            layers=self.layers,
            dropout=self.dropout,
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_one(self, sample: Dict) -> Dict[str, Any]:
        """
        손전등 CAPTCHA 1회 수행 로그를 입력받아 bot_risk_score를 산출한다.

        sample 구조
        {
          "dynamic_features": [
            {
              "dx": ..., "dy": ..., "dt": ...,
              "distance": ..., "velocity": ...,
              "acceleration": ..., "angle_change": ...
            }
          ],
          "static_features": {
            "duration": ...,
            "log_count": ...,
            "total_distance": ...,
            "straight_distance": ...,
            "distance_ratio": ...,
            "avg_speed": ...,
            "max_speed": ...,
            "speed_std": ...,
            "direction_changes": ...,
            "pauses": ...
          }
        }
        """

        seq = self.normalizer.transform_seq(sample)
        static = self.normalizer.transform_static(sample)

        # 단일 요청 기준 shape
        # x_seq: (batch=1, seq_len=N, feat=7)
        # x_static: (batch=1, feat=10)
        # lengths: (batch=1), 예: [N]
        x_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(seq)], dtype=torch.long).to(self.device)
        x_static = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self.model(x_seq, lengths, x_static)
        bot_risk_score = torch.sigmoid(logits)[0].detach().cpu().item()

        risk_band = classify_single_attempt_risk(
            bot_risk_score,
            low_risk_threshold=self.low_risk_threshold,
            high_risk_threshold=self.high_risk_threshold,
        )

        return {
            "bot_risk_score": float(bot_risk_score),
            "risk_band": risk_band,
            "low_risk_threshold": self.low_risk_threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "input_shape": {
                "gru_input": [1, int(len(seq)), len(SEQ_FEATURES)],
                "static_input": [1, len(STATIC_FEATURES)],
                "lengths": [int(len(seq))],
            },
        }

    def decide_three_attempts(self, scores: List[float]) -> Dict[str, Any]:
        return decide_three_attempts(
            scores=scores,
            low_risk_threshold=self.low_risk_threshold,
            high_risk_threshold=self.high_risk_threshold,
            block_suspicious_count=self.three_attempt_policy["block_suspicious_count"],
            block_high_risk_count=self.three_attempt_policy["block_high_risk_count"],
            block_total_score=self.three_attempt_policy["block_total_score"],
        )


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Sentient-CAPTCHA mouse model inference")

    parser.add_argument("--model", type=str, default=None, help="모델 가중치 .pth 경로")
    parser.add_argument("--normalizer", type=str, default=None, help="정규화기 .joblib 경로")
    parser.add_argument("--metadata", type=str, required=True, help="모델 메타데이터 .json 경로")
    parser.add_argument("--sample", type=str, default=None, help="CAPTCHA 1회 수행 로그 JSON 경로")
    parser.add_argument("--scores", type=float, nargs="*", default=None, help="3회 누적 판정용 bot_risk_score 목록")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    # 점수만 받아 3회 누적 판정하는 모드
    if args.scores is not None and len(args.scores) > 0:
        metadata = load_json(args.metadata)

        low_risk_threshold = float(metadata.get("low_risk_threshold", 0.05))
        high_risk_threshold = float(metadata.get("high_risk_threshold", 0.60))

        policy = metadata.get("three_attempt_policy", {})

        result = decide_three_attempts(
            scores=args.scores,
            low_risk_threshold=low_risk_threshold,
            high_risk_threshold=high_risk_threshold,
            block_suspicious_count=int(policy.get("block_suspicious_count", 2)),
            block_high_risk_count=int(policy.get("block_high_risk_count", 1)),
            block_total_score=float(policy.get("block_total_score", 0.25)),
        )

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 샘플 추론 모드
    if not args.model or not args.normalizer or not args.sample:
        raise ValueError(
            "샘플 추론을 위해서는 --model, --normalizer, --metadata, --sample이 모두 필요합니다. "
            "3회 누적 점수 판정만 하려면 --scores를 사용하세요."
        )

    detector = MouseBotRiskDetector(
        model_path=args.model,
        normalizer_path=args.normalizer,
        metadata_path=args.metadata,
        device=args.device,
    )

    sample = load_json(args.sample)
    result = detector.predict_one(sample)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
