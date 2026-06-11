"""
Face Anti-Spoofing CNN

MobileNetV2 (ImageNet pretrained) 백본 위에 이진 분류 헤드를 얹은 경량 모델.

- 입력: (B, 3, 224, 224) float32 — ImageNet 정규화 적용된 RGB
- 출력: (B,) float32 — 로짓 (sigmoid 전)  →  score = sigmoid(logit)
- score ≈ 1.0 → spoof,  score ≈ 0.0 → real

ONNX export 시 output 이름: "spoof_score"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class FaceAntiSpoofingCNN(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)  # (B,)
