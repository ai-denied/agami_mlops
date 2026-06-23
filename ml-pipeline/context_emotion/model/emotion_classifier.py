"""14-class emotion classifier: pretrained CNN backbone + linear head.

Training from scratch isn't realistic at this dataset size (~5.4k train
images across 14 classes, see docs/context_emotion_label_distribution_v2.md)
so the backbone starts from ImageNet weights and only the head (plus,
optionally, the last backbone block) gets fine-tuned.
"""
import torch.nn as nn
import torchvision.models as models

from context_emotion.common.constants import EMOTION_CLASSES

NUM_CLASSES = len(EMOTION_CLASSES)


class EmotionClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3, freeze_backbone: bool = True):
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = backbone.classifier[1].in_features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)
