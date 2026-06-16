"""
이미지 기반 얼굴 위변조 탐지 데이터셋 (CNN 학습용).

samples_manifest.jsonl 포맷:
  {"file": "relative/path.jpg", "label": 0, "split": "train", "subject_id": "S001"}

label: 0=real, 1=spoof
split: "train" | "valid" | "test"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_manifest(manifest_path: str) -> List[dict]:
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def split_manifest(samples: List[dict], split: str) -> List[dict]:
    return [s for s in samples if s.get("split") == split]


def get_pos_weight(samples: List[dict], device: torch.device) -> torch.Tensor:
    n_neg = sum(1 for s in samples if s.get("label") == 0)
    n_pos = sum(1 for s in samples if s.get("label") == 1)
    n_neg = max(n_neg, 1)
    n_pos = max(n_pos, 1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)


class FaceAntiSpoofDataset(Dataset):
    """JSONL 매니페스트 기반 얼굴 위변조 이미지 데이터셋."""

    def __init__(
        self,
        samples: List[dict],
        img_dir: str,
        img_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.samples  = samples
        self.img_dir  = Path(img_dir)
        self.img_size = img_size
        self.transform = _TRAIN_TRANSFORM if augment else _EVAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = self.img_dir / sample["file"]
        label    = float(sample["label"])

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32)
