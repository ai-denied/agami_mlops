import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from flashlight.data.dataset import MouseDataset, collate_fn


def make_loader(
    samples: List[Dict],
    normalizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seq_noise_std: float,
    static_noise_std: float,
    training: bool,
    device: Optional[torch.device] = None,
) -> data.DataLoader:
    pin = device is not None and device.type == "cuda"
    return data.DataLoader(
        MouseDataset(samples, normalizer, seq_noise_std, static_noise_std, training),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
    )


def get_pos_weight(samples: List[Dict], device: torch.device) -> torch.Tensor:
    labels = np.array([int(s.get("label", 0)) for s in samples])
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0:
        return torch.tensor([1.0], device=device)
    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def check_data_path(data_path: str) -> None:
    if not os.path.exists(data_path):
        print(f"\n[에러] 데이터 파일을 찾을 수 없습니다.")
        print(f"입력한 경로: {data_path}")
        raise FileNotFoundError(data_path)


def diagnose_dataset(all_data: List[Dict], out_dir: str) -> None:
    rows = [
        {
            "label": d.get("label"),
            "user_id": d.get("user_id"),
            "bot_type": d.get("bot_type"),
            "image_id": d.get("image_id"),
            "original_file": d.get("original_file"),
            "source_file": d.get("source_file"),
        }
        for d in all_data
    ]
    df = pd.DataFrame(rows)
    print("\n[데이터 진단]")
    print("전체 데이터 수:", len(df))
    print("label 분포:", df["label"].value_counts(dropna=False).to_dict())
    print("user_id 고유 개수:", df["user_id"].nunique(dropna=False))
    print("bot_type 고유 개수:", df["bot_type"].nunique(dropna=False))
    print("image_id 고유 개수:", df["image_id"].nunique(dropna=False))
    diag_path = os.path.join(out_dir, "dataset_diagnostics.csv")
    df.to_csv(diag_path, index=False)
    print(f"데이터 진단 CSV 저장: {diag_path}")


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device, grad_clip: float) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_seq, lengths, x_static)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        n = y.size(0)
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss, total_count = 0.0, 0
    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x_seq, lengths, x_static)
        loss = criterion(logits, y)
        n = y.size(0)
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_risk_scores(model, loader, device: torch.device):
    model.eval()
    all_scores: List[float] = []
    all_labels: List[float] = []
    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        logits = model(x_seq, lengths, x_static)
        scores = torch.sigmoid(logits)
        all_scores.extend(scores.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())
    return np.array(all_scores, dtype=np.float32), np.array(all_labels, dtype=np.float32)
