from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from flashlight.common.constants import SEQ_FEATURES, STATIC_FEATURES


class MouseDataset(Dataset):
    def __init__(self, samples: List[Dict], normalizer, seq_noise_std=0.0, static_noise_std=0.0, training=False):
        self.samples = samples
        self.normalizer = normalizer
        self.seq_noise_std = seq_noise_std
        self.static_noise_std = static_noise_std
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        seq = [[float(f.get(k, 0.0)) for k in SEQ_FEATURES] for f in s.get("dynamic_features", [])]
        if not seq:
            seq = [[0.0] * len(SEQ_FEATURES)]
        seq_arr = np.array(seq, dtype=np.float32)
        seq_scaled = self.normalizer.transform_seq(seq_arr)

        if self.training and self.seq_noise_std > 0:
            seq_scaled += np.random.normal(0, self.seq_noise_std, seq_scaled.shape).astype(np.float32)

        sf = s.get("static_features", {})
        static_arr = np.array([float(sf.get(k, 0.0)) for k in STATIC_FEATURES], dtype=np.float32)
        static_scaled = self.normalizer.transform_static(static_arr)

        if self.training and self.static_noise_std > 0:
            static_scaled += np.random.normal(0, self.static_noise_std, static_scaled.shape).astype(np.float32)

        label = float(s.get("label", 0))
        return torch.tensor(seq_scaled, dtype=torch.float32), torch.tensor(static_scaled, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    seqs, statics, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int64)
    max_len = max(s.shape[0] for s in seqs)
    seq_dim = seqs[0].shape[1]

    x_seq = torch.zeros(len(seqs), max_len, seq_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        x_seq[i, :s.shape[0]] = s

    x_static = torch.stack(statics)
    y = torch.stack(labels)
    return x_seq, lengths, x_static, y
