from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from flashlight.common.constants import SEQ_FEATURES, STATIC_FEATURES


class MouseFeatureNormalizer:
    def __init__(self):
        self.seq_scaler = StandardScaler()
        self.static_scaler = StandardScaler()

    def fit(self, samples):
        seq_rows, static_rows = [], []
        for s in samples:
            for feat in s.get("dynamic_features", []):
                seq_rows.append([float(feat.get(k, 0.0)) for k in SEQ_FEATURES])
            sf = s.get("static_features", {})
            static_rows.append([float(sf.get(k, 0.0)) for k in STATIC_FEATURES])

        self.seq_scaler.fit(np.array(seq_rows, dtype=np.float32))
        self.static_scaler.fit(np.array(static_rows, dtype=np.float32))
        return self

    def transform_seq(self, seq_arr: np.ndarray) -> np.ndarray:
        return self.seq_scaler.transform(seq_arr).astype(np.float32)

    def transform_static(self, static_arr: np.ndarray) -> np.ndarray:
        return self.static_scaler.transform(static_arr.reshape(1, -1)).astype(np.float32).reshape(-1)
