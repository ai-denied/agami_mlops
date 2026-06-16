from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_feature_extractor import FaceFeatureExtractor, SELECTED_FEATURES


def main() -> None:
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(4)]
    extractor = FaceFeatureExtractor(target_frames=16)
    try:
        x_seq, seq_length, info = extractor.extract_from_frames(frames)
    finally:
        extractor.close()

    print("x_seq shape:", x_seq.shape)
    print("seq_length:", seq_length)
    print("info:", info)

    assert x_seq.shape == (16, 20)
    assert x_seq.dtype == np.float32
    assert len(SELECTED_FEATURES) == 20
    assert 1 <= seq_length <= 16

    print("face feature extractor shape test passed")


if __name__ == "__main__":
    main()
