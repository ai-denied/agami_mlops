from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_liveness_predictor import FaceLivenessPredictor


def main():
    predictor = FaceLivenessPredictor(
        model_path="runs/gru_h32_lr0005_v1/best_gru.pt",
        device="auto",
    )

    print("model loaded")
    print("threshold:", predictor.threshold)
    print("selected_features:", predictor.selected_features)

    input_dim = len(predictor.selected_features)
    x_seq = np.zeros((16, input_dim), dtype=np.float32)

    result = predictor.predict_dict(x_seq, seq_length=16)

    print(result)


if __name__ == "__main__":
    main()
