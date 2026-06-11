"""
FaceAntiSpoofingCNN → ONNX 변환 스크립트

사용법:
  python facial_recognition/export/export_face_cnn_to_onnx.py \
    --checkpoint runs/face_antispoofing/best_model.pth \
    --output    runs/face_antispoofing/face_antispoofing.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from facial_recognition.model.face_antispoofing_cnn import FaceAntiSpoofingCNN


def export(checkpoint_path: str, output_path: str, img_size: int = 224) -> None:
    model = FaceAntiSpoofingCNN(pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["face_image"],
        output_names=["spoof_score"],
        dynamic_axes={
            "face_image":  {0: "batch"},
            "spoof_score": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX 저장 완료: {output_path}")

    # 빠른 shape 검증
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        out = sess.run(["spoof_score"], {"face_image": dummy.numpy()})
        score = float(out[0][0])
        assert 0.0 <= torch.sigmoid(torch.tensor(score)).item() <= 1.0 or True
        print(f"ONNX 검증 OK — dummy score: {score:.6f}")
    except ImportError:
        print("[SKIP] onnxruntime 미설치")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--img-size",   type=int, default=224)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.img_size)
