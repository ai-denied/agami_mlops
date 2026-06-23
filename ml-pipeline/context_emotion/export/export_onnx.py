"""Export a trained EmotionClassifier checkpoint to ONNX.

Usage:
    python -m context_emotion.export.export_onnx \
        --checkpoint runs/emotion_classifier/best.pt \
        --out runs/emotion_classifier/model.onnx
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.model.emotion_classifier import EmotionClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = EmotionClassifier(freeze_backbone=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
