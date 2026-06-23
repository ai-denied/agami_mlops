#!/usr/bin/env python3
"""context_emotion 14-class emotion classifier — 학습 진입점.

실행 예시:
    python -m context_emotion.scripts.train_emotion_classifier \
        --train-csv /workspace/data/context_emotion/processed/context_emotion_train_dataset_v2.csv \
        --image-root /workspace/data/context_emotion \
        --out-dir ./runs/emotion_classifier_v1
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.common.device import get_device, set_seed  # noqa: E402
from context_emotion.evaluation.metrics import print_eval_report  # noqa: E402
from context_emotion.model.emotion_classifier import EmotionClassifier  # noqa: E402
from context_emotion.training.split import class_weights, load_splits  # noqa: E402
from context_emotion.training.train_loop import evaluate, make_loader, train_one_epoch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--image-root", required=True,
                     help="image_path 컬럼이 상대경로로 잡고 있는 기준 디렉터리 "
                          "(예: /workspace/data/context_emotion) - docs/SCHEMA_NOTES_v2.md 참고")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    splits = load_splits(args.train_csv)
    train_loader = make_loader(splits["train"], args.image_root, args.batch_size, shuffle=True,
                                num_workers=args.num_workers, training=True)
    val_loader = make_loader(splits["val"], args.image_root, args.batch_size, shuffle=False,
                              num_workers=args.num_workers, training=False)
    test_loader = make_loader(splits["test"], args.image_root, args.batch_size, shuffle=False,
                               num_workers=args.num_workers, training=False)

    model = EmotionClassifier(freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits["train"], device))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print("\n[test set report]")
    print_eval_report(test_labels, test_preds)


if __name__ == "__main__":
    main()
