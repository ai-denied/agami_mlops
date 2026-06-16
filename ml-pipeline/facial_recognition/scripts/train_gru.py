"""
GRU 얼굴 활성도 모델 학습 엔트리포인트.

사용법:
  python -m facial_recognition.scripts.train_gru \\
    --data face_clip_data.npz \\
    --out  runs/gru_h32_lr0005_v1 \\
    --feature-mode all \\
    --epochs 80 \\
    --hidden-size 32 \\
    --lr 0.0005
"""

from __future__ import annotations

import argparse
import sys

from facial_recognition.training.train_gru import train


def _parse_args():
    p = argparse.ArgumentParser(description="GRU 얼굴 활성도 모델 학습")
    p.add_argument("--data",               required=True,  help="face_clip_data.npz 경로")
    p.add_argument("--out",                required=True,  help="출력 디렉토리")
    p.add_argument("--feature-mode",       default="all",
                   choices=["all", "no_abs", "motion_only", "eye_mouth", "head"])
    p.add_argument("--min-seq-len",        type=int,   default=1)
    p.add_argument("--min-face-rate",      type=float, default=0.0)
    p.add_argument("--epochs",             type=int,   default=80)
    p.add_argument("--batch-size",         type=int,   default=64)
    p.add_argument("--hidden-size",        type=int,   default=32)
    p.add_argument("--num-layers",         type=int,   default=1)
    p.add_argument("--dropout",            type=float, default=0.3)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--weight-decay",       type=float, default=1e-4)
    p.add_argument("--patience",           type=int,   default=15)
    p.add_argument("--threshold-strategy", default="best_f1",
                   choices=["best_f1", "eer_like", "low_far", "default"])
    p.add_argument("--max-frr",            type=float, default=None)
    p.add_argument("--device",             default="auto")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--num-threads",        type=int,   default=1)
    return p.parse_args()


def main():
    args = _parse_args()
    out  = train(
        data_path=args.data,
        out_dir=args.out,
        feature_mode=args.feature_mode,
        min_seq_len=args.min_seq_len,
        min_face_rate=args.min_face_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        threshold_strategy=args.threshold_strategy,
        max_frr=args.max_frr,
        device=args.device,
        seed=args.seed,
        num_threads=args.num_threads,
    )
    print(f"\n학습 완료: {out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
