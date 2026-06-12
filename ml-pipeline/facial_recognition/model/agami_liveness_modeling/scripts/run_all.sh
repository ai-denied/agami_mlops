#!/usr/bin/env bash
set -e

DATA=${1:-data/face_clip_data.npz}

mkdir -p runs

python src/check_dataset.py --data "$DATA" --out runs/dataset_check

python src/train_static_models.py   --data "$DATA"   --out runs/static_baselines   --threshold-strategy best_f1

python src/train_gru.py   --data "$DATA"   --out runs/gru_all   --feature-mode all   --epochs 80   --batch-size 64   --hidden-size 64   --dropout 0.3   --lr 0.001   --weight-decay 0.0001   --patience 10   --device auto

python src/train_gru.py   --data "$DATA"   --out runs/gru_no_abs   --feature-mode no_abs   --epochs 80   --batch-size 64   --hidden-size 64   --dropout 0.3   --lr 0.001   --weight-decay 0.0001   --patience 10   --device auto

python src/train_gru.py   --data "$DATA"   --out runs/gru_all_minseq5   --feature-mode all   --min-seq-len 5   --epochs 80   --batch-size 64   --hidden-size 64   --dropout 0.3   --lr 0.001   --weight-decay 0.0001   --patience 10   --device auto

echo "Done. Check runs/* outputs."
