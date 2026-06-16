# ML Pipeline Team Handoff

This folder contains the final dataset, model artifacts, training/evaluation scripts, and performance reports.

## Main Files

- `face_clip_data.npz`: final selected dataset
- `runs/gru_h32_lr0005_v1/best_gru.pt`: final PyTorch checkpoint
- `runs/gru_h32_lr0005_v1/best_gru.onnx`: exported ONNX model
- `runs/gru_h32_lr0005_v1/run_config.json`: final run config
- `runs/gru_h32_lr0005_v1/seq_scaler.joblib`: scaler backup
- `runs/gru_h32_lr0005_v1/model_results.csv`: overall valid/test metrics
- `runs/gru_h32_lr0005_v1/group_results.csv`: group metrics
- `MODEL_PERFORMANCE.md`: summarized model performance report
- `docs/performance/`: CSV summaries and PNG visualizations

## Install

```bash
pip install -r requirements.txt
```

## Reproduce Final Training

```bash
PYTHONPATH=src python src/train_gru.py \
  --data face_clip_data.npz \
  --out runs/gru_h32_lr0005_v1 \
  --feature-mode all \
  --epochs 80 \
  --batch-size 64 \
  --hidden-size 32 \
  --dropout 0.3 \
  --lr 0.0005 \
  --weight-decay 0.0001 \
  --patience 15 \
  --device auto
```

## Export ONNX

```bash
PYTHONPATH=. python scripts/export_face_liveness_onnx.py
```

## Final Decision

The final model remains `runs/gru_h32_lr0005_v1`. `R_live_clip` is treated as a stress-test source and excluded from the main presentation benchmark.
