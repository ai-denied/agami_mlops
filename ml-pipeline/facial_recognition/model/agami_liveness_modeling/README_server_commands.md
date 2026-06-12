# Facial Liveness Modeling - Server Run Guide

서버 작업 경로:

```bash
cd /home/ubuntu/agami-mlops/ml-pipeline/facial_recognition/model/trainmodel
```

## 1. 파일 배치

최종 데이터셋을 아래 경로에 둡니다.

```bash
mkdir -p data
# face_clip_data(1).npz 파일을 data/face_clip_data.npz 로 업로드 또는 복사
ls -lh data/face_clip_data.npz
```

## 2. 환경 설치

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# XGBoost/LightGBM까지 돌릴 경우
pip install xgboost lightgbm
```

GPU 확인:

```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## 3. 데이터 검증

```bash
python src/check_dataset.py   --data data/face_clip_data.npz   --out runs/dataset_check
```

## 4. SVM / RandomForest / XGBoost / LightGBM baseline

```bash
python src/train_static_models.py   --data data/face_clip_data.npz   --out runs/static_baselines   --threshold-strategy best_f1
```

XGBoost/LightGBM까지 포함하려면:

```bash
python src/train_static_models.py \
  --data data/face_clip_data.npz \
  --out runs/static_baselines_boosting \
  --threshold-strategy best_f1 \
  --include-boosting
```

## 5. GRU 시계열 모델

전체 feature 사용:

```bash
python src/train_gru.py   --data data/face_clip_data.npz   --out runs/gru_all   --feature-mode all   --epochs 80   --batch-size 64   --hidden-size 64   --dropout 0.3   --lr 0.001   --weight-decay 0.0001   --patience 10   --device auto
```

절대좌표 제거 버전:

```bash
python src/train_gru.py   --data data/face_clip_data.npz   --out runs/gru_no_abs   --feature-mode no_abs   --epochs 80   --batch-size 64   --hidden-size 64   --dropout 0.3   --lr 0.001   --weight-decay 0.0001   --patience 10   --device auto
```

짧은 시퀀스 제거 실험:

```bash
python src/train_gru.py   --data data/face_clip_data.npz   --out runs/gru_all_minseq5   --feature-mode all   --min-seq-len 5   --epochs 80   --batch-size 64   --hidden-size 64   --dropout 0.3   --lr 0.001   --weight-decay 0.0001   --patience 10   --device auto
```

## 6. 추천 실행 순서

```bash
bash scripts/run_all.sh
```

## 7. 보고서에 적을 핵심

- `x_static` 기반 전통 ML baseline과 `x_seq` 기반 GRU 시계열 모델을 동일 split에서 비교했다.
- 모델 학습 전 NaN/Inf, label 분포, split 분포, seq_length, padding, face_detect_rate, source 편향, 데이터 누수 가능성을 검증했다.
- 최종 모델은 accuracy뿐 아니라 F1, ROC-AUC, PR-AUC, FAR, FRR, attack_type별 성능, source_group별 성능을 함께 비교하여 선정한다.
