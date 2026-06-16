# AGAMI Liveness CAPTCHA - Server Commands

## 1. Workdir

```bash
cd /workspace/code/model/agami_liveness_modeling
```

## 2. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes runtime and export dependencies:

```text
numpy<2.0
pandas
scikit-learn
joblib
matplotlib
torch
onnx
onnxruntime
mediapipe==0.10.14
opencv-python-headless==4.10.0.84
```

Optional ML experiment packages are commented in `requirements.txt`:

```text
xgboost
lightgbm
```

## 3. System Packages for OpenCV

If `cv2` or `mediapipe` fails with `libGL.so.1`, install:

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

Verify imports:

```bash
python - <<'PY'
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
print("cv2:", cv2.__version__)
print("mediapipe:", mp.__version__)
print("numpy:", np.__version__)
print("onnxruntime:", ort.__version__)
print("OK")
PY
```

## 4. GPU Check

```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## 5. Final Dataset Check

Final selected dataset:

```text
face_clip_data.npz
```

Inspect keys and shapes:

```bash
python - <<'PY'
import numpy as np

d = np.load("face_clip_data.npz", allow_pickle=True)
for k in d.files:
    arr = d[k]
    print(k, getattr(arr, "shape", None), getattr(arr, "dtype", None))
PY
```

Expected important shapes:

```text
x_seq: (966, 16, 20)
x_static: (966, 16)
y: (966,)
```

## 6. Final Model Files

```text
runs/gru_h32_lr0005_v1/best_gru.pt
runs/gru_h32_lr0005_v1/best_gru.onnx
runs/gru_h32_lr0005_v1/best_gru_onnx_meta.json
runs/gru_h32_lr0005_v1/run_config.json
runs/gru_h32_lr0005_v1/seq_scaler.joblib
runs/gru_h32_lr0005_v1/model_results.csv
runs/gru_h32_lr0005_v1/group_results.csv
```

## 7. Required Checks Before Demo

```bash
PYTHONPATH=. python scripts/test_face_liveness_predictor.py
PYTHONPATH=. python scripts/test_face_feature_extractor.py
PYTHONPATH=. python scripts/test_full_captcha_dummy.py
```

## 8. Webcam Demo

```bash
PYTHONPATH=. python scripts/demo_webcam_full_captcha.py --camera 0
PYTHONPATH=. python scripts/demo_webcam_full_captcha.py --camera 0 --device cpu
```

If the server has no webcam, this message is expected:

```text
Cannot open webcam. If running on remote server, run this demo locally or connect webcam frames from frontend.
```

## 9. ONNX Export and Verification

Regenerate website model:

```bash
PYTHONPATH=. python scripts/export_face_liveness_onnx.py
```

ONNX interface:

```text
input name: x_seq
input shape: [batch, 16, 20]
input dtype: float32
output name: spoof_score
output shape: [batch]
threshold: 0.21000000000000002
```

The ONNX graph includes the training scaler. Website code should pass raw 20 features in the selected feature order as `float32`.

## 10. Training Command for Reproduction

The final run was trained with:

```bash
PYTHONPATH=. python src/train_gru.py \
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

## 11. Final Download Package

The zip artifact for sharing is:

```text
deliverables/agami_liveness_final_package.zip
```

Do not commit `deliverables/` to git. Recreate the zip from the final files when needed.

## 12. Artifact Policy for Git

Do not commit model, dataset, or zip artifacts to git. Keep them in the shared final package or external storage.

Required artifact paths after download:

```text
face_clip_data.npz
runs/gru_h32_lr0005_v1/best_gru.pt
runs/gru_h32_lr0005_v1/best_gru.onnx
runs/gru_h32_lr0005_v1/best_gru_onnx_meta.json
runs/gru_h32_lr0005_v1/run_config.json
runs/gru_h32_lr0005_v1/seq_scaler.joblib
runs/gru_h32_lr0005_v1/model_results.csv
runs/gru_h32_lr0005_v1/group_results.csv
```
