# AGAMI Final Liveness CAPTCHA Package

This document summarizes the files that should be shared for the final website/demo handoff.

## Install

```bash
pip install -r requirements.txt
```

If OpenCV fails on Linux server:

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

## Included Final Assets

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

## Runtime Code

```text
src/face_feature_extractor.py
src/face_liveness_predictor.py
src/hand_gesture_detector.py
src/captcha_decision.py
scripts/demo_webcam_full_captcha.py
scripts/export_face_liveness_onnx.py
```

## ONNX Interface

```text
model: runs/gru_h32_lr0005_v1/best_gru.onnx
input name: x_seq
input shape: [batch, 16, 20]
input dtype: float32
output name: spoof_score
output shape: [batch]
threshold: 0.21000000000000002
```

Feature order:

```text
ear, mar, smile_w, nose_x, nose_y, cx, cy, roll, yaw, pitch,
nose_dx, nose_dy, center_dx, center_dy, nose_speed,
ear_velocity, mar_velocity, yaw_velocity, pitch_velocity, roll_velocity
```

The ONNX graph embeds the training scaler. Pass raw feature values in the order above.

## Checks

```bash
PYTHONPATH=. python scripts/test_face_liveness_predictor.py
PYTHONPATH=. python scripts/test_face_feature_extractor.py
PYTHONPATH=. python scripts/test_full_captcha_dummy.py
```

## Webcam Demo

```bash
PYTHONPATH=. python scripts/demo_webcam_full_captcha.py --camera 0
```

Run the webcam demo locally if the remote server does not expose a physical camera.
