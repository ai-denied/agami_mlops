# CAPTCHA Engine Team Handoff

This folder contains the runtime files needed for website/webcam CAPTCHA integration.

## Main Files

- `runs/gru_h32_lr0005_v1/best_gru.onnx`: ONNX model for website/ONNX Runtime integration
- `runs/gru_h32_lr0005_v1/best_gru.pt`: PyTorch fallback model
- `runs/gru_h32_lr0005_v1/best_gru_onnx_meta.json`: ONNX input/output metadata
- `src/face_feature_extractor.py`: webcam frame -> `(16, 20)` face feature sequence
- `src/hand_gesture_detector.py`: MediaPipe Hands mission detector
- `src/captcha_decision.py`: 3-round PASS / RETRY / FAIL decision logic
- `scripts/demo_webcam_full_captcha.py`: full webcam demo

## Install

```bash
pip install -r requirements.txt
```

If OpenCV fails on Linux:

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

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

Remote server pods may not expose a webcam. Run the demo locally or forward frontend webcam frames to backend.

## ONNX Interface

```text
input name: x_seq
input shape: [batch, 16, 20]
input dtype: float32
output name: spoof_score
output shape: [batch]
threshold: 0.21000000000000002
```

The ONNX graph includes the training scaler. Pass raw face features in the selected feature order documented in `best_gru_onnx_meta.json`.
