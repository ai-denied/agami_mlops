# AGAMI Liveness CAPTCHA Modeling

얼굴 liveness 모델과 손가락 랜덤 미션을 결합한 발표/시현용 CAPTCHA 프로젝트다. 얼굴 모델은 단독 최종 판정기가 아니라 `spoof_score`를 제공하고, 최종 결과는 얼굴/손 미션 성공 여부와 3라운드 누적 위험도로 `PASS`, `RETRY`, `FAIL`을 결정한다.

## Final Selection

최종 모델은 아래 데이터셋과 run을 기준으로 한다.

```text
Dataset: face_clip_data.npz
Model run: runs/gru_h32_lr0005_v1
PyTorch model: runs/gru_h32_lr0005_v1/best_gru.pt
ONNX model: runs/gru_h32_lr0005_v1/best_gru.onnx
Threshold: 0.21000000000000002
```

`face_clip_data.npz` 구성:

```text
x_seq: (966, 16, 20)
x_static: (966, 16)
y: (966,)
splits: train 614, valid 169, test 183
attack_types: live 548, print 141, replay 277
```

최종 모델은 GRU 기반 시계열 모델이며, 각 샘플은 16프레임, 프레임당 20개 얼굴 feature를 사용한다.

## Install

```bash
pip install -r requirements.txt
```

서버에서 `cv2` import 시 `libGL.so.1` 오류가 나면 아래 패키지를 설치한다.

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

## Runtime Flow

```text
webcam frames
-> MediaPipe FaceMesh face feature extraction
-> FaceLivenessPredictor / ONNX model spoof_score
-> MediaPipe Hands hand mission detection
-> 3 MissionRound results
-> decide_three_round_captcha
-> PASS / RETRY / FAIL
```

## Important Files

```text
src/face_feature_extractor.py      # OpenCV frame -> FaceMesh -> (16, 20) feature sequence
src/face_liveness_predictor.py     # PyTorch best_gru.pt inference wrapper
src/hand_gesture_detector.py       # MediaPipe Hands rule-based mission detector
src/captcha_decision.py            # 3-round risk decision logic

scripts/demo_webcam_full_captcha.py      # full webcam demo
scripts/export_face_liveness_onnx.py     # PyTorch -> ONNX export
scripts/test_face_liveness_predictor.py  # model load and dummy inference check
scripts/test_face_feature_extractor.py   # extractor shape check without webcam
scripts/test_full_captcha_dummy.py       # final decision logic check without webcam

runs/gru_h32_lr0005_v1/best_gru.pt
runs/gru_h32_lr0005_v1/best_gru.onnx
runs/gru_h32_lr0005_v1/best_gru_onnx_meta.json
runs/gru_h32_lr0005_v1/run_config.json
runs/gru_h32_lr0005_v1/seq_scaler.joblib
runs/gru_h32_lr0005_v1/model_results.csv
runs/gru_h32_lr0005_v1/group_results.csv
```

## Checks

Detailed final model metrics are documented in `MODEL_PERFORMANCE.md`. Visualization files are stored in `docs/performance/`, including `docs/performance/overall_test_metrics.png` and `docs/performance/confusion_matrix_test.png`.

```bash
cd /workspace/code/model/agami_liveness_modeling
PYTHONPATH=. python scripts/test_face_liveness_predictor.py
PYTHONPATH=. python scripts/test_face_feature_extractor.py
PYTHONPATH=. python scripts/test_full_captcha_dummy.py
```

Expected checks:

- face predictor loads `best_gru.pt` and returns `spoof_score`
- feature extractor returns `x_seq.shape == (16, 20)`
- dummy 3-round decision returns expected `PASS`, `RETRY`, `FAIL`

## Webcam Demo

```bash
PYTHONPATH=. python scripts/demo_webcam_full_captcha.py --camera 0
PYTHONPATH=. python scripts/demo_webcam_full_captcha.py --camera 0 --device cpu
```

Remote VS Code server pods often do not expose a physical webcam or OpenCV GUI. In that case the demo prints:

```text
Cannot open webcam. If running on remote server, run this demo locally or connect webcam frames from frontend.
```

For website integration, capture webcam frames in the frontend and either run ONNX Runtime there or forward extracted feature frames to a backend.

## ONNX Website Integration

Use:

```text
runs/gru_h32_lr0005_v1/best_gru.onnx
```

Interface:

```text
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

The ONNX graph embeds the training scaler. Pass raw 20 face features in the order above as `float32`. A higher `spoof_score` means higher spoof risk.

Regenerate ONNX:

```bash
PYTHONPATH=. python scripts/export_face_liveness_onnx.py
```

The exported ONNX was verified against PyTorch with a zero input; the observed absolute difference was about `5.96e-08`.

## Final Decision Policy

`decide_three_round_captcha(rounds)` requires exactly 3 rounds.

Policy summary:

```text
face and hand mission must both appear at least once
mission failures >= 2 -> FAIL
face missing >= 2 -> FAIL
timeout > 1 -> FAIL
total_risk < 1.2 -> PASS
1.2 <= total_risk < 2.0 -> RETRY
total_risk >= 2.0 -> FAIL
```

## Final Package

Create or refresh the downloadable package after changing code/docs:

```bash
python -m zipfile -c deliverables/agami_liveness_final_package.zip agami_liveness_final
```

The current prepared package is:

```text
deliverables/agami_liveness_final_package.zip
```

`deliverables/` is ignored by git and should be treated as a handoff artifact, not source control content.

## Artifact Policy

Model, dataset, and handoff zip files are not committed to git. They are shared as external artifacts.

Ignored artifact examples:

```text
face_clip_data.npz
runs/gru_h32_lr0005_v1/best_gru.pt
runs/gru_h32_lr0005_v1/best_gru.onnx
runs/gru_h32_lr0005_v1/seq_scaler.joblib
deliverables/agami_liveness_final_package.zip
```

After cloning the repository, place the shared artifact files back at the paths documented in this README before running the predictor or webcam demo.
