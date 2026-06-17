# 인수인계 — 안면인식 CAPTCHA: R_live_clip 오탐 원인 분석 및 수정

**작성일**: 2026-06-17
**관련 회고록**: `RETROSPECTIVE_facial_recognition_pipeline.md` (1~7장: 파이프라인 전체 구조, 8~9장: 이번 세션의 버그 분석·실험 — 이 문서는 그 전체를 압축 + "지금 당장 이어서 할 일")

---

## 0. 한 줄 요약

R_live_clip(실기기 촬영 데이터)의 FRR이 95%로 비정상적으로 높았던 원인을 "domain shift"가 아니라 **feature extraction 코드의 종횡비(aspect ratio) 버그**로 재규명하고 수정했다. 수정 후 재학습으로 FRR을 95%→81%까지 낮췄고, 추가로 source-weighted 샘플링 실험(weight×3/5/8/10)을 했지만 **R_live FRR<15%와 공격 차단율>70~80%를 동시에 만족하는 조합을 찾지 못했다**. 그래서 **아직 ONNX export도, production 교체도 하지 않은 상태**다. 코드 수정은 모두 끝났지만 **git commit은 하지 않았다** — 워킹 트리에 변경사항이 그대로 남아 있다.

---

## 1. 파이프라인 전체 구조 (회고록 1~7장 압축)

> 이번 세션 이전에 이미 완성되어 있던 부분. 자세한 내용은 `RETROSPECTIVE_facial_recognition_pipeline.md` 1~7장 참고.

### 1-1. 목표

얼굴 위변조 탐지(Face Liveness Detection) GRU 모델을 손 제스처 미션과 결합한 **3라운드 CAPTCHA 시스템**으로 만들어 실서비스 API로 배포하는 것.

### 1-2. 전체 흐름

```
raw dataset (face_images + face_videos)
  → 1) manifest 생성 (samples_manifest.jsonl, subject-aware split)
  → 2) 피처 추출 v1 (MediaPipe FaceMesh → EAR/MAR/headpose/이동량, 절대좌표)
  → 3) 피처 추출 v2 (상대좌표 개선) / v3 (시간 정규화 추가, 이번 세션에서 다룸)
  → 4) GRU 모델 학습 (gru_h32_lr0005_v1, hidden=32, lr=5e-4, epochs=80)
  → 5) PyTorch → ONNX 추출 + scaler 번들
  → 6) 3라운드 CAPTCHA 결정 로직 (face risk + hand mission + risk 누적)
  → 7) FastAPI + ONNX Runtime 추론 API (face-liveness-api)
  → 8) K8s 배포 (PVC 마운트, ArgoCD)
```

### 1-3. 데이터 소스 (source_group)

| source_group | 정체 | 특징 |
|---|---|---|
| `S_dataset_sequence` | CASIA-style 공개 데이터셋 (`live_001` 등) | 정사각형 크롭, 파일명에 subject/session/type 인코딩 (`_60.jpg`처럼 프레임 번호가 60단위로 듬성듬성) |
| `R_live_clip` | 자체 디바이스 촬영 (`real_NN_phone`) | 16:9 풀프레임, 연속 프레임(`001.jpg, 002.jpg...`) |
| `ATK_external_clip` | 외부 공격 영상(print/replay) | 16:9 풀프레임, 연속 프레임 — R_live와 같은 촬영 포맷 |
| `VL_face_video` / `VS_face_video` | `face_videos/`에서 균등 샘플링한 영상 | 세로 영상(휴대폰 세로 촬영) |

**subject-aware split**이 핵심: `live_001`과 `spoof_001`은 동일 피험자라서 같은 split에 묶어야 leakage가 안 생긴다 (train 614 / valid 169 / test 183 clips, 원본 기준).

### 1-4. 모델

배포된 모델 `gru_h32_lr0005_v1`: GRU hidden_size=32, lr=0.0005, epochs=80, patience=15, dropout=0.3, weight_decay=1e-4, 파라미터 5,217개. 입력 `(batch, 16, 20)` — 16프레임 × 20피처. 학습 데이터는 **v1 전처리(`face_clip_data.npz`, 절대좌표, 시간 정규화 없음)**.

원본(버그 수정 전) 성능(test): Accuracy 81.97%, F1 83.42%, ROC-AUC 84.53%, Attack Block 92.22%, FRR(전체) 27.96%. 그런데 R_live_clip만 따로 보면 FRR **95.24%**로 매우 높았다 — 이게 이번 세션에서 판 문제.

### 1-5. CAPTCHA 결정 로직 (`captcha_decision.py`)

3라운드(face/hand 미션 혼합) 동안의 risk를 누적해 PASS/RETRY/FAIL을 정한다. face 단독 점수만으로 FAIL 처리하지 않도록(R_live FRR이 높은 모델 특성 보완), `risk_band`(real_safe/suspicious/spoof_detected)로 점수를 양자화하고, spoof_detected가 2/3 라운드 이상 반복돼야 face 근거만으로 FAIL이 가능하게 되어 있다. 이 로직 자체는 이번 세션 초반(8장 작업 전)에 별도로 손봤고 이미 완료된 상태다.

### 1-6. 알아둘 운영/설계 포인트

- 모델 파일은 PVC 마운트로 서빙 — API 이미지 재빌드 없이 모델 교체 가능
- R_live_clip(실환경, 93개 전부 real)은 따로 "stress test"로 분리 보고하는 게 정직하다고 판단함 (메인 벤치마크에 합치면 수치가 왜곡됨)
- `ml_pipeline_team_handoff/`, `captcha_engine_team_handoff/`, `agami_liveness_final_package/` 세 디렉토리는 서로 다른 팀에게 넘긴 인계 패키지인데 **내부 파일(특히 `face_feature_extractor.py`, `runs/gru_h32_lr0005_v1/`)이 거의 동일하게 중복 보관**되어 있다 — 코드를 고칠 때 3곳 다 고쳐야 한다는 뜻 (이번 세션에서도 그렇게 했다, 섹션 3 참고).
- 트러블슈팅 이력(`.gitignore` 과도 적용, libGL.so.1 누락, session/illumination float 캐스팅 등)은 회고록 5장에 정리되어 있다.

---

## 2. 이번 세션이 시작된 이유

기존 회고록(`RETROSPECTIVE_facial_recognition_pipeline.md` 1~7장)에 정리된 얼굴 liveness GRU + 3라운드 CAPTCHA 파이프라인 작업을 이어가는 중, 사용자가 다음 의문을 제기했다:

> R_live와 다른 live 데이터(S_dataset)는 원본 영상의 촬영 방식/형식이 동일한데, 왜 모델 성능이 그렇게 다르게 나오는가? "도메인이 달라서"라고 단정하지 말고 전처리/라벨링/샘플링/feature extraction 버그를 먼저 의심해서 재분석해달라.

→ 이 의심이 맞았다. 아래가 그 결과다.

---

## 3. 발견한 버그

### 3-1. 증거

매니페스트의 실제 이미지 종횡비(가로/세로)를 확인한 결과:

| source_group | 종횡비 | 표본 수 |
|---|---|---|
| S_dataset_sequence | **1.0** (정사각형 크롭) | 685 |
| R_live_clip | **1.778** (16:9 풀프레임) | 93 |
| ATK_external_clip | **1.779** (16:9 풀프레임) | 188 |
| VL/VS (영상 추출) | 0.562 / 0.75 (세로 영상) | 56 |

R_live와 ATK_external이 거의 동일한 16:9 비율을 쓰고 S_dataset만 정사각형이라는 것이 핵심 단서였다.

### 3-2. 원인

MediaPipe FaceMesh는 landmark의 `x`를 이미지 **너비**로, `y`를 이미지 **높이**로 각각 독립 정규화한다. 그런데 모든 feature extraction 스크립트의 `_dist()` (`math.hypot(dx, dy)`)가 이 둘을 같은 단위처럼 섞어서 거리를 계산했다. 이미지가 정사각형이 아니면(R/ATK의 16:9) y축 기반 거리가 체계적으로 왜곡된다.

실제 이미지로 검증:

```
R_live (1920x1080, 16:9):
  버그 EAR(기존 코드)     ≈ 0.40 ~ 0.43
  픽셀보정 EAR(올바른 값)  ≈ 0.23 ~ 0.25   ← S_dataset과 거의 같은 분포!

S_dataset (거의 정사각형):
  버그 EAR ≈ 픽셀보정 EAR ≈ 0.21 ~ 0.26   (정사각형이라 버그 영향 없음)
```

회고록 4-2의 "EAR Cohen's d=4.29"는 실제 얼굴 차이가 아니라 이 버그 때문이었다. 자세한 검증 과정은 **RETROSPECTIVE 8장** 참고.

---

## 4. 적용한 코드 수정 (완료, 커밋 안 됨)

MediaPipe landmark를 받는 즉시 `y_corrected = y / aspect_ratio` (`aspect_ratio = width/height`)로 보정하고, 이후 거리 계산 함수(`_dist`, `_ear` 등)는 건드리지 않는 방식으로 최소 변경했다.

수정된 파일 (모두 `git status`에 `M`으로 표시됨, 아직 add/commit 안 함):

| 파일 | 역할 |
|---|---|
| `ml-pipeline/facial_recognition/preprocessing/extract_features_rel.py` | v2 전처리 (학습에 실사용) |
| `ml-pipeline/facial_recognition/preprocessing/extract_features_time_norm.py` | v3 전처리 (학습에 실사용) |
| `ml-pipeline/facial_recognition/preprocessing/extract_image_features.py` | v1 전처리 (실제 배포 모델이 쓰던 데이터) |
| `ml-pipeline/facial_recognition/preprocessing/extract_video_features.py` | 레거시, 일관성 유지 |
| `ml-pipeline/facial_recognition/captcha_engine_team_handoff/captcha_engine_team/src/face_feature_extractor.py` | **운영 서빙 코드** |
| `ml-pipeline/facial_recognition/ml_pipeline_team_handoff/ml_pipeline_team/src/face_feature_extractor.py` | 위와 동일 파일(3개 인계 패키지 중복) |
| `ml-pipeline/facial_recognition/agami_liveness_final_package/agami_liveness_final/src/face_feature_extractor.py` | 위와 동일 |

별개로 같은 세션 초반에 (이 버그 발견 전) captcha 판정 로직/API 쪽도 수정했었다 — 이건 다른 이슈(risk_band corroboration)였고 이미 완료된 상태:
- `ml-pipeline/facial_recognition/api/main.py`, `api/schemas.py`
- `ml-pipeline/facial_recognition/captcha_decision.py`
- `ml-pipeline/facial_recognition/export/export_face_liveness_onnx.py`
- `ml-pipeline/facial_recognition/inference/onnx_face_liveness_detector.py`

`extract_face_features.py`(최상위, 어디서도 import 안 됨)는 사용되지 않는 죽은 코드라 수정 대상에서 제외했다.

---

## 5. 재추출 + 재학습 + 실험 결과

모든 실험 산출물은 `ml-pipeline/facial_recognition/model/retrain_aspectfix/`에 있다 (git에 untracked 상태, `?? ` 표시).

```
model/retrain_aspectfix/
├── face_clip_data.npz                 # 종횡비만 수정한 v1 재추출 (1022 clips)
├── face_clip_data_time_norm.npz       # 종횡비+시간정규화 v3 재추출 (1013 clips)
├── train_source_weighted.py           # source-weighted 실험 스크립트
└── runs/
    ├── gru_h32_lr0005_aspectfix/       # v1 재추출 재학습 (weight 없음)
    ├── gru_h32_lr0005_v3_aspectfix/    # v3 재추출 재학습 (weight 없음)
    ├── gru_v3_aspectfix_w3/            # v3 + R_live weight×3
    ├── gru_v3_aspectfix_w5/            # v3 + R_live weight×5
    ├── gru_v3_aspectfix_w8/            # v3 + R_live weight×8  ← 가장 균형 잡힌 후보
    └── gru_v3_aspectfix_w10/           # v3 + R_live weight×10
```

각 run 디렉토리에는 `best_gru.pt`, `seq_scaler.joblib`, `run_config.json`, `model_results.csv`, `group_results.csv`, `train_history.csv`가 있고, weighted 실험들은 추가로 `threshold_sweep.csv`(threshold 0.20~0.95, 0.05 간격)가 있다.

### 5-1. 비교표 (test split, 각 모델의 best_f1 threshold 기준)

| 모델 | threshold | R_live FRR | print 차단율 | replay 차단율 | 전체 attack block | ROC-AUC |
|---|---|---|---|---|---|---|
| v1 (원본, 버그 있음, 기존 배포 모델) | 0.21 | **95.24%** | 92.86% | 91.94% | 92.22% | 0.845 |
| v1 재추출(종횡비만 수정) | 0.22 | **95.24%** (변화 없음) | 92.86% | 90.91% | 91.49% | 0.850 |
| v3 (종횡비+시간정규화, weight 없음) | 0.26 | **80.95%** | 92.86% | 67.19% | 75.00% | 0.850 |
| v3 + weight×3 | 0.50 | 42.86% | 92.86% | 56.25% | 67.39% | 0.857 |
| v3 + weight×5 | 0.21 | 66.67% | 92.86% | 71.88% | 78.26% | 0.857 |
| v3 + weight×8 | 0.18 | 71.43% | 92.86% | 76.56% | 81.52% | 0.870 |
| v3 + weight×10 | 0.18 | 71.43% | 92.86% | 75.00% | 80.43% | 0.870 |

**중요한 함정**: weight를 올릴수록 모델 자체의 `best_f1` 자동 threshold가 낮아져서(0.26→0.18) 위 표만 보면 R_live FRR이 오히려 나빠진 것처럼 보인다. `best_f1`은 전체 F1만 최적화하고 R_live FRR을 직접 보지 않기 때문이다. **threshold를 수동으로 조정해야** weight의 효과가 드러난다 (아래 5-2).

### 5-2. Threshold를 수동으로 올렸을 때 (test split)

| 모델 | threshold | R_live FRR | print 차단율 | replay 차단율 | 전체 attack block |
|---|---|---|---|---|---|
| v3 + weight×8 | 0.65 | **14.29%** | 89.29% | 45.31% | 58.70% |
| v3 + weight×8 | 0.70 | **9.52%** | 82.14% | 39.06% | 52.17% |
| v3 + weight×10 | 0.60 | **14.29%** | 82.14% | 48.44% | 58.70% |
| v3 + weight×10 | 0.65 | **9.52%** | 82.14% | 42.19% | 54.35% |

valid split에서도 동일 패턴 재현됨(우연 아님).

### 5-3. 결론 — 목표 미달, 둘 다 만족하는 조합 없음

목표였던 "R_live FRR<10~15% **그리고** 공격 차단율>70~80%"를 동시에 만족하는 weight×threshold 조합은 **찾지 못했다**. R_live FRR을 낮추려고 threshold를 올리면 replay 차단율이 39~48%까지 무너진다(print는 82~89%로 비교적 안정). 병목은 **replay 공격의 spoof_score 분포가 R_live(진짜 사람)의 분포와 너무 가깝다는 것**이고, 이건 sampler 가중치로 해결되는 문제가 아니다.

**가장 균형 잡힌 후보**: `v3 + weight×8`, threshold≈0.65 (R_live FRR 14.29%/test, 8.33%/valid — 목표 충족. 단 전체 attack block 58.7%로 목표 미달).

→ 그래서 **ONNX export와 production promote는 보류 중**이다 (사용자가 명시적으로 보류 지시함).

---

## 6. 지금 git 상태

세션 내내 **commit을 한 번도 하지 않았다.** `git status --short` 결과:

```
 M ml-pipeline/facial_recognition/agami_liveness_final_package/.../face_feature_extractor.py
 M ml-pipeline/facial_recognition/api/main.py
 M ml-pipeline/facial_recognition/api/schemas.py
 M ml-pipeline/facial_recognition/captcha_decision.py
 M ml-pipeline/facial_recognition/captcha_engine_team_handoff/.../face_feature_extractor.py
 M ml-pipeline/facial_recognition/export/export_face_liveness_onnx.py
 M ml-pipeline/facial_recognition/inference/onnx_face_liveness_detector.py
 M ml-pipeline/facial_recognition/ml_pipeline_team_handoff/.../face_feature_extractor.py
 M ml-pipeline/facial_recognition/preprocessing/extract_features_rel.py
 M ml-pipeline/facial_recognition/preprocessing/extract_features_time_norm.py
 M ml-pipeline/facial_recognition/preprocessing/extract_image_features.py
 M ml-pipeline/facial_recognition/preprocessing/extract_video_features.py
?? RETROSPECTIVE_20260616_context_emotion.md       (이 작업과 무관, 다른 작업의 산물)
?? RETROSPECTIVE_facial_recognition_pipeline.md     (이 작업의 전체 회고록, 8/9장 참고)
?? ml-pipeline/context_emotion/                     (이 작업과 무관)
?? ml-pipeline/facial_recognition/model/retrain_aspectfix/   (이번 실험 산출물 전체)
?? ml-pipeline/facial_recognition/model/retrain_v4/  (이전 세션의 다른 실험, 이 작업과 무관)
```

새 계정/새 세션에서 이어가려면 **이 워킹 트리 그대로**(또는 같은 변경사항이 적용된 브랜치)에서 시작해야 한다. 커밋을 만들지 여부는 아직 사용자가 정하지 않았다.

---

## 7. 다음에 이어서 할 일 (우선순위 순)

1. **(가장 먼저 의논할 것)** 5-3 결론을 사용자에게 다시 확인 — 58.7% attack block을 받아들이고 v3+w8을 임시 운영 후보로 promote할지, 아니면 근본 해결(아래 2~4)을 먼저 할지 결정 필요.
2. **R_live_clip 추가 데이터 수집** — 현재 93개 → 200개 이상. sampler weight로는 분포 자체를 못 늘리므로 가장 근본적인 해법.
3. **replay 공격 특화 피처 추가 검토** — 현재 피처셋(EAR/MAR/움직임 기반)은 "느리게 움직이는 진짜 사람"과 "느리게 움직이는 재생 영상(replay)"을 구분하기 어렵다. moiré, flicker, 프레임레이트 아티팩트 등 화면-재생 특유의 신호 추가 검토.
4. **augmentation 시도** — `model/agami_liveness_modeling/src/train_gru_aug.py`(기존에 준비되어 있던 스크립트, 이번 세션에서는 안 씀)로 time-warp 등 변형 기반 증강을 v3+종횡비fix 데이터에 적용해보는 것도 옵션.
5. 운영 후보가 확정되면:
   - ONNX export (`export/export_face_liveness_onnx.py`, 이미 `--high-threshold` 옵션 지원하도록 수정됨)
   - `captcha_engine_team_handoff` 등 서빙 코드의 `selected_features`를 v1 네이밍(`nose_x` 등) → v3 네이밍(`nose_x_rel`, `*_tn`)으로 맞추는 작업 필요 (현재 서빙 extractor는 v1 feature set 기준으로 짜여 있음 — v3 모델로 교체하려면 feature 계산 방식 자체를 다시 맞춰야 함)
   - 그 다음에야 production promote
6. 코드 수정사항(섹션 4)을 git commit할지 사용자와 확인.

---

## 8. 참고 — 새 세션에서 컨텍스트 빠르게 잡는 법

1. 이 파일(`HANDOFF_facial_recognition_aspectfix.md`)을 먼저 읽는다 — 1장이 파이프라인 전체 구조, 2장부터가 이번 세션 작업이다.
2. 자세한 근거/검증 과정이 필요하면 `RETROSPECTIVE_facial_recognition_pipeline.md`의 1~7장(파이프라인 전체), 8장(버그 발견·재추출·재학습), 9장(source-weighted 실험)을 읽는다.
3. `git diff`로 실제 코드 변경 내용을 확인한다 (아직 커밋 안 됐으므로 워킹 트리에 그대로 있음).
4. `ml-pipeline/facial_recognition/model/retrain_aspectfix/runs/*/threshold_sweep.csv`를 보면 위 비교표의 원본 데이터를 그대로 확인할 수 있다.
