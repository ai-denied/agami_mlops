# 안면인식 ML 파이프라인 정리 회고록

**기간**: 2026-06-08 ~ 2026-06-16  
**담당**: agami MLOps

---

## 1. 무엇을 했나

얼굴 위변조 탐지(Face Liveness Detection) 전체 파이프라인을 처음부터 구축하고 정리했다.  
단순 모델 학습에 그치지 않고, 전처리 → 학습 → 평가 → ONNX 추출 → 추론 API → K8s 배포까지 엔드-투-엔드를 완성했다.

최종 목표는 **얼굴 liveness GRU 모델을 손 제스처 미션과 결합한 3라운드 CAPTCHA 시스템**이었고, 이를 실서비스 API로 배포하는 것까지 완료했다.

---

## 2. 파이프라인 전체 구조

```
raw dataset (face_images + face_videos)
        │
        ├─ 1단계: 데이터 구조 파악 + manifest 생성
        │          samples_manifest.jsonl (966 clips, subject-aware split)
        │
        ├─ 2단계: 피처 추출 (전처리 v1)
        │          MediaPipe FaceMesh → EAR, MAR, head pose, 이동량
        │          face_clip_features.jsonl (966 clips × 27 피처)
        │
        ├─ 3단계: 전처리 개선 (v2, 상대좌표)
        │          절대좌표 → bbox 정규화 + 클립 평균 기준 상대좌표
        │          face_clip_data_rel.npz (957 clips × seq(16,20) + static(16))
        │
        ├─ 4단계: GRU 모델 학습
        │          runs/gru_h32_lr0005_v1 (최종 선정)
        │          test AUC 84.53%, Attack Block Rate 92.22%
        │
        ├─ 5단계: 모델 패키징
        │          PyTorch → ONNX 추출, scaler 번들
        │
        ├─ 6단계: 3라운드 CAPTCHA 결정 로직
        │          face liveness + hand mission + risk 누적
        │
        ├─ 7단계: 추론 API 구축
        │          FastAPI + ONNX Runtime → face-liveness-api
        │
        └─ 8단계: K8s 배포
                   PVC 마운트 방식, ArgoCD 관리
```

---

## 3. 단계별 주요 작업

### 3-1. 데이터셋 구조 파악 및 manifest 생성

데이터셋은 두 가지 소스가 혼재했다.
- **CASIA-style**: `live_001/live_001-1-1-1-1_60.jpg` 형식 — 파일명에 subject/session/type/illumination/device 인코딩
- **디바이스 캡처**: `real_01_phone/` — 순번 파일명, 30프레임씩 청크

두 소스를 통합해 **clip 단위 manifest** (`samples_manifest.jsonl`)를 생성했다.  
핵심은 **subject-aware split**: `live_001`과 `spoof_001`은 동일 피험자이므로 반드시 같은 split에 배치해 leakage를 방지했다.

| Split | clip 수 | real | spoof |
|-------|---------|------|-------|
| train | 614 | 371 | 243 |
| valid | 169 | 84 | 85 |
| test  | 183 | 93 | 90 |

Leakage 검증 결과: train ∩ valid = ∅, train ∩ test = ∅ ✅

---

### 3-2. 전처리 v1 — 피처 추출

**MediaPipe FaceMesh** (468 랜드마크)로 clip당 프레임을 처리해 아래 피처를 추출했다.

| 카테고리 | 핵심 피처 |
|---------|---------|
| 눈 | EAR (Eye Aspect Ratio), blink_count, ear_std |
| 입 | MAR (Mouth Aspect Ratio), smile_ratio |
| 머리 자세 | yaw, pitch, roll (mean + std) |
| 이동량 | nose_movement, face_movement, face_stability |

**실행 결과**: 9,627장 중 9,591장 성공(99.6%), 영상 56개 전수 성공(100%). 스킵은 모두 `no_face_detected` — 측면·가림·저해상도.

**핵심 발견 (Cohen's d)**:  
`nose_movement` (d=1.32), `face_stability` (d=1.04), `ear_std` (d=0.89) 순으로 real/spoof 판별력이 강했다. **real은 자연스럽게 움직이고 깜빡이는 반면, spoof(사진·영상 재생)는 움직임이 고정적**이라는 직관과 일치.

---

### 3-3. 전처리 v2 — 상대좌표 개선

v1의 절대좌표(`nose_x`, `cx`)는 카메라 위치·거리에 민감한 문제가 있었다.

**개선 방향**: 각 클립의 시계열 평균을 빼서 상대좌표로 변환.
```
nose_x_rel = nose_x - mean(nose_x)  // 클립 내 얼굴 이동량만 남음
```

추가 변경:
- bbox(얼굴 폭) 기준 정규화로 카메라 거리 편차 제거
- velocity 피처 10개 추가 (코/얼굴 중심 dx, dy, EAR/MAR/head pose 변화량)
- train set 1~99% percentile clipping으로 이상치 억제
- 유효 프레임 < 3개 클립 9개 제거

**결과**: 957 clips × (seq: 16프레임 × 20 피처) + (static: 16 피처)

---

### 3-4. 모델 학습 및 선정

GRU 기반 시계열 분류기를 학습했다. 최종 선정 모델: `gru_h32_lr0005_v1`

```
hidden_size=32, lr=0.0005, epochs=80, patience=15
dropout=0.3, weight_decay=0.0001, pos_weight 보정
parameter count: 5,217개 (경량 모델)
input: (batch, 16, 20)
```

| Split | Accuracy | F1 | ROC-AUC | Attack Block | FRR |
|-------|---------|-----|---------|-------------|-----|
| valid | 81.07% | 83.33% | 88.43% | 94.12% | 32.14% |
| test  | 81.97% | 83.42% | 84.53% | 92.22% | 27.96% |

**공격 유형별 test 성능**:
- print 공격 차단율: **92.86%**
- replay 공격 차단율: **91.94%**

---

### 3-5. 모델 패키징 및 CAPTCHA 결정 로직

PyTorch 모델을 ONNX로 추출하고, 학습 scaler를 ONNX 그래프에 번들했다.

```
input: x_seq [batch, 16, 20] (float32)
output: spoof_score [batch]    (0~1, 높을수록 위험)
threshold: 0.21
```

얼굴 liveness 단독 판정이 아닌 **3라운드 CAPTCHA 결정 로직**을 설계했다:

```
3라운드 누적 risk score 계산
    face + hand mission 모두 1회 이상 등장 필수
    mission 실패 ≥ 2회 → FAIL
    얼굴 미검출 ≥ 2회 → FAIL
    total_risk < 1.2 → PASS
    1.2 ≤ total_risk < 2.0 → RETRY
    total_risk ≥ 2.0 → FAIL
```

---

### 3-6. 추론 API 및 K8s 배포

FastAPI 기반 추론 API를 구축하고 Docker 이미지를 빌드, K8s에 배포했다.

```
feat(face-liveness): 얼굴 활성도 추론 API 추가          (d804537)
feat(face-liveness): PVC + K8s 매니페스트 추가           (96e1dfd)
feat(facial_recognition): ML 파이프라인 패키지 구조 완성  (69bfbe1)
```

모델 파일은 PVC 마운트 방식으로 서빙 — API 이미지에 모델을 포함하지 않아 모델 교체가 이미지 재빌드 없이 가능하다.

---

## 4. 핵심 인사이트

### 4-1. subject leakage는 반드시 설계 단계에서 잡아야 한다

CASIA-style 데이터는 `live_001`과 `spoof_001`이 같은 피험자다. random split을 쓰면 동일인의 real과 spoof가 train/test에 분리되어 **모델이 사람 얼굴을 암기하는** 형태로 학습된다. subject-aware split을 구현하지 않으면 test AUC가 실제보다 훨씬 좋게 나온다.

### 4-2. 절대좌표는 과대평가를 만든다

v1의 `nose_x` 같은 절대좌표는 "얼굴이 화면 중앙에 있냐"를 반영한다. 통제 환경 데이터는 카메라 세팅이 일정해서 좌표만으로도 real/spoof가 갈리는 경우가 생겼다. 상대좌표 v2로 바꾸니 S_dataset_sequence AUC는 소폭 상승했지만, R_live_clip FRR이 오히려 악화됐다 — 실환경 일반화 문제는 좌표 표현만으로는 해결되지 않는다.

### 4-3. R_live_clip은 별도 stress test로 분리하는 게 정직하다

R_live_clip(실환경 촬영, 93개 전부 real)의 FRR이 95%로 매우 높다.  
이를 메인 benchmark에 포함하면 전체 지표가 낮아져 발표용 수치가 흐릿해진다.  
반대로 제거하면 과대 포장이 된다.  
**"Main Benchmark + R_live Stress Test" 분리 보고**가 가장 솔직한 방식이었다.

| Benchmark | n | Accuracy | Attack Block | FRR |
|-----------|---|---------|-------------|-----|
| Main (S_dataset + ATK_external) | 162 | 91.98% | 92.22% | 8.33% |
| R_live Stress Test | 21 | 4.76% | — | **95.24%** |

> **[2026-06-17 정정]** 이 절의 "R_live는 실환경이라 다르다"는 전제 자체가 틀렸다.
> 실제 원인은 feature extraction 코드의 종횡비(aspect ratio) 버그였다. **8장** 참고.

### 4-4. 얼굴 liveness는 단독 판정기가 아니다

R_live_clip FRR 95%는 서비스에서 쓸 수 없는 수치다.  
이를 극복하는 방법으로 **3라운드 CAPTCHA 결합**을 선택했다 — 얼굴 liveness는 risk score를 제공하고, 손 미션 성공 여부와 함께 최종 판정을 내린다. 하나의 지표가 불안정할 때 다중 신호를 결합하는 설계가 현실적이었다.

---

## 5. 트러블슈팅 모음

| # | 문제 | 원인 | 해결 |
|---|------|------|------|
| 1 | `.gitignore data/` 패턴이 `flashlight/data/` Python 모듈까지 제외 | `.gitignore` 글로브 과도 적용 | `!ml-pipeline/flashlight/data/**` 예외 추가 + `git add -f` |
| 2 | `face_image_features.csv` 구버전 → 현재 폴더 구조와 불일치 | flat 경로로 추출한 구버전 스크립트 사용 | `face_clip_features.jsonl`로 완전 대체 |
| 3 | `session` / `illumination` 컬럼이 float으로 저장 | CSV 저장 시 null 혼재 → int → float 변환 | 사용 시 `int(session)` 캐스팅 명시 |
| 4 | 서버에서 `libGL.so.1` 오류 | OpenCV headless 환경에서 cv2 import 시 | `apt-get install -y libgl1 libglib2.0-0` |
| 5 | `ATK08_replay_clip003` face_detect_rate 0.47 | gopro 광각 촬영으로 얼굴이 작아 검출 실패 | `face_detect_rate < 0.5` 필터링 권고로 문서화 |
| 6 | R_live_clip EAR 등 거리 피처가 ATK_spoof와 비슷하고 S_dataset과 다르게 나옴 (d=4.29) | feature extraction의 종횡비(aspect ratio) 미보정 버그. 자세한 내용은 **8장** | landmark y좌표를 `aspect_ratio = width/height`로 나눠 가로폭 기준 단위로 통일 |

---

## 6. 미해결 / 다음 단계

| 항목 | 상세 |
|------|------|
| **R_live_clip FRR 개선** | ~~source-weighted training / augmentation 적용~~ → **8장**에서 근본 원인(종횡비 버그)을 고쳤으므로, 먼저 v2/v3 npz를 재추출하고 재학습한 뒤 FRR이 실제로 개선되는지 확인하는 것이 우선. 그래도 잔차가 남으면 source-weighted/augmentation을 적용 (`train_gru_source_weighted.py`, `train_gru_aug.py` 준비됨) |
| **R_live_clip 추가 수집** | 현재 93개 → 최소 200개 이상 권장. 야외·실환경 다양한 기기 조건 |
| **피처 분포 시각화** | PCA/t-SNE로 R_live vs S_dataset 분리 정도 확인 — 어떤 피처에서 도메인 갭이 벌어지는지 특정 |
| **print 클립 보강** | 현재 print(141) < replay(277) < live(548) — print 클립 부족 |
| **ONNX 프론트엔드 통합** | 웹 프론트에서 MediaPipe로 피처 추출 후 ONNX Runtime 직접 실행 또는 백엔드 전송 |

---

## 7. 최종 산출물

| 산출물 | 경로 |
|--------|------|
| clip manifest | `dataset/samples_manifest.jsonl` |
| 전처리 v1 | `features/face_clip_features.jsonl` (966 clips) |
| 전처리 v2 | `features/face_clip_data_rel.npz` (957 clips) |
| 최종 데이터셋 | `face_clip_data.npz` (966 clips, v1 기반) |
| PyTorch 모델 | `runs/gru_h32_lr0005_v1/best_gru.pt` |
| ONNX 모델 | `runs/gru_h32_lr0005_v1/best_gru.onnx` |
| 성능 보고서 | `ml_pipeline_team_handoff/MODEL_PERFORMANCE.md` |
| 추론 API | `facial_recognition/api/main.py` |
| K8s 매니페스트 | `manifests/face-liveness-api.yaml` |
| ML팀 인계 패키지 | `ml_pipeline_team_handoff/` |
| CAPTCHA팀 인계 패키지 | `captcha_engine_team_handoff/` |

---

## 8. 후속 조치 (2026-06-17) — R_live_clip 오탐의 진짜 원인은 domain shift가 아니라 종횡비 버그였다

### 8-1. 문제 제기

4-3에서 "R_live_clip은 실환경 도메인이라 FRR이 높다"고 결론 냈었다. 하지만 원본 영상을 직접 확인한 결과 **R_live와 다른 live 데이터의 촬영 방식·형식은 동일**했다 — "실환경이라 다르다"는 전제 자체가 사실과 달랐다. 그래서 domain shift로 단정하지 않고 전처리/라벨링/샘플링/feature extraction 버그 가능성을 처음부터 다시 따라가봤다.

### 8-2. 원인 — MediaPipe 정규화 좌표를 종횡비 보정 없이 섞어 쓴 버그

MediaPipe FaceMesh는 landmark의 `x`를 이미지 **너비**로, `y`를 이미지 **높이**로 각각 독립 정규화해서 돌려준다. 그런데 `_dist()`(`math.hypot(dx, dy)`)는 이 둘을 그냥 같은 단위처럼 섞어서 거리를 계산했다. 이미지가 정사각형이면 문제가 없지만, **가로세로 비율이 다르면 y축 변화량이 체계적으로 부풀려지거나 줄어든다.**

실제 manifest 이미지의 종횡비를 확인해보니:

| source group | 원본 이미지 비율(가로/세로) | 표본 수 |
|---|---|---|
| S_dataset_sequence (`live_001` 등) | **1.0** (정사각형 크롭) | 685 |
| R_live_clip (`real_NN_phone`) | **1.778** (16:9 풀프레임) | 93 |
| ATK_external_clip | **1.779** (16:9 풀프레임) | 188 |
| VL/VS (영상 추출) | 0.562 / 0.75 (세로 영상) | 56 |

R_live_clip과 ATK_external_clip은 거의 동일한 16:9 비율을 쓰고, S_dataset만 정사각형이다. 이게 "R_live ≈ ATK_spoof"로 보였던 진짜 이유였다 — 같은 위협 도메인이라서가 아니라 **같은 종횡비 왜곡을 똑같이 겪기 때문**이었다.

실제 이미지로 EAR을 "버그 있는 방식"과 "픽셀 단위로 보정한 방식"으로 둘 다 계산해서 검증했다:

```
R_live (1920x1080, 16:9):
  버그 EAR(기존 코드)     ≈ 0.40 ~ 0.43
  픽셀보정 EAR(올바른 값)  ≈ 0.23 ~ 0.25

S_dataset (267~665px, 거의 정사각형):
  버그 EAR ≈ 픽셀보정 EAR ≈ 0.21 ~ 0.26   (정사각형이라 버그 영향 거의 없음)
```

보정 전에는 R_live의 EAR이 S_dataset보다 약 1.7~2배 부풀려져 있었다. 보정 후에는 R_live(0.23~0.25)와 S_dataset(0.21~0.26)이 거의 같은 분포가 된다 — 회고록 4-2에 적었던 "EAR Cohen's d=4.29"는 실제 얼굴 차이가 아니라 이 버그 때문이었다.

### 8-3. 검증한 항목과 결론

| # | 확인 항목 | 결론 |
|---|---|---|
| 1 | 원본 구조/해상도 비교 | S=정사각형 크롭(267~665px), R/ATK=16:9 풀프레임(1920x1080 등). **해상도가 아니라 종횡비**가 핵심 변수 |
| 2 | manifest 단계 label/source 오분류 | 없음. `infer_source_group()`은 sample_id 접두사로만 분기, label/attack_type 정상 |
| 3 | train/valid/test split 유지 | R 9개 subject 전부 한 split에만 존재 — leakage 없음 |
| 4 | feature extraction의 source별 분기 | 없음. 모든 버전(v1/v2/v3)이 source 조건 없이 동일 코드 경로 사용 — **분기 버그가 아니라 입력 이미지 종횡비를 보정하지 않은 게 문제** |
| 5 | face_detect_rate/seq_len 차이 | 별도 원인(조명, 카메라 각도 등) — 이번 분석 범위 밖, 버그 아님 |
| 6 | EAR d=4.29 원인 | **확인됨** — landmark 거리 계산의 종횡비 미보정 버그 (8-2) |
| 7 | cx/cy/nose_speed 원인 | 동일 버그의 연장. `face_w`(가로 거리)는 영향이 적지만 y축이 섞이는 모든 `hypot()` 계산(nose_movement, face_movement, head_roll 등)이 16:9 그룹(R, ATK)에서 동일하게 왜곡됨 |
| 8 | 제거 대신 원인 규명 | R_live를 제거하지 않고도, 추출 코드의 거리 계산을 종횡비 보정하는 것만으로 "R_live ≈ ATK_spoof로 보이던 현상"과 "R_live ≠ S_dataset으로 보이던 현상"이 동시에 설명됨 |

### 8-4. 수정한 코드

MediaPipe landmark를 받는 즉시 `y_corrected = y / aspect_ratio` (`aspect_ratio = width / height`)로 보정해, 이후의 모든 거리·각도 계산이 가로폭 기준 단일 단위를 쓰도록 했다. 거리 함수(`_dist`, `_ear`, `_mar`, `_head_roll` 등) 자체는 건드리지 않고 입력 landmark만 보정해, 변경 범위를 최소화했다.

| 파일 | 역할 |
|---|---|
| `preprocessing/extract_features_rel.py` (v2) | `face_clip_data_rel.npz` 생성 — 학습에 실제 사용 |
| `preprocessing/extract_features_time_norm.py` (v3) | `face_clip_data_time_norm.npz` 생성 — 학습에 실제 사용 |
| `preprocessing/extract_image_features.py` (v1, 레거시) | 일관성을 위해 동일 수정 |
| `preprocessing/extract_video_features.py` (레거시) | 일관성을 위해 동일 수정 |
| `captcha_engine_team_handoff/.../face_feature_extractor.py` | **운영 서빙 코드** — CAPTCHA 엔진이 실시간으로 쓰는 feature extractor |
| `ml_pipeline_team_handoff/.../face_feature_extractor.py` | 위와 동일 파일(3개 인계 패키지에 중복 보관) |
| `agami_liveness_final_package/.../face_feature_extractor.py` | 위와 동일 |

`extract_face_features.py`(최상위 레거시, 어디서도 import되지 않음)는 사용되지 않는 코드라 수정 대상에서 제외했다.

### 8-5. 운영 영향

이 버그는 학습 스크립트뿐 아니라 **실서비스 feature extractor에도 동일하게 존재**했다 — 즉 사용자가 카메라를 가로/세로로 어떻게 들고 있는지(또는 디바이스 카메라의 기본 비율)에 따라 spoof_score가 왜곡될 수 있는 상태로 배포되어 있었다. 이번 수정으로 서빙 경로도 함께 고쳤다.

### 8-6. 재추출 + 재학습 결과 (2026-06-17 실행)

종횡비 보정된 코드로 두 가지 데이터셋을 재추출하고, 원본과 동일한 하이퍼파라미터(hidden=32, lr=5e-4, epochs=80, patience=15, dropout=0.3, weight_decay=1e-4, threshold_strategy=best_f1)로 재학습했다. 결과물은 `model/retrain_aspectfix/`에 저장.

| 데이터 | R_live_clip FRR (test) | 비고 |
|---|---|---|
| 원본 v1 (버그 있음, 4-3 기준) | **95.24%** | 기존 배포 모델 |
| v1 재추출(종횡비만 수정) + 재학습 | **95.24%** (변화 없음) | EAR 등 절대 피처는 정상화됐지만 velocity 피처의 frame_interval 문제가 그대로 남아 FRR을 지배함 |
| v3 재추출(종횡비 수정 + 시간 정규화) + 재학습 | **80.95%** (▼14.3%p) | valid set 기준 75%. overall test ROC-AUC도 0.85→0.85 유사하나 valid AUC 0.85→0.91로 개선 |

**해석**: 종횡비 버그는 실재했고 EAR 등 정적 피처는 완전히 정상화됐지만(8-2 검증), 단독으로는 R_live_clip FRR을 개선하지 못했다 — v1 feature set에는 frame_interval(시간 정규화)이 전혀 없어서 velocity 피처가 여전히 R_live(frame_interval=1)와 S_dataset(frame_interval=60)을 다른 스케일로 만들기 때문이다. **종횡비 수정 + 시간 정규화(v3)를 같이 적용해야** R_live FRR이 95%→81%로 의미 있게 개선된다. 단, 81%는 여전히 실서비스 기준으로는 높아 추가 개선이 필요하다.

| 산출물 | 경로 |
|---|---|
| 재추출 v1 데이터(종횡비만 수정) | `model/retrain_aspectfix/face_clip_data.npz` |
| 재추출 v3 데이터(종횡비+시간정규화) | `model/retrain_aspectfix/face_clip_data_time_norm.npz` |
| 재학습 v1 결과 | `model/retrain_aspectfix/runs/gru_h32_lr0005_aspectfix/` |
| 재학습 v3 결과 (권장 후보) | `model/retrain_aspectfix/runs/gru_h32_lr0005_v3_aspectfix/` |

### 8-7. 다음 단계

- v3+종횡비fix 모델을 ONNX로 export하기 전에, source-weighted 추가 실험으로 R_live FRR을 운영 기준까지 더 낮출 수 있는지 확인 → **9장에서 실행**
- production 배포 전, captcha_engine_team_handoff 등 서빙 코드의 selected_features를 v1(`nose_x` 등) → v3(`nose_x_rel`, `*_tn`) 네이밍으로 맞춰야 함 — 현재 서빙 extractor는 v1 feature set과 매칭되어 있어 v3 모델로 교체 시 feature 이름/계산 방식도 함께 바꿔야 한다
- R_live_clip 추가 수집(현재 93개 → 200개 이상)은 여전히 유효한 권장사항

---

## 9. Source-weighted 추가 실험 (2026-06-17)

### 9-1. 목적

8장에서 v3(종횡비 수정 + 시간 정규화)로 R_live_clip FRR을 95.24% → 80.95%까지 낮췄지만, 운영 기준(R_live FRR 10~15% 이하)에는 한참 못 미쳤다. v3 데이터 위에 train 샘플러에서 R_live_clip을 더 자주 뽑도록 가중치를 줘서(source-weighted sampling) 추가 개선 여지가 있는지 확인했다. 목표:

- R_live_clip FRR < 10~15%
- 공격(print/replay) 차단율 > 70~80%
- 전체 성능이 과도하게 무너지지 않을 것

### 9-2. 실험 설계

`model/retrain_aspectfix/train_source_weighted.py`로 v3 npz(`face_clip_data_time_norm.npz`) 위에 `WeightedRandomSampler`를 적용해 R_live_clip 학습 샘플(650개 train 중 60개)의 추출 확률을 weight배로 올렸다. 가중치 3 / 5 / 8 / 10에 대해 동일 하이퍼파라미터(hidden=32, lr=5e-4, epochs=80, patience=15, dropout=0.3, weight_decay=1e-4)로 재학습하고, split은 npz에 고정된 기존 train/valid/test를 그대로 사용했다. 각 모델에 대해 threshold 0.20~0.95(0.05 간격)로 sweep하며 R_live FRR, live 전체 FRR, print/replay 차단율, 전체 attack block rate, ROC-AUC를 기록했다 (`runs/gru_v3_aspectfix_w{3,5,8,10}/threshold_sweep.csv`).

### 9-3. 결과 — best_f1 임계값 기준 (test)

| 모델 | threshold | R_live FRR | print 차단율 | replay 차단율 | 전체 attack block | ROC-AUC |
|---|---|---|---|---|---|---|
| v1 (원본 버그) | 0.21 | 95.24% | 92.86% | 91.94% | 92.22% | 0.845 |
| v1 재추출(종횡비만) | 0.22 | 95.24% | 92.86% | 90.91% | 91.49% | 0.850 |
| v3 (종횡비+시간정규화, weight 없음) | 0.26 | 80.95% | 92.86% | 67.19% | 75.00% | 0.850 |
| v3 + weight×3 | 0.50 | 42.86% | 92.86% | 56.25% | 67.39% | 0.857 |
| v3 + weight×5 | 0.21 | 66.67% | 92.86% | 71.88% | 78.26% | 0.857 |
| v3 + weight×8 | 0.18 | 71.43% | 92.86% | 76.56% | 81.52% | 0.870 |
| v3 + weight×10 | 0.18 | 71.43% | 92.86% | 75.00% | 80.43% | 0.870 |

**weight를 올릴수록 best_f1이 고른 기본 threshold가 낮아져서(0.26→0.18) 오히려 R_live FRR이 다시 악화된다** — best_f1은 전체 F1을 최적화할 뿐 R_live FRR을 직접 보지 않기 때문이다. 즉 "weight만 올리고 threshold는 자동 선택"으로는 목표를 달성할 수 없고, **weight와 threshold를 함께 수동으로 조정**해야 한다.

### 9-4. Threshold sweep — R_live FRR을 직접 낮추는 지점 탐색 (test)

| 모델 | threshold | R_live FRR | live 전체 FRR | print 차단율 | replay 차단율 | 전체 attack block |
|---|---|---|---|---|---|---|
| v3 + weight×8 | 0.65 | **14.29%** | 6.19% | 89.29% | 45.31% | 58.70% |
| v3 + weight×8 | 0.70 | **9.52%** | 5.15% | 82.14% | 39.06% | 52.17% |
| v3 + weight×10 | 0.60 | **14.29%** | 5.15% | 82.14% | 48.44% | 58.70% |
| v3 + weight×10 | 0.65 | **9.52%** | 4.12% | 82.14% | 42.19% | 54.35% |
| v3 + weight×5 | 0.85 | 9.52% | 4.12% | 71.43% | 15.63% | 32.61% |
| v3 + weight×3 | 0.85 | 14.29% | 5.15% | 75.00% | 29.69% | 43.48% |

valid split에서도 동일 패턴 재현(w8@0.65: R_live FRR 8.33%, attack block 58.62%; w10@0.60: R_live FRR 16.67%, attack block 59.77%) — 우연이 아니라 안정적인 trade-off다.

### 9-5. 목표 기준 확인 — 결론: 두 목표를 동시에 만족하는 조합은 없다

R_live FRR을 10~15%까지 낮추려면 threshold를 0.6~0.7까지 올려야 하는데, 그 구간에서 **replay 차단율이 39~48%까지 무너진다** (print는 threshold를 꽤 올려도 82~89%로 비교적 안정적). 그 결과 전체 attack block rate는 52~59%에 머물러 목표(70~80%)에 도달하지 못한다. weight를 8~10까지 올려도 이 trade-off 곡선 자체는 거의 바뀌지 않고(같은 threshold에서 R_live FRR과 replay 차단율이 동시에 소폭 개선되는 정도), **결정적인 병목은 replay 공격과 R_live의 spoof_score 분포가 너무 가깝다는 것**이다 — sampler 가중치로는 분포 자체를 떼어놓지 못한다.

### 9-6. 추천

- **가장 균형 잡힌 조합**: `v3 + weight×8`, threshold≈0.65 — R_live FRR 14.29%(test)/8.33%(valid)로 목표 달성, 단 전체 attack block 58.7%로 목표(70~80%) 미달
- 목표를 둘 다 만족하는 조합이 없으므로, **이 시점에서 ONNX export 및 production promote는 보류**해야 한다 (사용자 지시와 일치)
- 근본적으로 풀려면 sampler 가중치가 아니라 다음이 필요해 보인다:
  1. R_live_clip 데이터 자체를 늘리는 것(93→200+, 6장에서도 권장) — 분포를 재현하는 게 아니라 실제로 채워야 함
  2. replay 공격에 특화된 피처 추가(예: 화면 재생 특유의 moiré/flicker, 프레임레이트 아티팩트) — 현재 피처셋은 움직임 기반이라 "느리게 움직이는 진짜 사람"과 "느리게 움직이는 재생 영상"을 구분하기 어려움
  3. source-weighted 대신 augmentation(time-warp 등, 6장에 준비된 `train_gru_aug.py`)으로 R_live 분포를 늘리는 시도 — 단순 재샘플링이 아니라 변형을 가하므로 다른 효과를 낼 수 있음

### 9-7. 산출물

| 산출물 | 경로 |
|---|---|
| 실험 스크립트 | `model/retrain_aspectfix/train_source_weighted.py` |
| weight×3/5/8/10 학습 결과 | `model/retrain_aspectfix/runs/gru_v3_aspectfix_w{3,5,8,10}/` |
| 각 모델의 threshold sweep (0.20~0.95) | `.../threshold_sweep.csv` |

ONNX export와 production promote는 운영 후보가 확정되기 전까지 보류한다.

---

## 10. 모델 단독 평가에서 "CAPTCHA 시스템 전체" 평가로 (2026-06-17)

### 10-1. 문제 제기

9장 결론(R_live FRR<15%와 replay 차단율>70~80%를 동시에 만족하는 조합 없음, replay 통과율 45~55%)을 두고 "이 상태로 배포해도 되는가"라는 질문이 나왔다. 그런데 실제 서비스는 GRU 모델 단독이 아니라 **① 랜덤 얼굴 미션 ② 랜덤 손동작 미션 ③ 시간제한 ④ anti-spoof 모델 ⑤ 다중 시도 제한**을 같이 쓰는 CAPTCHA 시스템이다. 모델 단독 수치(45~55% 통과)를 시스템 전체의 위험도로 그대로 쓰는 게 맞는지부터 따져봐야 했다.

### 10-2. 1차 분석 — ML팀이 가진 코드만으로는 판단 불가

ML팀이 보유한 `captcha_engine_team_handoff`(다른 팀에 넘긴 레퍼런스 데모 코드)를 까보니, 데모 코드 자체에는 구조적 결함이 있었다:
- `mission_pass = bool(face_detected and spoof_score <= threshold)` — 얼굴 "미션"이 실제로는 spoof_score 재확인일 뿐, 지시받은 동작(왼쪽 눈 감기 등)을 검증하지 않음
- 손동작은 4종류(index_up/two_fingers/open_palm/fist)뿐으로 엔트로피가 낮음(2bit)
- 손 영역과 얼굴 영역이 같은 프레임에서 각자 독립적으로만 검사됨 — "같은 사람의 손과 얼굴인지" 확인하는 로직 없음

하지만 이건 **ML팀이 다른 팀에 넘긴 레퍼런스 데모**일 뿐, 실제 운영 중인 캡차 위젯(별도 레포, ML팀은 코드 접근 권한 없음)이 그대로 그 구조를 쓰는지는 알 수 없었다. 그래서 "데모 코드 기준 결함"과 "실제 운영 위젯의 결함"을 섞어 말하지 않도록 정정하고, `CAPTCHA_ENGINE_VERIFICATION_CHECKLIST.md`라는 별도 문서를 만들어 실제 위젯을 담당하는 쪽에 확인을 요청하는 방향으로 전환했다.

### 10-3. 검증 체크리스트 설계

`CAPTCHA_ENGINE_VERIFICATION_CHECKLIST.md`에 A~F 6개 카테고리, 13개 질문을 정리했다:

| 카테고리 | 질문 | 왜 중요한가 |
|---|---|---|
| A. 얼굴 미션의 실체 | 다양한가? spoof_score와 독립적으로 검증되는가? | **체크리스트에서 가장 무거운 질문** — 독립적이지 않으면 미션 카탈로그가 몇 종이든 보안적으로 무의미 |
| B. 손-얼굴 결합 | 손과 얼굴이 같은 사람인지 확인하는가? | 없으면 "얼굴은 재생 영상, 손은 라이브"로 미션을 우회 가능 |
| C. 손 미션 다양성 | 제스처 종류, 엔트로피 | 풀이 작으면 정적 공격자가 클립 몇 개로 대응 가능 |
| D. 라운드 간 독립성 | 같은 영상으로 3라운드 다 버티는가? | risk 누적 로직(`spoof_detected≥2/3`)의 실효성과 직결 |
| E. 인젝션 방어 | 가상 카메라/파일 업로드를 막는가? | 미션·모델 위 계층을 우회하는 유일한 경로 — 코드만 봐서는 끝까지 확인 불가능했던 항목 |
| F. 모델/로직 버전 | 어느 모델, 어느 판정 로직을 쓰는가 | 9장에서 고친 버그가 실제로 반영됐는지 |

코드를 못 보는 상황을 고려해, 각 질문을 **직접 위젯을 조작해서 결과만 보면 답이 나오는 블랙박스 테스트(G1~G7)**로도 변환해뒀다 (예: G1 — 지시받은 동작을 일부러 안 하고 가만히 있어도 PASS가 뜨는지 확인).

### 10-4. 실제 확인된 결과

서비스 담당자에게 직접 확인받은 내용:

**A1 (미션 다양성) — 확인됨, 양호.** 실제 운영 미션 카탈로그:
- 얼굴 약 10종: 미소 짓기, 왼쪽/오른쪽 눈 감기, 고개 끄덕이기, 도리도리, 입벌리기, 고개 상/하/좌/우 돌리기
- 손동작 약 7종 + 조합형: 브이, 따봉, OK사인, 손가락 번호 지정(조합형이라 변형 다수), 손 흔들기, 손 좌/우 밀기, **손가락으로 좌/우 볼 찌르기**

ML팀 데모(4종)보다 훨씬 풍부하다.

**A2 (spoof_score와의 독립성) — 확인됨, 양호.** "완전 독립적"이라는 확답을 받았다 — 미션 pass/fail이 spoof_score 재탕이 아니라 지시받은 동작을 실제로 검증한다. 체크리스트에서 가장 무겁다고 표시했던 질문이 좋은 쪽으로 해소됐다.

**B (손-얼굴 결합) — 부분적으로만 해소.** "손가락으로 볼 찌르기" 미션은 손이 물리적으로 얼굴에 닿아야 하므로 occlusion 성격의 결합 검증이 가능한 유일한 미션이다. 단, 나머지 6종 손동작에는 이런 공간적 결합이 없어 보이고, "볼 찌르기" 자체가 실제로 위치를 검사하는지(G3-b)도 별도 확인이 필요한 상태로 남아 있다.

**E (인젝션 방어), D (라운드 독립성), F (모델/로직 버전) — 미확인.** 다음 단계의 검증 대상.

### 10-5. 핵심 통계 함정 — "세션 조합의 수" vs "공격자가 준비해야 하는 클립의 수"

미션이 얼굴×손 랜덤 조합이라 세션 단위로 보면 경우의 수가 매우 크다(수천~수만). 하지만 이게 보안적으로 "거의 무한대"를 의미하진 않는다:

- 라운드가 독립적으로 채점되므로, 공격자는 **세션 조합 전체를 미리 준비할 필요가 없다.** 그 순간 뜨는 지시 하나에 맞는 클립 하나만 그때그때 바꿔 끼우면 된다.
- 그러므로 실제로 막아야 하는 건 세션 조합 수가 아니라 **"원자(atomic) 단위 지시 vocabulary의 크기"**다. 얼굴 10종 + 손 6종 + 손가락 지정(현실적으로 1~2개 손가락 지정 패턴이면 C(10,1)+C(10,2)≈55가지) 등을 합치면 대략 **70~100가지 정도**로 추정된다.
- 70~100가지는 "무궁무진"이 아니라, **의지가 있는 공격자가 하루~주말 동안 한 사람을 데려다 클립을 찍으면 만들 수 있는, 유한하고 측정 가능한 작업량**이다.

### 10-6. 최종 결론 — 답은 하나가 아니라 위협 모델에 달려 있다

| 위협 모델 | 배포 가능 여부 | 근거 |
|---|---|---|
| **일반 자동화 봇 방지** (대부분의 CAPTCHA가 실제로 막아야 하는 대상) | **가능** | 시스템 구조상 3라운드 중 얼굴 미션 1회 이상 + 손동작 미션 1회 이상이 보장됨(`captcha_decision.py`의 `has_face_mission`/`has_hand_mission` 체크). 고정 영상 하나로 자동화하는 공격자는 손 라운드(라이브 손 없음) 또는 얼굴 라운드(랜덤 지시에 반응 못함) 중 하나에서 반드시 걸린다. 모델 단독 수치(45~55% 통과)는 이 위협 수준에는 그대로 적용되지 않는다 |
| **표적 공격자** (70~100개 클립 라이브러리를 미리 준비) | **불가** | 클립 각각이 "진짜로 그 동작을 하는" 콘텐츠이므로 미션 검증(A2)을 그대로 통과한다. 이 경우 방어는 결국 spoof_score 하나로 좁아지고, 그 성능이 9장에서 측정한 45~55% 통과율이다 — 미션이 도움이 안 됨 |

**두 결론 모두에 걸리는 공통 변수**: E(인젝션/가상카메라 방어)가 아직 미확인이다. 이게 뚫려 있으면 "일반 봇 방지용으로는 가능"이라는 결론도 무효가 된다 — 실시간 합성 영상 주입은 미션 다양성 자체를 무력화하기 때문이다.

### 10-7. 실무 권장사항

1. 서비스팀과 **"이 캡차가 막아야 하는 공격 수준"을 문서로 명시**한다 (예: "일반 자동화 봇 차단용. 표적 딥페이크·인젝션 공격은 보증 범위 밖"과 같은 한 줄).
2. 그 범위가 일반 봇 방지 수준이면 **지금 모델(v3 + source-weight×8, threshold≈0.65)로 배포 가능** — 모델을 더 개선하는 것보다 ROI가 높은 선택.
3. **E(인젝션 방어)는 배포 여부와 무관하게 최우선으로 확인**한다.
4. G3/G3-b(손-얼굴 결합 블랙박스 테스트)로 B를 마무리 확인한다.
5. 표적 공격까지 막아야 하는 use case라면 이 모델로는 부족하고, R_live 데이터 추가 수집 + 모델 재설계(6장, 9-6)가 먼저다.

### 10-8. 산출물

| 산출물 | 경로 |
|---|---|
| 캡차 엔진 검증 체크리스트 (A~F 질문 + G1~G7 블랙박스 테스트 + 후속조치 매핑) | `CAPTCHA_ENGINE_VERIFICATION_CHECKLIST.md` |
| 이번 세션 전체 인수인계 요약 | `HANDOFF_facial_recognition_aspectfix.md` |
