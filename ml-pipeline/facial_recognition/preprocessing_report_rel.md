# 얼굴 위변조 탐지 — 전처리 v2 보고서 (상대좌표)

**작성일**: 2026-06-12  
**환경**: Python 3.10 / MediaPipe 0.10.9 / OpenCV / pandas  
**스크립트**: `preprocessing/extract_features_rel.py`  
**출력 파일**: `features/face_clip_data_rel.npz`

---

## 1. v1 대비 변경사항

| 항목 | v1 (`face_clip_data.npz`) | v2 (`face_clip_data_rel.npz`) |
|---|---|---|
| 위치 좌표 | 절대좌표 (`nose_x`, `nose_y`, `cx`, `cy`) | 시퀀스 평균 기준 상대좌표 (`nose_x_rel` 등) |
| bbox 정규화 | 없음 | 얼굴 폭(`face_w`) 기준 정규화 |
| velocity 기준 | 절대좌표 차분 | 정규화 상대좌표 차분 (동일 값, 의미 명확화) |
| 얼굴 이동량 정규화 | 누적 합 | 프레임 수로 나눔 (클립 길이 편차 제거) |
| face_detect_rate 임계 | 없음 | 유효 프레임 < 3개 클립 제거 |
| 피처 수 (seq) | 20개 | 20개 (좌표 피처 상대화) |
| seq clipping | 없음 | train set 1~99% percentile |

**핵심 변경 이유**:  
`nose_x`, `cx` 같은 절대좌표는 "얼굴이 화면 어디에 있냐"를 반영하므로 카메라 거리·위치 조건에 민감하다.  
상대좌표 `nose_x_rel = nose_x - mean(nose_x)` 로 변환하면 클립 내 얼굴 이동량만 남아 카메라 환경 편차에 강해진다.

---

## 2. 전처리 파이프라인

```
samples_manifest.jsonl  (966 clips, 프레임 경로 목록)
        │
        ▼
    각 clip의 frame JPG 목록
        │
        ├─ cv2.imread → BGR→RGB
        │
        ├─ MediaPipe FaceMesh (video mode, tracking 활성)
        │    max_num_faces=1, refine_landmarks=True
        │    min_detection_confidence=0.5
        │
        ├─ 검출 실패 프레임 → nearest-neighbor 보간
        │    (유효 프레임 < 3개 → clip 제거)
        │
        ├─ extract_frame_raw()
        │    bbox(face_w) 정규화 → ear, mar, smile_w, nose_x, nose_y, cx, cy, roll, yaw, pitch
        │
        ├─ build_seq_array()
        │    nose_x_rel = nose_x - mean(nose_x)  등 상대좌표 변환
        │    velocity = 상대좌표 차분 (nose_dx, nose_dy, ...)
        │    최대 16프레임, 부족하면 zero-padding
        │
        └─ aggregate_static()
             16개 집계 피처 (blink_count, std, mean 등)
        │
        ▼  train set 기준 1~99% percentile clipping
        │
        ▼
face_clip_data_rel.npz
```

---

## 3. 실행 결과 요약

| 항목 | 값 |
|---|---|
| 입력 클립 | 966개 |
| 성공 | **957개 (99.1%)** |
| 실패 (유효 프레임 < 3) | 9개 (0.9%) |
| 출력 파일 | `features/face_clip_data_rel.npz` |
| 얼굴 검출률 평균 | **99.35%** |

### 출력 배열 구조

| 배열 | shape | dtype | 설명 |
|---|---|---|---|
| `x_seq` | (957, 16, 20) | float32 | GRU 입력 시퀀스 피처 |
| `x_static` | (957, 16) | float32 | SVM/RF/ET 비교용 집계 피처 |
| `y` | (957,) | int64 | 레이블 (0=real, 1=spoof) |
| `seq_lengths` | (957,) | int32 | 실제 유효 프레임 수 (padding 전) |
| `face_detect_rates` | (957,) | float32 | 클립별 얼굴 검출 성공률 |
| `sample_ids` | (957,) | object | 클립 고유 ID |
| `subject_ids` | (957,) | object | 피험자 ID |
| `splits` | (957,) | object | train / valid / test |
| `source_groups` | (957,) | object | S_dataset_sequence / R_live_clip / ATK_external_clip |
| `attack_types` | (957,) | object | live / print / replay |
| `devices` | (957,) | object | 촬영 기기 정보 |
| `seq_feature_names` | (20,) | object | 시퀀스 피처 이름 순서 |
| `static_feature_names` | (16,) | object | 정적 피처 이름 순서 |

---

## 4. 데이터 분포

### 4-1. split × label

| split | 전체 | real (0) | spoof (1) |
|---|---|---|---|
| train | 610 | 371 | 239 |
| valid | 166 | 83 | 83 |
| test | 181 | 93 | 88 |
| **합계** | **957** | **547** | **410** |

> valid / test는 real:spoof = 1:1 균형 유지. train은 real이 1.55배 많아 `pos_weight`로 보정.

### 4-2. attack_type × split

| attack_type | train | valid | test | 합계 |
|---|---|---|---|---|
| live | 371 | 83 | 93 | 547 |
| replay | 151 | 58 | 60 | 269 |
| print | 88 | 25 | 28 | 141 |
| **합계** | **610** | **166** | **181** | **957** |

### 4-3. source_group × split

| source_group | train | valid | test | 합계 |
|---|---|---|---|---|
| S_dataset_sequence | 422 | 126 | 128 | 676 |
| ATK_external_clip | 128 | 28 | 32 | 188 |
| R_live_clip | 60 | 12 | 21 | 93 |
| **합계** | **610** | **166** | **181** | **957** |

> `R_live_clip`: 93개 전부 real(live). train 60 / valid 12 / test 21.

---

## 5. 시퀀스 길이 분포

| 프레임 수 | 클립 수 | 비고 |
|---|---|---|
| 3 | 7 | 최소 (보간 후 MIN_VALID_FRAMES 간신히 충족) |
| 4 | 9 | |
| 5 | 34 | |
| 6 | 55 | |
| 7 | 145 | |
| 8~15 | 263 | |
| **16 (최대)** | **444** | **전체의 46.4% — 풀 시퀀스** |
| **합계** | **957** | |

- 평균 11.96 프레임 / 중앙값 14 / 최소 3 / 최대 16  
- `SEQ_LENGTH=16` 미만 클립은 zero-padding 후 `seq_lengths`에 실제 길이 저장

---

## 6. 얼굴 검출률 분포

| 임계값 | 클립 수 | 비율 |
|---|---|---|
| ≥ 1.00 (전 프레임 성공) | 841 | 87.9% |
| ≥ 0.90 | 941 | 98.3% |
| ≥ 0.80 | 957 | 100.0% |

- 최저 검출률: **0.80** (모든 클립이 80% 이상)
- 검출 실패 프레임은 nearest-neighbor 보간으로 대체됨

---

## 7. 스킵 클립 (9개)

| 이유 | 건수 |
|---|---|
| `insufficient_frames` (유효 프레임 < 3) | 9 |

| sample_id | label | frame_count | valid_count |
|---|---|---|---|
| S064_sess2_t1_il2_dev2 | real | 2 | 2 |
| S004_spoof_sess2_t3_il4_dev1 | spoof | 1 | 1 |
| S004_spoof_sess2_t3_il4_dev2 | spoof | 2 | 2 |
| S012_spoof_sess2_t3_il4_dev2 | spoof | 2 | 2 |
| S013_spoof_sess2_t3_il4_dev1 | spoof | 1 | 1 |
| S015_spoof_sess2_t3_il4_dev2 | spoof | 2 | 2 |
| S018_spoof_sess2_t3_il4_dev1 | spoof | 1 | 1 |
| S019_spoof_sess2_t3_il4_dev1 | spoof | 1 | 1 |
| S019_spoof_sess2_t3_il4_dev2 | spoof | 2 | 2 |

> 전부 프레임 수 1~2개짜리 클립. 보간 후에도 `MIN_VALID_FRAMES=3` 미충족으로 제거.  
> spoof 8개 + real 1개 → 학습 데이터에 미치는 영향 미미.

---

## 8. 피처 정의

### 8-1. 시퀀스 피처 20개 (`x_seq`)

#### 위치 / 표정

| 번호 | 피처명 | 계산 | 설명 |
|---|---|---|---|
| 1 | `ear` | `(EAR_left + EAR_right) / 2` | 눈 뜨임 정도 |
| 2 | `mar` | `수직(13-14) / 수평(61-291)` | 입 벌림 정도 |
| 3 | `smile_w` | `mouth_width / face_w` | 입 가로폭 비율 (미소 변화) |
| 4 | `nose_x_rel` | `nose_x_bbox - mean(nose_x_bbox)` | 코 x, 클립 평균 기준 상대좌표 |
| 5 | `nose_y_rel` | `nose_y_bbox - mean(nose_y_bbox)` | 코 y, 클립 평균 기준 상대좌표 |
| 6 | `cx_rel` | `cx_bbox - mean(cx_bbox)` | 얼굴 중심 x, 상대좌표 |
| 7 | `cy_rel` | `cy_bbox - mean(cy_bbox)` | 얼굴 중심 y, 상대좌표 |
| 8 | `roll` | `atan2(Δeye_y, Δeye_x)` degrees | 고개 좌우 기울기 |
| 9 | `yaw` | `(nose_x - face_cx) / half_face_w` | 고개 좌우 회전 |
| 10 | `pitch` | `(nose_y - eye_mouth_mid_y) / half_face_h` | 고개 상하 회전 |

> **bbox 정규화**: `x_bbox = (x_raw - face_cx) / face_w`  
> 얼굴 폭을 기준으로 정규화하므로 카메라 거리·얼굴 크기에 무관.

#### 변화량 (velocity)

| 번호 | 피처명 | 계산 | 설명 |
|---|---|---|---|
| 11 | `nose_dx` | `nose_x_rel[t] - nose_x_rel[t-1]` | 코 x 변화량 |
| 12 | `nose_dy` | `nose_y_rel[t] - nose_y_rel[t-1]` | 코 y 변화량 |
| 13 | `center_dx` | `cx_rel[t] - cx_rel[t-1]` | 얼굴 중심 x 변화량 |
| 14 | `center_dy` | `cy_rel[t] - cy_rel[t-1]` | 얼굴 중심 y 변화량 |
| 15 | `nose_speed` | `hypot(nose_dx, nose_dy)` | 코 이동 속도 |
| 16 | `ear_velocity` | `ear[t] - ear[t-1]` | 눈 뜨임 변화량 |
| 17 | `mar_velocity` | `mar[t] - mar[t-1]` | 입 벌림 변화량 |
| 18 | `yaw_velocity` | `yaw[t] - yaw[t-1]` | yaw 변화량 |
| 19 | `pitch_velocity` | `pitch[t] - pitch[t-1]` | pitch 변화량 |
| 20 | `roll_velocity` | `roll[t] - roll[t-1]` | roll 변화량 |

> 첫 프레임(t=0)의 velocity는 0으로 처리.

### 8-2. 정적 피처 16개 (`x_static`)

| 번호 | 피처명 | 설명 |
|---|---|---|
| 1 | `blink_count` | EAR < 0.20 구간 진입 횟수 |
| 2 | `eye_open_ratio` | EAR 클립 평균 |
| 3 | `mouth_open_ratio` | MAR 클립 평균 |
| 4 | `smile_ratio` | 입 가로폭 비율의 std |
| 5 | `head_yaw` | yaw 시계열 std |
| 6 | `head_pitch` | pitch 시계열 std |
| 7 | `head_roll` | roll 시계열 std (degrees) |
| 8 | `face_movement` | 얼굴 중심 이동량 합 / (n-1) |
| 9 | `face_stability` | 얼굴 중심 위치의 √(σx²+σy²) |
| 10 | `ear_std` | EAR 시계열 std |
| 11 | `mar_mean` | MAR 클립 평균 |
| 12 | `mar_std` | MAR 시계열 std |
| 13 | `nose_movement` | 코 이동량 합 / (n-1) |
| 14 | `head_yaw_mean` | yaw 클립 평균 |
| 15 | `head_pitch_mean` | pitch 클립 평균 |
| 16 | `head_roll_mean` | roll 클립 평균 |

---

## 9. 피처 판별력 분석 (train set, Cohen's d)

Cohen's d 기준: **d < 0.2** 소효과 / **0.2 ≤ d < 0.5** 중효과 / **d ≥ 0.5** 대효과

### 9-1. 시퀀스 피처 (프레임 단위 기준)

| 피처 | real 평균 | spoof 평균 | Cohen's d | 효과 |
|---|---|---|---|---|
| `ear` | 0.3148 | 0.4089 | **0.732** | 대 |
| `smile_w` | 0.3683 | 0.3481 | **0.508** | 대 |
| `nose_speed` | 0.1201 | 0.0827 | 0.306 | 중 |
| `pitch` | -0.0515 | -0.0332 | 0.119 | 소 |
| `mar` | 0.0298 | 0.0421 | 0.149 | 소 |
| `ear_velocity` | -0.0005 | 0.0024 | 0.035 | 소 |
| `nose_x_rel` | ~0.000 | ~0.000 | 0.003 | 소 |
| `cx_rel` | ~0.000 | ~0.000 | 0.003 | 소 |

> `nose_x_rel`, `cx_rel` 등 상대좌표는 프레임 평균이 0에 수렴하므로 mean 차이 없음.  
> 판별력은 **분산(std) 차이**에 있음 — spoof는 절대좌표가 고정되어 std가 낮음.

### 9-2. 정적 피처 (클립 단위 기준)

| 피처 | real 평균 | spoof 평균 | Cohen's d | 효과 |
|---|---|---|---|---|
| `head_pitch` | 0.1199 | 0.0681 | **0.746** | 대 |
| `blink_count` | 0.873 | 0.418 | **0.551** | 대 |
| `eye_open_ratio` | 0.3055 | 0.3654 | **0.543** | 대 |
| `head_roll` | 1.219 | 1.966 | **0.506** | 대 |
| `nose_movement` | 0.1359 | 0.0972 | 0.434 | 중 |
| `face_movement` | 0.0763 | 0.0552 | 0.415 | 중 |
| `smile_ratio` | 0.0201 | 0.0169 | 0.229 | 중 |
| `mouth_open_ratio` | 0.0309 | 0.0402 | 0.191 | 소 |
| `head_yaw` | 0.2691 | 0.2479 | 0.097 | 소 |

> **v2 주요 관찰**: `head_pitch` (d=0.746) — real은 고개가 상하로 더 많이 움직임.  
> `blink_count` (d=0.551), `eye_open_ratio` (d=0.543) — spoof의 EAR이 real보다 높은 것은  
> print/replay 영상에서 눈을 크게 뜬 상태가 많기 때문으로 추정.

---

## 10. v1 vs v2 학습 성능 비교

**모델 설정**: GRU hidden=32, lr=0.0005, epochs=80, patience=15, feature_mode=all

| 지표 | v1 (절대좌표, threshold=0.21) | v2 (상대좌표, threshold=0.08) |
|---|---|---|
| val AUC (전체) | — | **0.862** |
| test AUC (전체) | — | **0.813** |
| **val AUC (S_dataset_sequence)** | 0.889 | **0.907** ↑ |
| **test AUC (S_dataset_sequence)** | 0.937 | **0.908** ↓ |
| val F1 (ATK_external_clip) | 0.982 | 0.923 ↓ |
| test F1 (ATK_external_clip) | 0.968 | 0.933 ↓ |
| **val FRR (R_live_clip)** | 0.667 | **0.833** ↑ 악화 |
| **test FRR (R_live_clip)** | 0.952 | **0.952** 동일 |

### R_live_clip 미개선 원인 분석

R_live_clip은 93개 전부 real(live)이며 train에 60개가 포함되어 있음에도 FRR이 83~95%로 매우 높다.

상대좌표 변환이 효과가 없는 이유:
1. **촬영 환경 분포 차이**: R_live_clip은 야외·실환경에서 다양한 기기로 촬영된 영상.  
   S_dataset_sequence (통제 환경)와 feature 분포가 근본적으로 다름.
2. **train set 불균형**: train 610개 중 R_live_clip은 60개(9.8%)뿐.  
   모델이 S_dataset_sequence 중심으로 학습되어 R_live_clip을 일반화하지 못함.
3. **좌표 기준이 아닌 texture/움직임 패턴 차이**: 위치 상대화만으로는 촬영 조건 편차를 제거할 수 없음.

---

## 11. 다음 단계 권장사항

```
1. [단기] source-weighted training
   → R_live_clip train 60개에 가중치 3~5배 적용
   → src/train_gru_source_weighted.py 사용

2. [단기] augmentation
   → R_live_clip 클립에 time-warp, noise 추가
   → src/train_gru_aug.py 사용

3. [중기] R_live_clip 피처 분포 시각화
   → PCA / t-SNE 로 R_live vs S_dataset 분리 정도 확인
   → 어떤 피처에서 분포가 벌어지는지 특정

4. [중기] 추가 데이터 수집
   → R_live_clip 유사 조건의 실환경 클립 보강
   → 현재 93개 → 최소 200개 이상 권장

5. [현상 유지] 서비스 목적 맞게 threshold 조정
   → R_live_clip FRR 허용 수준과 ATK FAR 목표를 trade-off로 결정
```

---

## 12. 생성 파일 목록

| 파일 | 크기 | 설명 |
|---|---|---|
| `features/face_clip_data_rel.npz` | 866 KB | 전처리 결과 (메인 출력) |
| `features/skipped_clips_rel.csv` | — | 제외 클립 9개 목록 |
