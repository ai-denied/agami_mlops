# 얼굴 인식 전처리 파이프라인 실행 보고서

**실행일**: 2026-06-08  
**실행 환경**: Python 3.10 / MediaPipe 0.10.21 / OpenCV 4.11.0  
**데이터셋 경로**: `facial_recognition/dataset/`

---

## 1. 파이프라인 실행 결과

| 스크립트 | 대상 | 총 파일 | 성공 | 실패 | 성공률 |
|---|---|---|---|---|---|
| `extract_image_features.py` | `dataset/face_images/` | 9,627장 | 9,591장 | 36장 | 99.6% |
| `extract_video_features.py` | `dataset/face_videos/` | 56개 | 56개 | 0개 | 100.0% |

### 생성된 파일

```
features/
├── face_image_features.csv   (9,591행 × 10열)
├── face_video_features.csv   (56행 × 11열)
└── skipped_files.csv         (36행 × 3열)
```

---

## 2. 생성 CSV 구조

### `face_image_features.csv` (이미지)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `file_path` | str | 원본 파일 경로 |
| `label` | int | 0=real, 1=spoof |
| `ear_left` | float | 왼쪽 눈 비율 (6점 EAR 공식) |
| `ear_right` | float | 오른쪽 눈 비율 |
| `ear_avg` | float | 양안 평균 EAR |
| `mar` | float | 입 벌림 비율 |
| `nose_x` | float | 정규화 코 끝 X 좌표 |
| `nose_y` | float | 정규화 코 끝 Y 좌표 |
| `face_width` | float | 정규화 얼굴 폭 |
| `face_height` | float | 정규화 얼굴 높이 |

### `face_video_features.csv` (영상, 5프레임 샘플링 → 영상 1개 = 1행)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `file_path` | str | 원본 파일 경로 |
| `label` | int | 0=real, 1=spoof |
| `ear_mean` | float | EAR 평균 |
| `ear_std` | float | EAR 표준편차 (눈 깜빡임 변동성) |
| `mar_mean` | float | MAR 평균 |
| `mar_std` | float | MAR 표준편차 (입 움직임 변동성) |
| `nose_movement` | float | 코 끝 총 이동 거리 |
| `face_stability` | float | 얼굴 중심 위치 표준편차 √(σx²+σy²) |
| `blink_count` | int | EAR < 0.20 구간 진입 횟수 |
| `valid_frame_count` | int | 얼굴 검출 성공 프레임 수 |
| `face_detect_rate` | float | valid_frame_count / total_sampled |

---

## 3. 레이블 분포 분석

### 이미지

| 클래스 | 샘플 수 | 비율 |
|---|---|---|
| real (0) | 7,134 | 74.4% |
| spoof (1) | 2,457 | 25.6% |
| **합계** | **9,591** | **100%** |

**비율**: real : spoof = 2.9 : 1 → **불균형 존재**

### 영상

| 클래스 | 샘플 수 | 비율 |
|---|---|---|
| real (0) | 29 | 51.8% |
| spoof (1) | 27 | 48.2% |
| **합계** | **56** | **100%** |

**비율**: real : spoof ≈ 1.07 : 1 → **균형**

---

## 4. 결측치 분석

| CSV | 결측치 |
|---|---|
| `face_image_features.csv` | **없음** |
| `face_video_features.csv` | **없음** |

결측치 없음. 별도 처리 불필요.

---

## 5. Feature 통계 분석

### 이미지 — real vs spoof 비교

| Feature | real (mean ± std) | spoof (mean ± std) | 차이 |
|---|---|---|---|
| `ear_avg` | 0.2741 ± 0.0740 | 0.2654 ± 0.0598 | 미미 |
| `mar` | 0.0425 ± 0.0900 | 0.0525 ± 0.0906 | 미미 |
| `nose_x` | 0.5059 ± 0.0800 | 0.5006 ± 0.0745 | 무시 |
| `nose_y` | 0.5240 ± 0.0706 | 0.5650 ± 0.0835 | **중간** |
| `face_width` | 0.8308 ± 0.1394 | 0.7502 ± 0.1813 | **중간** |
| `face_height` | 0.9864 ± 0.1508 | 0.8767 ± 0.2057 | **중간** |

> 이미지 단일 프레임 feature는 real/spoof 간 평균 차이가 작음.  
> 얼굴 크기(face_width, face_height)와 위치(nose_y)에서 부분적 구분 가능.

### 영상 — real vs spoof 비교 (Cohen's d 포함)

| Feature | real (mean) | spoof (mean) | Cohen's d | 해석 |
|---|---|---|---|---|
| `ear_std` | 0.0655 | 0.0322 | **0.891** | 대 — real은 눈 깜빡임 변동이 2배 |
| `nose_movement` | 2.1220 | 1.2355 | **1.319** | 대 — real은 더 많이 움직임 |
| `face_stability` | 0.1108 | 0.0779 | **1.038** | 대 — spoof는 얼굴 위치가 고정적 |
| `blink_count` | 3.97 | 2.56 | 0.559 | 중간 — real이 더 많이 깜빡임 |
| `face_detect_rate` | 0.9588 | 0.8597 | 0.681 | 중간 — real 검출률 더 높음 |
| `mar_std` | 0.0163 | 0.0265 | 0.256 | 소 |

> **핵심 판별 feature**: `nose_movement` (d=1.319) > `face_stability` (d=1.038) > `ear_std` (d=0.891)  
> 세 feature 모두 Cohen's d > 0.8 → 대효과 크기. **영상 기반 분류에 강한 신호.**

---

## 6. skipped_files 분석

| 항목 | 값 |
|---|---|
| 총 스킵 파일 | 36개 (이미지) |
| 스킵 이유 | `no_face_detected` 36건 (100%) |
| real 스킵 | 23개 (63.9%) |
| spoof 스킵 | 13개 (36.1%) |

- 스킵 비율: 36 / 9,627 = **0.37%** (무시 가능한 수준)
- 스킵 이유 전체가 `no_face_detected` → 이미지 품질 문제 (측면, 가림, 저해상도 등)
- 영상은 스킵 **0건**

---

## 7. 이상치 분석

| 조건 | 해당 수 | 비율 | real/spoof |
|---|---|---|---|
| `ear_avg` < 0.05 또는 > 0.6 | 68개 | 0.71% | 62 real / 6 spoof |
| `mar` > 0.5 | 51개 | 0.53% | 41 real / 10 spoof |

- 이상치 비율 **1% 미만** → 학습에 미치는 영향 미미
- real에 이상치가 집중: 눈 감은 순간, 크게 웃는 표정 등 자연스러운 동작에서 발생
- 별도 클리닝 없이 진행 가능하나, 필요 시 `ear_avg < 0.05` 68건 제거 검토

---

## 8. 학습 가능 여부 평가

### 체크리스트

| 항목 | 이미지 | 영상 |
|---|---|---|
| 결측치 없음 | ✅ | ✅ |
| 최소 샘플 충족 (>500 / >20) | ✅ 9,591개 | ✅ 56개 |
| 레이블 인코딩 정상 (0/1) | ✅ | ✅ |
| 검출 성공률 > 95% | ✅ 99.6% | ✅ 100% |
| Feature 간 판별력 존재 | ⚠️ 약함 | ✅ 강함 |
| 클래스 균형 (>0.5 기준) | ⚠️ 0.34 불균형 | ✅ 0.93 균형 |

### 최종 판정

| 데이터 | 상태 | 비고 |
|---|---|---|
| **이미지** | ✅ **READY** (조건부) | 불균형 대응 필요 |
| **영상** | ✅ **READY** | 소규모이나 균형·판별력 우수 |

---

## 9. 학습 전 권장 조치

### 이미지 (필수)
- **클래스 불균형 처리**: real:spoof = 2.9:1 → 아래 중 하나 선택
  - `class_weight='balanced'` (sklearn) 또는 `pos_weight` (PyTorch)
  - spoof 오버샘플링 (RandomOverSampler / SMOTE)
  - real 언더샘플링 (7,134 → 2,457)

### 이미지 (선택)
- `ear_avg < 0.05` 68개 제거 (눈 완전히 감긴 이미지)
- `train_test_split` 시 `stratify=label` 필수

### 영상 (선택)
- 샘플 56개는 전통 ML(SVM, RandomForest, XGBoost) 권장
- 딥러닝 적용 시 Leave-One-Subject-Out 교차 검증 필요
- 핵심 3개 feature(`nose_movement`, `face_stability`, `ear_std`)로 경량 모델 우선 시도

---

## 10. 다음 단계

```
1. 이미지: train/val/test split (stratify=label, 70:15:15)
2. 이미지: class_weight 설정 후 분류기 학습 (LightGBM, SVM 등)
3. 영상: 핵심 3-feature 기반 SVM 또는 Random Forest 학습
4. 평가지표: Accuracy + F1-score(macro) + ROC-AUC 병행
5. 영상 샘플 부족 시 이미지 모델 기반 앙상블 검토
```
