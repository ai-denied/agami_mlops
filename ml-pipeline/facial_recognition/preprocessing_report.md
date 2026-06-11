# 얼굴 위변조 탐지 — 전처리 파이프라인 보고서

**작성일**: 2026-06-10  
**환경**: Python 3.10 / MediaPipe 0.10.5 / OpenCV / pandas  
**데이터셋 경로**: `facial_recognition/dataset/`

---

## 1. 전처리 파이프라인 개요

```
dataset/
├── face_images/          ─┐
│   ├── real/             │  extract_image_features.py
│   └── spoof/            │       ↓
│                         │  features/face_clip_features.csv
└── samples_manifest.jsonl┘       ↓ (변환)
                                  features/face_clip_features.jsonl

dataset/
└── face_videos/          ─── extract_video_features.py
    ├── real/                      ↓
    └── spoof/             features/face_video_features.csv
```

전처리는 두 스크립트로 구성됩니다.

| 스크립트 | 입력 | 출력 | 처리 단위 |
|---|---|---|---|
| `extract_image_features.py` | `face_images/` + `samples_manifest.jsonl` | `face_clip_features.csv` | clip (프레임 묶음) |
| `extract_video_features.py` | `face_videos/` | `face_video_features.csv` | video (mp4 1개) |

---

## 2. 데이터셋 구조 분석

### 2-1. face_images (정지 이미지)

```
face_images/
├── real/
│   ├── live_001/ ~ live_110/          # 57 subject — CASIA-style
│   │   └── {prefix}_{ID}-{S}-{T}-{I}-{D}_{ms}.jpg
│   └── real_01_phone/ ~ real_09_*/   # 9 subject × 디바이스 — 순번 파일명
└── spoof/
    ├── spoof_001/ ~ spoof_019/        # 12 subject (live_XXX와 동일인물 쌍)
    │   └── {prefix}_{ID}-{S}-{T}-{I}-{D}_{ms}.jpg
    └── attack_01_* ~ attack_08_*/    # 8개 공격유형 폴더 — 익명 클립
```

**파일명 세그먼트 해독 (CASIA-style)**

```
live_001 - 1 - 1 - 1 - 1 _ 120.jpg
  │         │   │   │   │   └─ frame timestamp (ms, 60ms 간격)
  │         │   │   │   └───── device        (1 | 2)
  │         │   │   └───────── illumination  (1 | 2)
  │         │   └───────────── type_code     (1=live, 2=print, 3=replay)
  │         └───────────────── session       (1 | 2)
  └─────────────────────────── subject_id    (001)
```

### 2-2. face_videos (영상)

```
face_videos/
├── real/   live_video.mp4 ~ live_video29.mp4    (29개)
└── spoof/  print_video*.mp4   (9개)
            cut_print_video*.mp4 (9개)
            replay_video*.mp4   (9개)             (총 27개)
```

---

## 3. samples_manifest.jsonl 생성

### 생성 규칙

이미지 전처리의 핵심 선행 작업으로, **clip** 단위 manifest를 생성했습니다.

**clip 정의**: 동일 `(subject_id, session, type_code, illumination, device)` 조합의 프레임 묶음

| 폴더 유형 | clip 정의 | subject_id |
|---|---|---|
| `live_XXX/`, `spoof_XXX/` | 파일명 세그먼트 그루핑 (S,T,I,D 조합) | `S001` ~ `S110` |
| `real_XX_device/` | 30프레임씩 청크 분할 | `R01` ~ `R09` |
| `attack_XX_*/` | 30프레임씩 청크 분할 | `null` (익명) |

**clip 예시 (JSONL 1행)**

```json
{
  "sample_id": "S001_sess1_t1_il1_dev1",
  "subject_id": "S001",
  "namespace": "S",
  "split": "train",
  "label": 0,
  "attack_type": "live",
  "source_folder": "real/live_001",
  "frames": [
    "real/live_001/live_001-1-1-1-1_60.jpg",
    "real/live_001/live_001-1-1-1-1_120.jpg"
  ],
  "frame_count": 8,
  "session": 1,
  "illumination": 1,
  "device": 1,
  "data_source": "casia_style"
}
```

### Subject-Aware Split

subject leakage 방지를 위해 **subject 단위로 split**을 할당했습니다.

- `live_001`과 `spoof_001`은 동일 피험자 → 반드시 같은 split
- S-namespace 57명, R-namespace 9명은 각각 독립 분리
- attack_XX (익명): attack_type별 stratified random split

| Split | clip 수 | real | spoof |
|---|---|---|---|
| train | 614 | 371 | 243 |
| valid | 169 | 84 | 85 |
| test  | 183 | 93 | 90 |
| **합계** | **966** | **548** | **418** |

**Leakage 검증 결과**: train ∩ valid = ∅, train ∩ test = ∅, valid ∩ test = ∅ ✅

---

## 4. 피처 추출 방법론

### 사용 도구

**MediaPipe FaceMesh** — 468개 3D 얼굴 랜드마크 검출

```
static_image_mode  = True   (이미지) / False (영상)
max_num_faces      = 1
refine_landmarks   = True
min_detection_confidence = 0.5
```

### 이미지 피처 추출 흐름 (clip 단위)

```
clip의 frame 목록
    │
    ├─ 각 프레임: cv2.imread → BGR→RGB 변환
    │                 ↓
    │           FaceMesh.process()
    │                 ↓
    │     랜드마크 468점 (정규화 좌표 0~1)
    │                 ↓
    │     프레임 단위 피처 계산
    │     (EAR, MAR, head pose, face center)
    │
    └─ 클립 단위 집계 (mean, std, 누적 이동량)
                 ↓
         face_clip_features.csv 1행 추가
```

### 영상 피처 추출 흐름

```
mp4 파일
    │
    └─ 5프레임마다 1프레임 샘플링
           ↓
       FaceMesh.process() (video mode — 트래킹 활성)
           ↓
       집계 → face_video_features.csv 1행
```

### 랜드마크 인덱스 정의

| 부위 | 인덱스 | 용도 |
|---|---|---|
| 왼쪽 눈 | [33, 160, 158, 133, 153, 144] | EAR 계산 |
| 오른쪽 눈 | [362, 385, 387, 263, 373, 380] | EAR 계산 |
| 입 (좌·우·상·하) | 61, 291, 13, 14 | MAR 계산 |
| 코 끝 | 4 | head pose yaw/pitch, 이동량 |
| 얼굴 좌·우 끝 | 234, 454 | 얼굴 폭, yaw 계산 |
| 눈 바깥 코너 (좌·우) | 33, 263 | roll 계산, 얼굴 중심 |

---

## 5. 컬럼 상세 설명 — face_clip_features.jsonl

### 5-1. 메타 컬럼 (11개)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `sample_id` | str | clip 고유 식별자. `{subject_id}_sess{S}_t{T}_il{I}_dev{D}` 형식 |
| `subject_id` | str \| null | 피험자 ID. S-namespace: `S001`~`S110`, R-namespace: `R01`~`R09`, 익명 attack 클립: `null` |
| `split` | str | 데이터 분할. `train` / `valid` / `test`. subject 단위로 할당되어 leakage 없음 |
| `label` | int | 정답 레이블. `0` = 실제 얼굴(real), `1` = 위변조(spoof) |
| `attack_type` | str | 공격 유형. `live` / `print` / `replay` |
| `session` | float \| null | 촬영 세션 번호 (1 또는 2). CASIA-style 클립만 해당, 나머지 `null` |
| `illumination` | float \| null | 조명 조건 코드 (1 또는 2). CASIA-style 클립만 해당 |
| `device` | str \| null | 촬영 디바이스 코드. CASIA-style: `"1"` 또는 `"2"`, 디바이스 캡처: `"phone"` 등 |
| `frame_count` | int | clip을 구성하는 전체 프레임 수 |
| `valid_frame_count` | int | FaceMesh 랜드마크 검출에 성공한 프레임 수 |
| `face_detect_rate` | float | `valid_frame_count / frame_count`. 클립 품질 지표 (0~1) |

### 5-2. 피처 컬럼 (16개)

#### 눈 관련

| 컬럼 | 단위 | 계산 공식 | 해석 |
|---|---|---|---|
| `blink_count` | 회 | EAR < 0.20 구간 진입 횟수 | clip 내 눈 깜빡임 횟수. real은 자연스러운 깜빡임, spoof는 정지 사진이므로 낮음 |
| `eye_open_ratio` | 0~1 | `mean(EAR_left, EAR_right)` — clip 평균 | 눈이 열린 정도 평균. `1`에 가까울수록 눈이 크게 뜬 상태 |
| `ear_std` | 0~1 | `std(EAR 시계열)` | clip 내 눈 열림 변동성. real은 깜빡임으로 인해 std 높음 |

**EAR 공식 (Eye Aspect Ratio)**:

```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
      수직 거리 합                수평 거리
```

- EAR ≈ 0.25 → 눈 정상적으로 뜸
- EAR < 0.20 → 눈 감은 상태 (깜빡임 임계값)

#### 입 관련

| 컬럼 | 단위 | 계산 공식 | 해석 |
|---|---|---|---|
| `mouth_open_ratio` | 0~1 | `mean(MAR)` — clip 평균 | 입 벌림 정도. 말하거나 표정 변화 시 상승 |
| `mar_mean` | 0~1 | `mouth_open_ratio`와 동일 | 동일 값 (명시적 보조 컬럼) |
| `mar_std` | 0~1 | `std(MAR 시계열)` | clip 내 입 움직임 변동성 |
| `smile_ratio` | 0~1 | `std(입꼬리너비/얼굴너비 시계열)` | 입꼬리 변화량. 0에 가까울수록 무표정 유지 |

**MAR 공식 (Mouth Aspect Ratio)**:

```
MAR = 수직 거리(13-14) / 수평 거리(61-291)
```

#### 머리 자세 (Head Pose)

| 컬럼 | 단위 | 계산 공식 | 해석 |
|---|---|---|---|
| `head_yaw` | 무차원 | `std(yaw 시계열)` | 고개 좌우 회전 변화량. real은 자연스럽게 움직여 std 높음 |
| `head_pitch` | 무차원 | `std(pitch 시계열)` | 고개 상하 회전 변화량 |
| `head_roll` | 도(°) | `std(roll 시계열)` | 고개 기울기 변화량. 도 단위 |
| `head_yaw_mean` | 무차원 | `mean(yaw 시계열)` | clip 평균 고개 방향 (0 = 정면) |
| `head_pitch_mean` | 무차원 | `mean(pitch 시계열)` | clip 평균 고개 상하 기울기 |
| `head_roll_mean` | 도(°) | `mean(roll 시계열)` | clip 평균 고개 좌우 기울기 |

**head pose 추정 공식 (기하학적 근사)**:

```
yaw   = (nose_x - face_center_x) / half_face_width
        → 코가 얼굴 중심 대비 얼마나 치우쳤는가

pitch = (nose_y - eye_mouth_midpoint_y) / half_face_height
        → 코가 눈-입 중간점 대비 얼마나 위/아래인가

roll  = atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
        → 눈 라인 기울기 (degrees)
```

#### 얼굴 이동 / 안정성

| 컬럼 | 단위 | 계산 공식 | 해석 |
|---|---|---|---|
| `face_movement` | 정규화 거리 | `Σ |face_center[i] - face_center[i-1]|` | clip 내 얼굴 중심의 누적 이동량. real은 크고, spoof(정지 사진/재생)는 작음 |
| `face_stability` | 정규화 거리 | `√(σx² + σy²)` — 얼굴 중심 표준편차 | 얼굴이 흔들리는 정도. `face_movement`의 집계 보완 지표 |
| `nose_movement` | 정규화 거리 | `Σ |nose[i] - nose[i-1]|` | 코 끝점의 누적 이동량. `face_movement`와 상관 높음 |

**얼굴 중심 정의**: 눈 4개 코너(33, 133, 362, 263)의 x,y 평균

---

## 6. 전처리 실행 결과

### 6-1. 이미지 클립 피처 (`face_clip_features.jsonl`)

| 항목 | 값 |
|---|---|
| 파일 경로 | `features/face_clip_features.jsonl` |
| 총 clip | 966 |
| 성공 | **966 (100%)** |
| 실패 | 0 |

**label 분포**

| label | clip 수 | 비율 |
|---|---|---|
| 0 (real) | 548 | 56.7% |
| 1 (spoof) | 418 | 43.3% |

**attack_type × split 분포**

| attack_type | train | valid | test | 합계 |
|---|---|---|---|---|
| live | 371 | 84 | 93 | 548 |
| replay | 155 | 60 | 62 | 277 |
| print | 88 | 25 | 28 | 141 |
| **합계** | **614** | **169** | **183** | **966** |

**face_detect_rate 분포**

| 임계값 | clip 수 | 비율 |
|---|---|---|
| ≥ 1.00 | 923 | 95.5% |
| ≥ 0.90 | 947 | 98.0% |
| ≥ 0.80 | 964 | 99.8% |
| ≥ 0.50 | 965 | 99.9% |
| < 0.50 | 1 | 0.1% |

- 최저 `face_detect_rate`: 0.4667 (`ATK08_replay_clip003`) — 얼굴이 작게 촬영된 gopro 클립

**frame_count 분포**: 평균 16.6 / 중앙값 13 / 최소 1 / 최대 39

### 6-2. 영상 피처 (`face_video_features.csv`)

| 항목 | 값 |
|---|---|
| 파일 경로 | `features/face_video_features.csv` |
| 총 영상 | 56 |
| 성공 | **56 (100%)** |
| 실패 | 0 |

| label | 영상 수 | 비율 |
|---|---|---|
| 0 (real) | 29 | 51.8% |
| 1 (spoof) | 27 | 48.2% |

---

## 7. 피처 판별력 분석

### 7-1. Cohen's d (효과 크기)

Cohen's d 기준: **d < 0.2** 소효과 / **0.2 ≤ d < 0.5** 중효과 / **d ≥ 0.5** 대효과 (real vs spoof 구분력)

#### 이미지 클립 피처 (966 clips)

| 피처 | real 평균 | spoof 평균 | Cohen's d | 효과 |
|---|---|---|---|---|
| `nose_movement` | 1.0086 | 0.6253 | **0.760** | 대 |
| `face_movement` | 0.7002 | 0.4872 | **0.681** | 대 |
| `head_pitch` | 0.1443 | 0.1083 | 0.410 | 중 |
| `blink_count` | 1.1259 | 0.6938 | 0.382 | 중 |
| `eye_open_ratio` | 0.3011 | 0.3370 | 0.383 | 중 |
| `head_yaw` | 0.2894 | 0.2081 | 0.353 | 중 |
| `head_roll` | 1.3684 | 1.8476 | 0.300 | 중 |
| `mouth_open_ratio` | 0.0311 | 0.0489 | 0.272 | 중 |
| `smile_ratio` | 0.0193 | 0.0157 | 0.270 | 중 |
| `ear_std` | 0.0679 | 0.0610 | 0.134 | 소 |
| `head_pitch_mean` | -0.0212 | -0.0009 | 0.165 | 소 |
| `face_stability` | 0.0474 | 0.0490 | 0.098 | 소 |
| `head_yaw_mean` | 0.0256 | 0.0289 | 0.032 | 소 |
| `mar_std` | 0.0387 | 0.0369 | 0.036 | 소 |
| `head_roll_mean` | -0.1914 | -0.1984 | 0.004 | 소 |

#### 영상 피처 (56 videos)

| 피처 | real 평균 | spoof 평균 | Cohen's d | 효과 |
|---|---|---|---|---|
| `head_yaw` | 0.7600 | 0.2228 | **2.099** | 대 |
| `smile_ratio` | 0.0286 | 0.0143 | **1.529** | 대 |
| `nose_movement` | 2.1800 | 1.2341 | **1.394** | 대 |
| `head_pitch` | 0.4143 | 0.1396 | **1.461** | 대 |
| `ear_std` | 0.0660 | 0.0228 | **1.240** | 대 |
| `face_stability` | 0.1135 | 0.0784 | **1.108** | 대 |
| `face_movement` | 1.4371 | 1.0100 | **0.957** | 대 |
| `head_roll` | 2.9081 | 1.5127 | **0.935** | 대 |
| `head_pitch_mean` | -0.1868 | -0.0711 | **0.993** | 대 |
| `blink_count` | 3.1379 | 2.1111 | 0.464 | 중 |
| `eye_open_ratio` | 0.1971 | 0.1800 | 0.530 | 대 |

> **관찰**: 영상 피처의 Cohen's d가 이미지 클립보다 전반적으로 높습니다.  
> 이미지 클립은 7~39프레임 수준의 짧은 시계열이지만, 영상은 수십~수백 프레임을 포함해  
> 시간적 신호(깜빡임, 고개 움직임)를 더 안정적으로 집계하기 때문입니다.

### 7-2. 핵심 판별 피처 요약

```
1위 face_movement / nose_movement
     → real은 자연스러운 머리 움직임으로 누적 이동량이 spoof 대비 40~60% 높음

2위 head_yaw / head_pitch / head_roll
     → real은 대화·시선 이동으로 고개 방향이 다양함
     → spoof(print/replay)는 카메라 앞에 고정된 사진·화면이라 변화 적음

3위 blink_count / ear_std
     → real은 자연스러운 눈 깜빡임이 있어 blink_count 높고 EAR 변동성 큼
     → print 공격은 사진이라 EAR 변화 없음, replay는 영상이지만 깜빡임 적음
```

---

## 8. subject 분포 요약

| 구분 | 내용 |
|---|---|
| S-namespace (CASIA-style) | 57명 (live_XXX), 이 중 12명은 spoof_XXX와 쌍 |
| R-namespace (디바이스 캡처) | 9명 (real_01_* ~ real_09_*) |
| Anonymous (attack_XX_*) | subject_id = null, 188개 clip |

---

## 9. 생성 파일 목록

| 파일 | 행 수 | 컬럼 수 | 설명 |
|---|---|---|---|
| `dataset/samples_manifest.jsonl` | 966 | 12 | clip 단위 메타데이터 + split 할당 |
| `features/face_clip_features.csv` | 966 | 27 | 이미지 클립 피처 (CSV) |
| `features/face_clip_features.jsonl` | 966 | 27 | 이미지 클립 피처 (JSONL, 최종 결과물) |
| `features/face_video_features.csv` | 56 | 20 | 영상 피처 (CSV) |
| `features/skipped_clips.csv` | 0 | — | 실패 클립 없음 |

---

## 10. 주의사항 및 다음 단계

### 알려진 이슈

1. **기존 `face_image_features.csv` 사용 불가**  
   구버전 스크립트가 subject 서브폴더 없이 flat 경로로 추출했습니다.  
   현재 폴더 구조(`real/live_001/live_001-*.jpg`)와 달라 참조 불가.  
   본 보고서의 `face_clip_features.jsonl`로 대체합니다.

2. **session / illumination 컬럼이 float으로 저장됨**  
   CSV 저장 시 `null` 혼재로 int → float 변환이 발생했습니다.  
   (`session: 1.0`) 사용 시 `int(session)` 캐스팅 필요.

3. **frame_count 편차 큼 (1~39프레임)**  
   일부 클립은 1프레임만 있습니다 (attack_XX 폴더 마지막 청크).  
   학습 전 `frame_count >= 3` 필터링을 권장합니다.

4. **`ATK08_replay_clip003` face_detect_rate 0.47**  
   gopro 광각 촬영으로 얼굴이 작아 검출 실패 프레임이 많습니다.  
   학습 전 `face_detect_rate < 0.5` 필터링 검토를 권장합니다.

### 권장 다음 단계

```
1. [ML 학습] face_clip_features.jsonl → 피처 기반 LightGBM / SVM 학습
   - 입력: 피처 16개
   - 출력: label (0/1)
   - 권장 피처: face_movement, nose_movement, head_yaw, blink_count, head_pitch

2. [CNN 학습] samples_manifest.jsonl → MobileNetV2 fine-tune
   - scripts/train_antispoofing.py 바로 실행 가능

3. [앙상블] 피처 ML + CNN 스코어 평균 또는 스태킹

4. [추가 수집] replay 클립 추가 필요
   - 현재 print(141) < replay(277) < live(548)로 print 클립 부족
```
