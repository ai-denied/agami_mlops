# context_emotion 전처리 보고서 (v1)

**대상 산출물**: `context_emotion_train_dataset_v1.csv` (6,721건)
**작업 위치**: `ml-pipeline/context_emotion/` (스크립트), `/workspace/data/context_emotion/processed/` (산출물)
**관련 문서**: 라벨 스키마 변경 경위 전체 기록은 `RECONSTRUCTION_NOTES.md`, 컬럼별 의미는 `FEATURE_DESCRIPTION_v1.md` 참고.

---

## 1. 배경 — pod 리셋으로 인한 데이터 소실과 재구성

`ml-pipeline/context_emotion/`는 작업 시작 시점에 완전히 비어 있었다. pod가
죽으면서 review_app.py, engagement_review_app.py,
restore_original_emotic_labels.py, captcha_bank_human_reviewed.csv,
manual_images_labeled.csv, excluded_pool.csv, qwen_attack_results.csv,
qwen_recheck_queue.csv 등 4일치 작업물이 전부 사라졌다. 복구된
워크스페이스 스냅샷(`recovered_workspace_*`)도 확인했지만
facial_recognition/flashlight 코드만 들어있고 context_emotion 관련 파일은
없었다.

남아있던 것은 ① 원본 데이터(`emotic_dataset/Annotations/Annotations.mat`,
`manual_images/<폴더>/`)와 ② 팀이 작성한 Day4/Day5 회고록 텍스트(자연어
기록)뿐이었다. 이 보고서가 다루는 모든 라벨은 그 회고록을 스펙으로 삼아
**다시 만든 것**이며, 사람이 한 건씩 검수했던 개별 판단(어떤 Engagement
895건 중 50건을 어떤 라벨로 재분류했는지, Qwen 공격 결과로 어떤 manual
30건을 재투입했는지 등)은 원천적으로 복원이 불가능했다.

## 2. 1단계 — EMOTIC/manual 라벨 복원

- **emotic**: `Annotations.mat`의 train(17,077장)/val(2,088장)/test(4,389장)
  전체를 파싱해, person(=얼굴/인물 bbox) 단위로 annotator들의 카테고리
  투표를 모아 **과반(≥50%) 동의** 카테고리만 채택했다 (`restore_emotic_labels.py`).
  train split은 annotator가 1명이라 그대로 채택된다.
- **manual**: `manual_images/`의 28개 폴더(aggression, danger, joy, ... 등
  248장)를 폴더명 → (emotion, situation) 매핑표로 라벨링했다
  (`build_manual_labels.py` + `emotion_mapping.MANUAL_FOLDER_TO_LABELS`).
  이 매핑표는 회고록에 적힌 "manual 감정/상황 최종 분포" 숫자를 역산해서
  맞춘 것인데, 재실행 결과가 그 분포와 거의 정확히 일치해 (예: safety 55,
  danger 43, happiness 27 등 그대로 재현) 매핑이 합리적이라는 근거로
  삼았다.
- 두 출력 모두 원본 라벨(`raw_categories_majority` / 폴더명)을 그대로
  같이 저장해, 매핑표가 바뀌어도 다시 가공할 수 있게 했다.

## 3. 라벨 스키마 확정 (감정 14종 + 상황 7종)

작업 중 스키마가 두 번 바뀌었다 (상세 경위는 `RECONSTRUCTION_NOTES.md`):

1. 처음엔 회고록 그대로 감정 14종/상황 7종(everyday 제외)으로 시작.
2. manual 전용 `empathy`를 15번째 감정으로 추가했다가,
3. v1 학습셋을 만들면서 사용자가 **최종적으로 감정 14종(empathy 제외) +
   상황 8종(everyday 포함)**으로 다시 고정했다가, v1 빌드 결과 everyday가
   실제 데이터 0건인 빈 슬롯이라는 게 확인되자 **다시 상황 7종(everyday
   제외)으로 되돌렸다**. 이게 최종 스키마다.

이 과정에서 라벨 매핑 자체도 두 차례 교정했다 — Esteem/Sympathy를
calm이 아닌 affection으로, Sensitivity를 confusion이 아닌 aversion으로,
manual의 despair를 (미해결 보류 대신) sadness로 확정했다. empathy 4건은
스키마 외 라벨로 드롭되어 `manual_images_unresolved.csv`에 보관 중이다.

**최종 감정 14종**: happiness, calm, anticipation, affection, anger, fear,
sadness, disconnection, suffering, aversion, embarrassment, confidence,
confusion, yearning

**최종 상황 7종**: conflict, danger, loss_absence, pressure, safety,
teasing, vanity

## 4. 2단계 — v1 학습 데이터셋 빌드 파이프라인 (`build_train_dataset_v1.py`)

입력은 `captcha_bank_human_reviewed.csv`(emotic, 26,466건)와
`manual_images_labeled.csv`(manual, 224건)만 사용했다.
`excluded_pool.csv`(Engagement/Surprise만 검출된 emotic 7,854건)는 학습
후보에서 처음부터 배제했다.

### 4-1. 이미지 경로 해석 및 검증 (`image_paths.py`)
디스크의 EMOTIC 원본은 같은 서브셋(mscoco/framesdb 등)이 최대 3곳에
중복 미러링되어 있고, annotator가 라벨링한 이미지 중 상당수가 실제로는
다운로드돼 있지 않다. 폴더별로 우선순위가 매겨진 후보 경로를 순서대로
시도하며 `PIL.Image.open().load()`로 실제 디코딩까지 성공하는 첫 경로를
채택했다 (단순히 파일 존재 여부만 보면 Day4 회고록에 적혀있던 "잘린
파일" 문제를 못 잡는다).

### 4-2. 라벨 정규화 (`normalize_label.py`)
대소문자, 구분자(공백/슬래시/언더스코어) 차이와 알려진 오타
(`doubt_confusion`, `doubt_confusning` 등)를 흡수해 14종/7종 캐노니컬
이름으로 정규화한다. `Engagement`/`Surprise`/`test` 같은 값은 오타가
아니라 명시적으로 스키마 밖으로 처리한다. (현재 입력 CSV는 이미
정제되어 있어 실제로 교정이 발생한 사례는 없었고, 향후 외부 데이터가
들어올 때를 위한 방어 코드다.)

### 4-3. 필터링
다음 중 하나라도 해당하면 학습 후보에서 빠지고, 사유와 함께
`context_emotion_excluded_v1.csv`에 남긴다.
- 이미지 파일을 찾을 수 없거나 디코딩 실패 → `image_not_found`
- emotion/situation 값이 14종/7종에 없음 → `*_label_not_in_schema`
- emotion·situation이 둘 다 비어있음 → `empty_label`

26,690건 후보 중 **22,162건 통과 / 4,528건 제외**(전부
`image_not_found` — emotic 원본 이미지가 디스크에 부분적으로만
존재하기 때문).

### 4-4. label_confidence 산정
- emotic: `Annotations.mat`을 다시 파싱해, 그 행이 채택한 원본 카테고리들의
  (동의 annotator 수 / 전체 annotator 수) 평균. train(annotator 1명)은
  항상 1.0, val/test는 실측 0.5~1.0.
- manual: 폴더명 기반 휴리스틱이라 사람/모델 검증이 없어 고정값 0.6.

### 4-5. 균형 샘플링
감정 클래스별 최대 800장 캡, 300장 미만이면 `low_resource`로 표시하고
가능한 만큼 포함. **manual 행은 source 우선권을 줘서 항상 전부 유지한
뒤, 같은 클래스의 emotic 행으로 800장 한도를 채운다** — 처음 구현에서는
manual과 emotic을 한 풀에서 무작위로 섞어 800장을 뽑았는데, manual은
양이 워낙 적어서(클래스당 최대 27건) 우연히 캡 밖으로 밀려나는 버그가
있었다(28건 소실, 사유 기록도 없이 사라짐). 이를 발견하고 수정해
manual 224건이 전부 보존되도록, 그리고 사라지는 행 없이 전부
`exclude_reason`과 함께 기록되도록 고쳤다.
이후 이미지 경로 기준 중복 제거(같은 사진이 두 번 들어가지 않게).

샘플링 단계에서 추가로 빠진 15,441건(`class_quota_exceeded` 14,410 +
`duplicate_image` 1,031)도 전부 `context_emotion_excluded_v1.csv`에
사유와 함께 보관된다.

### 4-6. train/val/test 분할
emotion_label 기준 stratified 70/15/15(시드 13), 이미지 단위로 먼저
중복 제거를 마친 뒤 분할하므로 같은 이미지가 서로 다른 split에 들어가는
경우는 없다.

## 5. 최종 결과

| 항목 | 값 |
|---|---|
| 최종 학습셋 (`context_emotion_train_dataset_v1.csv`) | **6,721건** |
| source별 | emotic 6,497 / manual 224 |
| split별 | train 4,704 / val 1,009 / test 1,008 |
| 감정 14종 중 최다 | happiness 772 |
| 감정 14종 중 최소(low_resource) | embarrassment 59, sadness 153, aversion 222 |
| 상황 7종 합계 | 128건 (전부 manual) |
| 제외 총합 (`context_emotion_excluded_v1.csv`) | 19,969건 (image_not_found 4,528 / class_quota_exceeded 14,410 / duplicate_image 1,031) |

감정×상황 동시 라벨은 `yearning x loss_absence` 3건뿐이다(manual의
"missing" 폴더, 그리움/부재 컨셉이라 두 축에 동시 매핑됨).

## 6. 알아두어야 할 한계 (정확도 관련 주의사항)

1. **사람 검수가 아니라 근사 재구성이다.** `review_status=reconstructed_approx`인
   행은 전부 회고록 텍스트 규칙을 거꾸로 적용해 만든 것이고, 원래
   존재했던 한 건 한 건의 사람 판단이 아니다.
2. **emotic 규모가 원래 캡차 뱅크(1,737건)보다 훨씬 크다.** 원래는
   `attack_candidates.csv`로 미리 후보를 선별했었는데 그 파일이 소실돼,
   이번엔 "선별 전 EMOTIC 전체"가 출발점이다. 캡차 난이도/품질 기준의
   추가 선별이 필요하면 별도 작업이다.
3. **emotion_label은 단일 라벨이다.** emotic 원본은 멀티라벨인데, 첫 번째
   매핑값만 대표로 썼다 (`FEATURE_DESCRIPTION_v1.md` 참고). 멀티라벨
   학습을 원하면 `original_labels`에서 다시 파싱해야 한다.
4. **situation 라벨은 거의 전부 manual 출처다.** emotic 쪽엔 situation
   축이 없어서, 상황 분류기를 학습하려면 manual 128건만으로는 데이터가
   매우 부족하다 (특히 pressure 2건, vanity 1건).
5. **(해소됨) everyday는 데이터가 0건이라 스키마에서 다시 제거했다.**
   필요해지면 manual_images에 everyday 폴더를 채워서 다시 추가하면 된다.
6. **manual의 label_confidence(0.6)는 emotic의 실측 신뢰도와 같은 척도가
   아니다.** 단순 비교/가중치로 섞어 쓰면 안 된다.

## 7. 산출물 위치

| 파일 | 위치 | 용도 |
|---|---|---|
| `context_emotion_train_dataset_v1.csv` | `/workspace/data/context_emotion/processed/` | **학습에 바로 사용** |
| `context_emotion_excluded_v1.csv` | 〃 | 제외된 19,969건 + 사유 (감사/검증용) |
| `context_emotion_label_distribution_v1.md` | 〃 | 자동 생성된 raw 분포 통계 |
| `context_emotion_label_mapping_v1.json` | 〃 | 라벨 매핑 규칙 전체(머신 readable) |
| `context_emotion_dataset_build_report_v1.md` | 〃 | 빌드 스크립트 자동 생성 로그 |
| `export_v1.zip` | `/workspace/data/context_emotion/` | 구글 드라이브 업로드용 (이미지+csv만 따로 묶음) |
| `RECONSTRUCTION_NOTES.md`, `FEATURE_DESCRIPTION_v1.md`, 이 문서 | `ml-pipeline/context_emotion/` | 사람이 읽는 설명 문서 |
| `emotion_mapping.py`, `restore_emotic_labels.py`, `build_manual_labels.py`, `build_train_dataset_v1.py`, `image_paths.py`, `normalize_label.py`, `export_train_dataset_v1.py` | `ml-pipeline/context_emotion/` | 전체 파이프라인 재실행 가능한 스크립트 |
