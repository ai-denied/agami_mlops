# context_emotion_train_dataset_v1 — 컬럼(feature) 설명

대상 파일: `context_emotion_train_dataset_v1.csv` (6,721행),
`context_emotion_excluded_v1.csv`(19,969행, `dataset_split` 컬럼만 없음)도 동일 스키마.

| 컬럼 | 타입 | 값 범위 / 예시 | 설명 |
|---|---|---|---|
| `sample_id` | string | `manual-4be531aa9e`, `emotic-7a1c... ` | `{source}-{md5(원본 키)[:10]}` 형태의 고유 ID. 원본 키는 emotic이면 `split│folder│filename│person_index`, manual이면 `folder│filename`. 같은 입력이면 재실행해도 항상 같은 id가 나오는 결정적(deterministic) 해시라서, 파이프라인을 다시 돌려도 행을 추적할 수 있다. |
| `image_path` | string (파일 경로) | `/workspace/data/.../elation/context_056.jpg` | 실제로 열어서 디코딩까지 성공한 이미지 파일의 절대경로. `export_train_dataset_v1.py`로 업로드용 폴더를 만들면 `images/{sample_id}.확장자` 상대경로로 바뀐다. |
| `source` | string (범주형) | `emotic` \| `manual` | 라벨의 출처. `emotic`은 EMOTIC 공개 데이터셋(Annotations.mat) 기반, `manual`은 팀이 직접 생성/수집한 이미지(`manual_images/<폴더명>/`) 기반. |
| `original_labels` | string | `Disconnection;Doubt/Confusion` (emotic), `elation` (manual) | 최종 라벨로 매핑되기 전의 원본 라벨. emotic은 annotator 과반 동의를 받은 EMOTIC 원본 카테고리(세미콜론 구분, 멀티라벨 가능), manual은 원본 폴더명 그대로. **항상 보존** — 매핑 규칙이 바뀌어도 이 컬럼만 있으면 재가공할 수 있다. |
| `emotion_label` | string (범주형, 단일값) | 14종 중 하나 또는 빈 문자열 | happiness, calm, anticipation, affection, anger, fear, sadness, disconnection, suffering, aversion, embarrassment, confidence, confusion, yearning. emotic 쪽이 원래 멀티라벨인 경우 첫 번째 매핑값을 대표 라벨로 채택(단일 라벨 분류기 학습을 가정). situation만 있는 manual 행(safety/danger 등)은 빈 문자열. |
| `situation_label` | string (범주형, 단일값) | 8종 중 하나 또는 빈 문자열 | conflict, danger, everyday, loss_absence, pressure, safety, teasing, vanity. emotic 데이터에는 situation 축 자체가 없어 항상 빈 문자열이고, manual 중 situation 폴더(safety/danger/concert→conflict/teasing/pressure/superiority→pressure/vanity/missing→loss_absence)에서만 채워진다. `everyday`는 스키마에는 있지만 현재 데이터에 매핑되는 행이 0건이다. |
| `review_status` | string (범주형) | 현재는 전부 `reconstructed_approx` | 라벨이 어떤 과정으로 만들어졌는지. `reconstructed_approx`는 "pod 리셋으로 소실된 사람 검수 기록을 문서화된 규칙으로 근사 재구성했다"는 뜻이며, 사람이 한 건씩 본 원래의 검수 상태가 아니다 (자세한 내용은 `RECONSTRUCTION_NOTES.md`). |
| `label_confidence` | float (0.0~1.0) | 0.5 ~ 1.0 (실측) | 라벨을 얼마나 신뢰할 수 있는지의 수치. **emotic**: `Annotations.mat`을 다시 파싱해 해당 행이 채택한 원본 카테고리들에 대해 (동의한 annotator 수 / 전체 annotator 수)의 평균. train split은 annotator가 1명이라 항상 1.0, val/test는 보통 annotator 5명 기준 다수결이라 0.5~1.0 사이로 갈린다. **manual**: 폴더명 기반 휴리스틱 매핑이라 사람/모델 검증이 없어 고정값 0.6 — emotic의 실측 신뢰도와 같은 척도로 직접 비교하면 안 된다. |
| `exclude_reason` | string | `context_emotion_train_dataset_v1.csv`에서는 항상 빈 문자열 | 이 행이 왜 학습셋에서 빠졌는지. `image_not_found`(파일이 디스크에 없거나 손상돼 디코딩 실패), `emotion_label_not_in_schema` / `situation_label_not_in_schema`(14종/8종 스키마 밖 라벨), `empty_label`(emotion·situation 둘 다 없음), `class_quota_exceeded`(같은 emotion 클래스에서 800장 캡을 넘겨 샘플링 단계에서 제외), `duplicate_image`(동일 이미지 파일이 이미 다른 행으로 포함됨). `context_emotion_excluded_v1.csv`에만 값이 채워진다. |
| `dataset_split` | string (범주형) | `train` \| `val` \| `test` | 모델 학습용 분할. emotion_label 기준 stratified 70/15/15(시드 13). 이미지 단위로 먼저 중복 제거를 한 뒤 분할해서, 같은 이미지가 두 split에 동시에 들어가는 경우는 없다. `context_emotion_excluded_v1.csv`에는 이 컬럼이 없다(분할 자체를 거치지 않고 빠진 행들이라서). |

## 참고: emotion_label이 "단일 라벨"인 이유

EMOTIC 원본은 사람이 여러 카테고리를 동시에 체크할 수 있는 멀티라벨 데이터다
(`original_labels`에 세미콜론으로 여러 개가 남아있는 게 그 증거). 하지만
`emotion_label` 컬럼은 그중 **첫 번째로 매핑된 값 하나만** 대표 라벨로 쓴다.
멀티라벨 분류기를 학습할 계획이면 `original_labels`를 다시 파싱해서
`emotion_mapping.EMOTIC_CATEGORY_TO_EMOTION`으로 재매핑하면 멀티라벨 버전을
복원할 수 있다.
