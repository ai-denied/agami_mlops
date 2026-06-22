# Context Emotion CAPTCHA 벤치마크 회고록 (Day 6)

**일자**: 2026-06-22
**담당**: agami MLOps
**선행 문서**: [RETROSPECTIVE_20260617_context_emotion.md](../2026-06-17/RETROSPECTIVE_20260617_context_emotion.md) (Day 2, 자동검수 신뢰도 스코어링까지)

> Day 3~5(2026-06-18, 06-19, 06-21)의 작업 — 사람 검수 워크플로우 전환,
> EMOTIC 원본 라벨 복원, 14클래스 최종 확정, Qwen 재검증 큐 생성 등 —도
> 실제로 있었지만, **그 사이 pod가 리셋되면서 retrospectives 파일 자체가
> 기록되지 못한 채 소실됐다.** 다행히 작업자가 별도로 텍스트로 남겨뒀던
> Day4/5 회고록 본문이 있어 이번 작업의 스펙으로 그대로 활용했다. 아래는
> 그 텍스트를 근거로 **`ml-pipeline/context_emotion/`을 통째로 재구축한**
> 오늘 하루 기록이다.

---

## 오늘의 핵심 — 빈 디렉터리에서 시작해서 v2 학습 데이터셋까지

작업 시작 시점에 `ml-pipeline/context_emotion/`는 완전히 비어 있었다.
review_app.py, restore_original_emotic_labels.py,
captcha_bank_human_reviewed.csv, qwen_attack_results.csv 등 4일치 산출물이
pod 리셋으로 전부 사라진 상태였다. 복구된 워크스페이스 스냅샷
(`recovered_workspace_*`) 4개를 다 확인했지만 facial_recognition/flashlight
코드만 들어있었고 context_emotion 관련 파일은 하나도 없었다.

남은 건 ① 원본 데이터(`Annotations.mat`, `manual_images/` 228장)와
② Day4/5 회고록 텍스트뿐이었다. 사람이 한 건씩 본 개별 판단(Engagement
895건 중 어떤 50건을 재분류했는지, Qwen 결과로 manual 30건을 재투입한
근거 등)은 원천적으로 복원 불가능하다고 판단해 **근사 재구축**으로
방향을 잡았다.

## 1. 라벨 복원 (`restore_emotic_labels.py`, `build_manual_labels.py`)

- emotic: `Annotations.mat` train/val/test 전체(23,554장)를 person 단위로
  파싱해 annotator 과반(≥50%) 동의 카테고리만 채택.
- manual: `manual_images/`의 28개 폴더를 폴더명 → (emotion, situation)
  매핑표로 라벨링. 이 매핑표는 Day5 회고록에 적힌 "manual 최종 분포" 숫자를
  거꾸로 맞춰서 역산한 것인데, 재실행 결과가 거의 정확히 일치해서(감정
  99건, 상황 128건, 폴더별 건수 대부분 그대로 재현) 매핑이 합리적이라는
  근거로 삼았다.

## 2. 라벨 스키마가 하루 동안 세 번 바뀜

`RECONSTRUCTION_NOTES.md`에 전체 경위를 시간순으로 남겼다. 요약하면:

1. 회고록 그대로 감정 14종/상황 7종으로 시작
2. manual 전용 `empathy`를 15번째 감정으로 추가
3. 사용자가 EMOTIC 원본 통합표를 제공 — Esteem/Sympathy→affection,
   Sensitivity→aversion, despair→sadness로 매핑 교정 (이전 추정이 틀렸던
   부분)
4. v1 학습셋을 만들면서 **감정 14종(empathy 제외) + 상황 8종(everyday
   포함)**으로 다시 고정
5. v1 빌드 결과 everyday가 매핑되는 행이 0건인 빈 슬롯이라는 게 드러나자
   **상황을 다시 7종(everyday 제외)으로** 되돌림

최종: **감정 14종**(happiness/calm/anticipation/affection/anger/fear/
sadness/disconnection/suffering/aversion/embarrassment/confidence/
confusion/yearning) + **상황 7종**(conflict/danger/loss_absence/pressure/
safety/teasing/vanity).

## 3. v1 학습 데이터셋 빌드 (`build_train_dataset_v1.py`)

- 디스크의 EMOTIC 원본이 같은 서브셋(mscoco/framesdb 등)을 최대 3곳에
  중복 미러링하고 있고, 라벨링된 이미지 중 다수가 실제로 다운로드돼
  있지 않다는 걸 확인 (`image_paths.py`로 후보 경로 우선순위 + 실제
  디코딩 검증 처리).
- `label_confidence`는 emotic의 경우 annotator 동의 비율을 다시 계산해서
  실측값으로, manual은 폴더명 휴리스틱이라 고정값 0.6으로 분리.
- **버그 발견 및 수정**: 균형 샘플링(클래스당 최대 800장)에서 manual과
  emotic을 한 풀에서 무작위로 섞었더니, manual은 클래스당 최대 27건뿐이라
  우연히 800장 캡 밖으로 밀려나 **28건이 사유 기록도 없이 사라지는** 문제를
  발견. manual을 항상 먼저 전부 보존하고 남은 quota만 emotic으로 채우도록
  고치고, 샘플링 단계에서 제외된 모든 행도 `context_emotion_excluded_v1.csv`에
  사유와 함께 남도록 했다.
- 최종: **6,721건** (emotic 6,497 + manual 224), train/val/test =
  4,704/1,009/1,008.

## 4. 업로드/문서화

- `export_train_dataset_v1.py` — v1이 참조하는 이미지만 골라
  `sample_id` 기준으로 이름 붙여 복사 + 상대경로 CSV 생성 → 1.3GB
  `export_v1.zip`으로 묶어 구글 드라이브 업로드용으로 준비.
- `RECONSTRUCTION_NOTES.md` (스키마 변경 전체 경위), `FEATURE_DESCRIPTION_v1.md`
  (컬럼별 설명), `PREPROCESSING_REPORT_v1.md` (전체 파이프라인 보고서) 작성.
- `/workspace/data/context_emotion/processed/`는 학습에 직접 필요 없는
  중간 재구조 파일(captcha_bank_human_reviewed.csv 등)과 감사용 보고서를
  정리해 최종 학습 csv + 문서 사본만 남김 — 중간 파일은 스크립트로
  언제든 재생성 가능.
- 코드/문서 전부 git에 커밋·푸시 (`ai-denied/agami_mlops` main). 데이터/zip은
  용량 때문에 레포 밖 `/workspace/data/`에만 둠.

## 5. v2 — 검수자/AI평가/서비스 단계까지 고려한 확장 스키마

작업 도중 "지금 상황에 맞는 최종 피처" 스펙이 새로 들어왔다. 핵심은 4단계
분리: ① 전처리 결과(지금 만들 것) ② 감정 선택지(MCQ, 추후) ③ AI 평가
결과(추후) ④ 서비스 운영 결과(추후). 1단계만 지금 만들기로 하고
`build_train_dataset_v2.py`를 새로 작성:

- `source_image_id`, `image_width/height`, `target_person_bbox` 추가
  (emotic은 실제 bbox, manual은 사람 탐지를 한 적이 없어 빈 값으로 둠)
- `emotion_label` 하나였던 걸 `candidate_emotions`(멀티라벨 전체) +
  `provisional_emotion`(대표 1개)로 분리
- `content_hash`(정확 중복) + `perceptual_hash`(근접 중복, `imagehash`
  평균해시) 추가
- `reviewer_answer`/`reviewer_confidence`/`reviewer_note`/`reviewed_at`
  컬럼은 전부 빈 값, `review_status`는 전부 `unreviewed`로 — 실제 사람
  검수자가 아직 아무것도 안 봤다는 사실을 숨기지 않음

**여기서 v1의 진짜 설계 결함을 하나 더 발견했다**: v1은 `image_path`
하나로만 중복을 판정했는데, 같은 사진 속 다른 인물(다른 bbox, 다른 라벨)도
"중복"으로 오인해 지워버리고 있었다. v2는 `target_person_bbox`로 인물을
구분하고, 대신 `split_group_id`(source_image_id/content_hash/
perceptual_hash 중 하나라도 겹치면 같은 그룹)로 묶어서 "같은 사진 또는
근접 중복은 같은 split에만 들어가게" 하는 방식으로 바꿨다. 그 결과
v2가 v1보다 약 1,000건 더 많다 — **7,752건**(emotic 7,528 + manual 224),
`split_group_id` 고유 6,708개, train/val/test = 5,412/1,159/1,181.

## 다음 단계 (우선순위 후보)

1. 새 스펙의 2~4단계(MCQ 선택지, AI 공격모델 평가, 실제 서비스 지표)는
   아직 착수 전 — "이후" 단계로 명시적으로 미뤄둔 상태.
2. situation 라벨이 거의 전부 manual 224건에서만 나와서 다양성이 낮다
   (pressure 2건, vanity 1건). 상황 분류기를 학습하려면 데이터 보강이
   필요.
3. `perceptual_hash`는 평균해시 정확 일치로만 그룹핑했고, 해밍 거리
   임계값 기반 근접매칭은 아직 구현 안 함 — 압축률이 크게 다른
   재인코딩본은 못 잡을 수 있음.
4. emotic 풀 규모(필터 후 22,162건)가 원래 캡차 뱅크(1,737건)보다 훨씬
   크다 — `attack_candidates.csv` 기반 후보 선별 단계가 소실되어 재현
   불가능했기 때문. 캡차 난이도/품질 기준의 추가 선별은 별도 작업.
5. 전처리 담당자에게 v2 CSV를 넘겨 `reviewer_answer` 등을 실제로
   채워야 review_status가 `single_reviewer_verified`로 바뀐다.

---

## 산출물 정리

| 파일 | 위치 | 상태 |
|---|---|---|
| `restore_emotic_labels.py`, `build_manual_labels.py`, `emotion_mapping.py` | `ml-pipeline/context_emotion/` | 라벨 복원/매핑, git에 있음 |
| `build_train_dataset_v1.py`, `build_train_dataset_v2.py` | 〃 | 학습셋 빌드, git에 있음 |
| `image_paths.py`, `normalize_label.py`, `export_train_dataset_v1.py` | 〃 | 보조 유틸, git에 있음 |
| `RECONSTRUCTION_NOTES.md`, `FEATURE_DESCRIPTION_v1.md`, `PREPROCESSING_REPORT_v1.md` | 〃 | 문서, git에 있음 |
| `context_emotion_train_dataset_v1.csv` (6,721건) | `/workspace/data/context_emotion/processed/` | 학습 가능, git 밖 |
| `context_emotion_train_dataset_v2.csv` (7,752건) | 〃 | 검수자 핸드오프용, git 밖 |
| `export_v1.zip` (1.3GB) | `/workspace/data/context_emotion/` | 구글 드라이브 업로드용, git 밖 |
