# Context Emotion 데이터셋 분석 및 전처리 회고록

**일자**: 2026-06-16  
**담당**: agami MLOps

---

## 목표

VLM 취약점 평가용 벤치마크 데이터셋 구축.  
Qwen이 감정 인식에 실패하는 "어려운 이미지"를 선별·정제해 새 레이블링의 입력으로 준비.

---

## 데이터셋 구조

```
data/context_emotion/
├── emotic_dataset/     # EMOTIC 공개 데이터셋 (레이블 포함)
│   ├── Annotations/    # Annotations.mat (6MB) — CVPR 2017
│   └── emotic/
│       ├── ade20k/        432장
│       ├── emodb_small/  1,386장
│       ├── framesdb/     2,864장
│       └── mscoco/       3,489장  → 합계 8,171장
│
├── emotic/             # 레이블 없는 추가 이미지
│   ├── framesdb/        4,010장
│   └── mscoco/         14,341장  → 합계 18,351장
│
└── manual_images/      # 수동 수집 이미지 (28개 감정 클래스, 239장)
    safety(55), danger(43), nervous(24), concert(16) ...
```

전체: **26,761장** (JPG 26,661 / PNG 89)

---

## 기존 전처리 현황 파악

`ml-pipeline/context_emotion/`에 두 파일 존재:

| 파일 | 내용 |
|------|------|
| `attack_candidates.csv` | emotic_dataset에서 선별된 3,728장 + manual 224장 = **3,952행** |
| `qwen_attack_results.csv` | 위 3,952장에 대한 Qwen 감정 예측 결과 |

### 선별 전략 (역산)

```
emotic_dataset 전체 8,171장
    ↓ Qwen으로 감정 예측
    ↓ 맞춘 것 4,443장 (54.4%) → 버림
    ↓ 틀린 것 3,728장 (45.6%) → 유지
    + manual_images 224장
    = attack_candidates.csv 3,952장
```

이미지당 annotation 1개 (person 중복 없음).

### 컬럼 구성

`split`, `image_path`, `folder`, `filename`, `bbox_x1/y1/x2/y2`, `categories`, `valence`, `arousal`, `dominance`, `gender`, `age`, `source`, `manual_label`

---

## 발견된 문제

| 문제 | 내용 |
|------|------|
| categories 포맷 혼재 | `['Disconnection']` / `Disconnection` / `['A']\|['B']\|['C']` 혼재 |
| 레이블 체계 불일치 | emotic 26클래스 vs manual_images 28클래스 (완전히 다른 체계) |
| VAD 결측 | valence/arousal/dominance 261개 누락 |
| split 비율 이상 | train(1,833) / test(1,264) / val(631) — test가 val보다 2배, 전체 비율 46/32/16 |
| attack_candidates 내 correct=True | Qwen 재실행 시 818개(20.7%)가 정답으로 전환됨 |
| CUDA OOM 59개 | framesdb 이미지에서 예측 자체 실패 |

---

## 전처리 결정 사항

### 1. correct=True 818개 + OOM 59개 제거

Qwen이 맞춘 이미지는 "어려운 이미지" 기준에 부합하지 않으므로 제거.  
CUDA OOM 59개는 정답/오답 불명이므로 함께 제거.

```
3,952개 → correct=False만 유지 → 3,075개
```

| | 전 | 후 |
|--|--|--|
| 전체 | 3,952 | **3,075** |
| emotic | 3,728 | 2,881 |
| manual | 224 | 194 |

split: train 1,639 / test 865 / val 377

### 2. VAD 컬럼 제거

- 229개 결측 존재
- 목적(VLM 취약점 평가)에 VAD가 불필요
- 새 레이블링으로 categories 대체 예정이므로 함께 정리

### 3. categories 포맷 정제 — 보류

새 감정/상황 레이블링을 처음부터 다시 진행하므로 기존 categories 포맷 정제는 불필요.

### 출력 파일

`ml-pipeline/context_emotion/attack_candidates_filtered.csv` — 3,075행 × 18컬럼

---

## Qwen 예측 편향 분석

Qwen이 가장 많이 예측한 레이블:

| 예측 레이블 | 건수 |
|------------|------|
| Happiness | 1,306 |
| Excitement | 772 |
| Fatigue | 385 |
| Disconnection | 320 |

실제 레이블 상위: Disconnection(879), Yearning(349), Doubt/Confusion(216)  
→ **Qwen은 긍정 감정(Happiness/Excitement)을 과도하게 예측하는 경향**  
→ 실제로는 중립/부정 감정(Disconnection, Yearning, Disquietment)이 다수인 이미지에서 주로 실패

---

## 미해결 / 다음 단계

| 항목 | 내용 |
|------|------|
| **레이블 스키마 정의** | ✅ 완료 (2026-06-17) — [LABEL_SCHEMA.md](ml-pipeline/context_emotion/LABEL_SCHEMA.md). emotion_class 9종 + situation_class 6종, 휴리스틱 초안을 `attack_candidates_labeled.csv`로 생성 |
| **새 레이블링** | 워크플로우 구축 완료 (2026-06-17) — `prepare_review_csv.py` → `emotion_review_queue.csv`(검수 큐, review_needed 우선 정렬 + final_emotion_class/final_situation_class/label_status/reviewer_note 컬럼) → 사람이 검수 후 `build_final_labeled_dataset.py`로 `label_status=reviewed`만 모아 `emotion_final_labeled.csv` + `emotion_label_summary.md` 생성. `score_auto_review.py`로 규칙 기반 신뢰도 점수를 매겨 auto_review(0.7%)/human_review(99.3%)/review_priority 후보를 분리 — situation_class가 emotic 출처(2,881장)에서 근거 없음(0.15) 이라 자동 승인 병목. **실제 검수 작업(3,075행)은 아직 미진행** |
| **split 재조정** | 레이블링 완료 후 train 70 / val 15 / test 15 비율로 재분배 |
| **emotic/ 18k 활용 여부** | 레이블 없는 18,351장 — pseudo-labeling 또는 pre-training 용도 검토 |
| **Qwen 평가 파이프라인** | 새 레이블 기준으로 프롬프트 재설계 + 응답 파싱 + 메트릭 정의 |

---

## 프로젝트 목적 재정의 (2026-06-17)

이 데이터셋의 목적은 감정 분류 모델 정확도 향상이 아니라 **감정추론 CAPTCHA의 강도를 검증하는
공격 모델(attacker model) 평가용 벤치마크 구축**이다. 즉 "사람은 쉽게 풀고 공격 모델(Qwen/GPT/
Gemini 등)은 틀려야" CAPTCHA 문제로서 가치가 있다 — 분류 정확도 관점과 정반대 기준으로
데이터를 다시 평가했다.

`analyze_captcha_strength.py` → `captcha_candidate_pool.csv` + `emotion_captcha_strength_report.md`

핵심 발견:
- emotion_class 9종으로 묶어도 Qwen 재정답률은 9.3%뿐 — 스키마가 너무 쉬운 건 아님.
- 그러나 오답의 41.3%(1,269건)는 "표정이 약하면 무조건 happiness로 찍는" Qwen의 단일 버릇에
  기댄 것(`bias_dependent`) — 다른 공격 모델엔 안 통할 수 있어 강도를 보장 못 함.
- 25%(768건)는 EMOTIC 원 annotator들끼리도 합의가 60% 미만 — 정답 자체가 불분명해 제외 권장.
- 진짜 신뢰할 수 있는 핵심 후보(`robust`)는 27.7%(853건)뿐.
- **다음 단계**: GPT-4V/Gemini 등으로 robust 풀 교차검증, 클래스 불균형(disconnection 38%) 해소,
  bbox가 작아 디테일이 손실되는 이미지(공격 모델에 불리) 우선 보강 검토.

**중요한 부수 발견**: `qwen_attack_results.csv`의 `actual_labels`에는 manual_label
(safety/danger/concert 등) 95건 전체에 대해 EMOTIC 26종 기준 ground truth가 이미 매핑돼 있음
(예: danger→Fear, safety→Confidence). [LABEL_SCHEMA.md](ml-pipeline/context_emotion/LABEL_SCHEMA.md)의
emotion_class 결측 95건을 이걸로 채울 수 있음 — 아직 미적용.

### 문제은행(CAPTCHA bank) 구조로 재정리 (2026-06-17)

3,075건을 학습 데이터셋이 아니라 **문제은행 후보**로 다루기로 결정. `build_captcha_bank.py`로
tier → bank_tier 재분류 + 난이도 태그 부착:

| bank_tier | 건수 | 내용 |
|---|---|---|
| core | 853 | robust 그대로, 1차 핵심 후보 |
| cross_validation_pending | 1,269 | bias_dependent, GPT/Gemini 교차검증 대기 |
| excluded | 953 | exclude_no_longer_hard + exclude_ambiguous_ground_truth |

태그: `tag_small_subject`(bbox/이미지 면적<10%), `tag_multi_person`(EMOTIC
`Annotations.mat`의 person 수 ≥2 — `person_count_lookup.csv`로 실측), `tag_context_dependent`
(저각성·맥락의존 클래스). core 중 82.8%가 맥락의존 클래스.

산출 파일: `captcha_bank_candidates.csv`, `cross_model_benchmark_template.csv`(2,122행, 멀티모델
정답 컬럼 템플릿 — qwen_correct/gpt_correct/gemini_correct/human_correct), `compute_strength_score.py`
(human_correct_rate − max(attacker_correct_rate) 계산, 현재는 qwen만 채워져 있어 부분 결과만 출력).

**다음 단계**: GPT-4V/Gemini로 2,122건 실제 교차검증 실행 → 사람 파일럿 테스트로 human_correct
채우기 → strength_score 계산 → cross_validation_pending 재분류. `label_status`는 계속 미변경.

### core 853건 검수 도구 (2026-06-17)

`review_app.py` — Streamlit 기반 검수 UI (`streamlit run review_app.py`로 실행). bbox를
빨간 사각형으로 표시한 이미지 + consensus_emotion_class/Qwen 예측/태그(multi_person,
context_dependent, small_subject)를 한 화면에서 보고, `human_label`(드롭다운, 9종)과
`human_correct`(Qwen이 사실은 맞았는지 체크박스), `reviewer_note`를 입력한다. Prev/Next 이동과
저장은 `label_status`를 바꾸지 않고, "Reviewed로 저장"/"Excluded로 저장" 버튼을 눌러야만
바뀐다. 매 동작마다 `captcha_bank_human_reviewed.csv`에 즉시 저장되어 중간에 종료해도 이어서
작업 가능(파일이 있으면 그걸 불러와 이어감, 없으면 core 853건으로 새로 시작).
