# Context Emotion CAPTCHA 벤치마크 회고록 (Day 2)

**일자**: 2026-06-17
**담당**: agami MLOps
**선행 문서**: [RETROSPECTIVE_20260616_context_emotion.md](../2026-06-16/RETROSPECTIVE_20260616_context_emotion.md) (전처리/필터링, 3,075건 확보까지)

---

## 오늘의 핵심 전환: 프로젝트 목적 재정의

작업 중간에 목적이 바뀌었다. 처음엔 "VLM 감정 인식 벤치마크용 레이블 정제"로 시작했지만,
실제 목적은 **감정추론 CAPTCHA의 강도를 검증하는 공격 모델(Qwen/GPT/Gemini) 평가 파이프라인
구축**이다. 즉:

- 사람은 CAPTCHA 문제를 쉽게 풀어야 한다 (Human Accuracy 높게)
- 공격 모델은 틀려야 한다 (Attacker Accuracy 낮게)
- "감정 분류 정확도를 높인다"는 기존 ML 관점과 정반대 기준으로 데이터를 평가해야 한다

이 전환 이후의 모든 작업은 "어떤 이미지가 공격 모델을 잘 속이는가"를 기준으로 재설계했다.

---

## 작업 1 — 레이블 스키마 정의

`ml-pipeline/context_emotion/LABEL_SCHEMA.md`

EMOTIC 26클래스 + manual_images 28클래스(서로 다른 체계, 감정/상황이 뒤섞여 있던 문제)를
정리해 **emotion_class 9종**(happiness/calm/anticipation/disconnection/doubt_confusion/
disquietment/fear/anger/sadness) + **situation_class 6종**(safety_danger/social_gathering/
conflict/loss_absence/competition_pressure/everyday) 두 축으로 분리.

`build_emotion_class.py`로 매핑표를 적용해 `attack_candidates_labeled.csv`(3,075행)에
emotion_class/situation_class 초안 + `review_needed` 플래그를 생성. 결과: emotion_class
결측 95건(manual_label 중 situation 전용 단어 — safety/danger/concert), situation_class는
emotic 출처(2,881건) 전체가 근거 없는 'everyday' placeholder.

---

## 작업 2 — 새 레이블링(검수) 워크플로우

`prepare_review_csv.py` → `emotion_review_queue.csv` (3,075행, review_needed 우선 정렬 +
`final_emotion_class`/`final_situation_class`/`label_status`/`reviewer_note` 컬럼 추가)

`build_final_labeled_dataset.py` → `label_status=reviewed`인 행만 모아 학습용
`emotion_final_labeled.csv` + `emotion_label_summary.md` 생성. `excluded`는 제외,
`reviewed`인데 final_* 값이 비어 있으면 경고 후 제외하는 방어 로직 포함.

이 단계까지는 아직 "분류 정확도" 관점이었다 — 목적 재정의 이후 문제은행 방향으로 대체됨.

---

## 작업 3 — 자동 검수 신뢰도 스코어링

`score_auto_review.py` — 사람 검수량을 줄이기 위해 emotion_confidence/situation_confidence를
규칙 기반으로 계산해 `auto_review.csv`/`human_review.csv`/`review_priority.csv` 분리.

핵심 결과: 자동 승인 가능한 건 0.7%(21/3,075)뿐. emotion_confidence만 보면 82.8%가 고신뢰지만,
situation_class가 emotic 출처에서 근거가 전혀 없어(0.15) 병목이 됨 — situation_class는 v1에서
별도 레이블링 없이는 신뢰할 수 없다는 게 솔직한 결론.

---

## 작업 4 — 목적 재정의 후: CAPTCHA 강도 분석

`analyze_captcha_strength.py` → `captcha_candidate_pool.csv` + `emotion_captcha_strength_report.md`

Qwen이 틀린 3,075건을 "분류 정확도"가 아니라 "공격 모델 혼동 유발력" 기준으로 재평가:

- emotion_class 9종으로 묶어도 Qwen 재정답률은 9.3%(286건)뿐 — 스키마가 너무 헐겁지 않음
- 오답 패턴은 거의 전부 "표정이 약하면 무조건 happiness"라는 **Qwen의 단일 버릇**으로 설명됨
  (disconnection→happiness 516건, anticipation→happiness 321건 등) — 다양한 맥락 오해가
  아니라 한 가지 약점에 가까움
- EMOTIC 원본 annotator 간 합의(consensus_share)가 60% 미만인 768건은 정답 자체가
  불분명 — 제외 권장
- tier 분류 결과: `robust` 853건(27.7%, 핵심 후보) / `bias_dependent` 1,269건(41.3%, Qwen
  버릇에만 의존 — 교차검증 필요) / `exclude_no_longer_hard` 185건 / `exclude_ambiguous_ground_truth` 768건

부수 발견: `qwen_attack_results.csv`의 `actual_labels`에 manual_label 95건 전체의 EMOTIC
기준 ground truth가 이미 매핑돼 있었음(danger→Fear, safety→Confidence 등) — 작업 1의
emotion_class 결측 95건을 채울 수 있는 정보, 아직 미적용.

---

## 작업 5 — 문제은행(CAPTCHA bank) 구조로 재구성

3,075건을 **학습 데이터셋이 아니라 문제은행 후보**로 다루기로 결정.

`build_captcha_bank.py` → tier를 bank_tier로 재분류 + 난이도 태그 부착:

| bank_tier | 건수 | 내용 |
|---|---|---|
| core | 853 | robust 그대로, 1차 핵심 후보 |
| cross_validation_pending | 1,269 | bias_dependent, GPT/Gemini 교차검증 대기 |
| excluded | 953 | exclude_no_longer_hard + exclude_ambiguous_ground_truth |

태그 3종: `tag_small_subject`(bbox/이미지 면적<10%), `tag_multi_person`(EMOTIC
`Annotations.mat`에서 person 수 ≥2를 실측 — `person_count_lookup.csv`, 23,554행 신규 생성),
`tag_context_dependent`(저각성·맥락의존 클래스). core 중 82.8%가 맥락의존 클래스로 확인.

`cross_model_benchmark_template.csv`(2,122행 = core+pending, excluded 제외)는 향후
GPT-4V/Gemini/사람 결과를 채울 공통 스키마(`qwen_correct`/`gpt_correct`/`gemini_correct`/
`human_correct` + 각 모델 예측 라벨).

`compute_strength_score.py` — `strength_score = human_correct_rate − max(attacker_correct_rate)`
계산기. 한 모델만 속이는 문제는 강도로 인정하지 않는다는 원칙을 반영. 현재는 qwen_correct만
채워져 있어(전부 False) 부분 통계만 출력하도록 구현, GPT/Gemini/사람 결과가 채워지면 그대로
재실행 가능.

---

## 작업 6 — core 853건 사람 검수 도구

`review_app.py` (Streamlit, `streamlit run review_app.py`로 실행 — 부팅 확인 완료)

- bbox를 빨간 사각형으로 표시한 이미지 + consensus_emotion_class/Qwen 예측/태그 3종을
  한 화면에서 확인
- `human_label`(9종 드롭다운, 기존 라벨로 프리필) / `human_correct`(Qwen이 사실은 맞았는지
  체크박스) / `reviewer_note` 입력
- ◀이전/다음▶ 이동, 진행률 바 + reviewed/excluded/pending 카운트
- **label_status는 "Reviewed로 저장"/"Excluded로 저장" 버튼을 눌러야만 바뀜** — 이동/저장만으로는
  안 바뀜
- 매 동작마다 `captcha_bank_human_reviewed.csv`에 즉시 저장 → 중단 후 재실행 시 이어서 작업

---

## 오늘 생성된 파일 전체 목록

`ml-pipeline/context_emotion/` 기준:

| 파일 | 종류 | 설명 |
|---|---|---|
| `LABEL_SCHEMA.md` | 문서 | emotion_class 9종 + situation_class 6종 정의 |
| `build_emotion_class.py` | 스크립트 | 휴리스틱 라벨 매핑 |
| `attack_candidates_labeled.csv` | 데이터 | 매핑 초안 (3,075행) |
| `prepare_review_csv.py` | 스크립트 | 검수 큐 생성 |
| `emotion_review_queue.csv` | 데이터 | 검수 큐 (label_status 전부 pending 유지) |
| `build_final_labeled_dataset.py` | 스크립트 | reviewed만 학습셋으로 변환 (현재 미사용 경로) |
| `score_auto_review.py` | 스크립트 | 규칙 기반 신뢰도 스코어링 |
| `auto_review.csv` / `human_review.csv` / `review_priority.csv` | 데이터 | 신뢰도 기반 분리 |
| `analyze_captcha_strength.py` | 스크립트 | CAPTCHA 강도 관점 재평가 |
| `captcha_candidate_pool.csv` | 데이터 | tier(robust/bias_dependent/exclude_*) 부착 |
| `emotion_captcha_strength_report.md` | 문서 | 6+2개 분석 항목 리포트 |
| `person_count_lookup.csv` | 데이터 | EMOTIC Annotations.mat에서 추출한 person 수 |
| `build_captcha_bank.py` | 스크립트 | bank_tier 재분류 + 태그 부착 |
| `captcha_bank_candidates.csv` | 데이터 | 전체 3,075건 + bank_tier/태그/멀티모델 컬럼 |
| `cross_model_benchmark_template.csv` | 데이터 | 교차검증용 템플릿 (2,122행) |
| `compute_strength_score.py` | 스크립트 | Strength Score 계산기 |
| `review_app.py` | 앱 | Streamlit 검수 도구 |

---

## 다음 단계 (우선순위 순)

1. **GPT-4V/Gemini 교차검증**: `cross_model_benchmark_template.csv` 2,122건 실행, `gpt_correct`/
   `gemini_correct` 채우기 — 현재 `bias_dependent` 1,269건이 진짜 강도인지 Qwen 버릇인지
   가려낼 핵심 단계.
2. **core 853건 사람 검수**: `review_app.py`로 실제 검수 진행, `captcha_bank_human_reviewed.csv`
   확보.
3. **사람 파일럿 테스트**: `human_correct` 채워서 `compute_strength_score.py`로 실제
   strength_score 산출.
4. **emotion_class 결측 95건 보완**: `qwen_attack_results.csv`의 `actual_labels`에 이미 있는
   매핑(danger→Fear 등) 반영.
5. **bank_tier 재산정 스크립트**: 교차검증 결과 반영해 cross_validation_pending → core/excluded
   재배치.
6. **클래스 불균형 해소**: core 풀의 disconnection/doubt_confusion 비중이 과반 — 클래스당
   상한 후 다양성 확보.

`label_status`(emotion_review_queue.csv)는 오늘 작업 전체에서 한 번도 변경되지 않았다 —
모든 tier/bank_tier/태그는 사람 최종 검수 전의 우선순위·후보 제안이다.
