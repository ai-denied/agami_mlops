# Context Emotion CAPTCHA 벤치마크 회고록 (Day 4)

**일자**: 2026-06-19
**담당**: agami MLOps
**선행 문서**: [RETROSPECTIVE_20260618_context_emotion.md](../2026-06-18/RETROSPECTIVE_20260618_context_emotion.md) (core 853건 사람 검수 착수, Qwen 오답 94% 확인, excluded 사유 미기록 문제 발견)

---

## 오늘의 핵심 — 사람 검수 방향을 전면 전환함

어제까지는 review_app.py로 3,075건을 처음부터 새로 검수하는 흐름이었다.
오늘 작업 중 **"EMOTIC은 원래 라벨링이 다 되어있던 데이터셋 아니었나?"**
라는 질문에서 출발해, 새로 검수하는 대신 **EMOTIC 원본 라벨을 복원해서
그대로 쓰는 방향**으로 전환했다. 결과적으로 사람 검수는 442건(원본
라벨이 매핑 안 되는 케이스)만 남기고 대부분 종료.

---

## 1. review_app.py 버그 수정 (오전)

- **이미지 깨짐 버그**: "image file is truncated" 에러가 새로고침해도
  계속 발생.
  - 원인: is_valid_image()가 PIL verify()만 썼는데, 이게 잘린 파일도
    통과시킴 → framesdb 쪽 손상 사본(emotic/framesdb/images/)을 먼저
    골라버리고, 정상 사본(emotic/framesdb/framesdb/images/)이 있어도
    못 씀. emotic 2,881건 중 **68건**이 해당.
  - 수정: verify() 대신 convert("RGB")까지 끝까지 디코드해서 검증하도록
    변경 → 68건 전부 정상 사본으로 해결.
- **manual 이미지 194건이 검수 큐에 섞여 있던 문제**: 폴더명(aggression,
  danger, joy 등)으로 이미 라벨이 확정된 생성형 이미지인데 emotic과
  함께 pending 상태로 검수 대상에 들어가 있었음. manual_images_labeled.csv로
  분리하고 captcha_bank_human_reviewed.csv에서 제거 (2,881 → emotic만
  2,881건... 정정: manual 194건 제거 후 emotic 2,881건만 남음).

## 2. EMOTIC 원본 라벨 검증 — categories 컬럼이 이미 손실 압축돼 있었음

Annotations.mat 원본과 attack_candidates.csv의 categories 컬럼을 직접
대조(emotic 3,728건, split+filename+bbox 매칭):

| 결과 | 건수 | 비율 |
|---|---|---|
| 원본 멀티레이블 그대로 보존 | 1,266 | 34% |
| 원본 중 일부 누락(보통 1개만 남김) | 2,462 | **66%** |

예: COCO_val2014_000000562243.jpg 원본 = ['Disconnection', 'Doubt/Confusion']
→ CSV에는 ['Disconnection']만.

RETROSPECTIVE_20260616_context_emotion.md에 이미 "categories 포맷 혼재"
문제로 기록돼 있었고 당시 "새 레이블링으로 대체할 거라 정제는
불필요"로 보류된 상태였다. VAD 연속값(valence/arousal/dominance)도
라벨링 단계에서 컬럼 자체가 사라져 있었음.

## 3. 방향 전환: 사람 검수 중단, EMOTIC 원본 라벨 복원

새 스크립트 [restore_original_emotic_labels.py](../../ml-pipeline/context_emotion/restore_original_emotic_labels.py)
작성, 실행 결과:

1. **excluded 240건**(검수 중 누적된 제외 건) → 본 풀에서 빼서
   excluded_pool.csv로 별도 보관(삭제 아님).
2. 남은 2,641건(기존 reviewed 224건 포함 전부) → Annotations.mat에서
   원본 annotator 카테고리를 다시 찾아 **과반(≥50%) 동의 기준**으로
   채택(train은 단일 annotator라 그대로 채택). EMOTIC_TO_EMOTION
   매핑표(build_emotion_class.py와 동일, 10클래스)로 변환해
   human_label/human_labels에 멀티레이블로 반영, label_status='reviewed',
   reviewer_note='restored_from_emotic_original_multilabel'로 표시.
3. **situation_class 축 완전 폐지** — consensus_situation_class/
   human_situation/human_situations 컬럼 삭제. (어제까지 검수 화면에
   있던 상황 라벨링 입력 전부 제거)
4. **442건은 매핑 실패** → 기존 값 유지하기로 결정, label_status='pending',
   reviewer_note='unmapped_needs_manual_review'로 표시해 따로 추적.
   - 262건: 과반이 Engagement(몰입/관심) — 애초에 10클래스 매핑표에
     없는 카테고리.
   - 180건: annotator 간 과반 합의 자체가 안 됨(의견 갈림).

## 4. review_app.py 재정비

- bbox 빨간 사각형 표시 제거(요청 사유: "엉뚱한 사람한테 박스가 잡힐
  때가 있어서 오히려 방해됨"). bbox 좌표 데이터는 그대로 두고 화면
  표시만 뺌.
- situation_class 관련 UI(멀티셀렉트, consensus 표시, 저장 로직)
  전부 제거 — 데이터에서 컬럼이 없어졌는데 코드가 참조하던 부분이라
  안 고치면 에러 났을 부분.
- 큐가 자동으로 442건(unmapped)만 보이게 됨 — 나머지는 reviewed라
  Prev/Next 탐색에서 건너뜀.

## 5. 미해결 — 10클래스 압축이 너무 거칠다는 문제 제기

emotic(원본 26클래스) + manual(폴더명 28종)을 객관식 보기 통일을 위해
10클래스로 묶어둔 상태인데, "라벨 선택지가 한정적"이라는 피드백.
압축을 풀면 emotic과 manual이 서로 다른 단어 집합을 쓰게 돼 객관식
보기를 하나로 통일할 수 없다는 트레이드오프를 확인하고, **다음에
이어서 논의하기로** 함.

오늘 뽑아둔 분포:

**emotic 원본 26클래스 분포** (과반 동의 기준, excluded 240건 포함
전체 2,881건, 멀티레이블 1,235건 포함):

Disconnection 990 · Engagement 895 · Yearning 379 · Doubt/Confusion 315 ·
Anticipation 299 · Disquietment 251 · Sadness 185 · Fatigue 157 ·
Confidence 151 · Peace 131 · Excitement 121 · Happiness 99 · Sympathy 98 ·
Suffering 96 · Fear 81 · Embarrassment 73 · Pleasure 51 · Sensitivity 40 ·
Surprise 37 · Annoyance 37 · Disapproval 34 · Pain 27 · Esteem 24 ·
Affection 21 · Anger 18 · Aversion 14

**manual 폴더명 분포** (현재 라벨된 194건 기준):

safety 55 · danger 39 · nervous 23 · aggression 12 · elation 9 ·
teasing 8 · joy 8 · exhausted 7 · embarrassment 3 · emptiness 3 ·
bittersweet 3 · relief 3 · missing 3 · euphoria 3 · manic 3 ·
protest 2 · (alienation/concert/bullying/hope/jealousy/forgiveness/
pressure/superiority/vanity/warmth 각 1)

**다음에 결정할 것**:

1. Engagement(895건, emotic 2위 빈도) — 독립 클래스로 살릴지,
   calm/anticipation에 합칠지.
2. safety/danger/concert/pressure/superiority/vanity — 감정이 아니라
   상황을 가리키는 manual 폴더명. situation_class를 없앤 지금 이걸
   감정으로 어떻게 해석할지(예: danger→fear, safety→calm 식 변환
   유지 여부).
3. 명백한 동의어 군집 묶기: happiness 계열(Happiness/Excitement/
   Pleasure/joy/elation/euphoria/manic), anger 계열(Anger/Annoyance/
   Disapproval/aggression/protest), sadness 계열(Sadness/Suffering/
   Pain/Aversion), disconnection 계열(Fatigue/Disconnection/exhausted/
   emptiness/alienation), doubt_confusion 계열(Embarrassment/
   Sensitivity/embarrassment/teasing), calm 계열(Affection/Peace/
   Esteem/Confidence/Sympathy/warmth/relief/hope/forgiveness).
4. Anticipation(299) vs Yearning(379) 분리 유지 여부(2026-06-18에
   이미 한 번 분리 결정함).
5. 최종 클래스 수를 몇 개로 갈지(현재 10개 → 15~20개 검토 제안 있었음)
   합의 필요.
6. (선행 문서에서 계속 미해결) GPT-4V/Gemini 교차검증, emotion_class
   결측 95건 — 이번 라벨 복원으로 상당 부분 해소됐을 가능성 있음,
   재확인 필요.

---

## 산출물 정리

| 파일 | 상태 |
|---|---|
| captcha_bank_human_reviewed.csv | 2,641건. reviewed 2,199 + pending(unmapped) 442. situation 컬럼 없음 |
| excluded_pool.csv | 신규. 240건, 본 풀에서 제외된 것만 보관 |
| manual_images_labeled.csv | 신규. manual 194건, 폴더명 기반 라벨 그대로 |
| restore_original_emotic_labels.py | 신규. 원본 라벨 복원 스크립트(재실행 가능) |
| review_app.py | bbox 표시 제거, situation UI 제거, 442건 전용 큐로 동작 |
