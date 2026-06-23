# context_emotion 라벨 재구성 노트

`ml-pipeline/context_emotion/`는 pod 리셋으로 비어 있었고, 4일치(Day1~5) 작업
(review_app.py, engagement_review_app.py, restore_original_emotic_labels.py,
captcha_bank_human_reviewed.csv, manual_images_labeled.csv, excluded_pool.csv,
qwen_attack_results.csv, qwen_recheck_queue.csv 등)이 전부 소실된 상태에서,
유일하게 남은 기록인 Day4/Day5 회고록 텍스트를 스펙으로 삼아 **근사 재구축**한
결과물이다. 사람이 한 건씩 검수한 개별 판단은 원천적으로 복원 불가능하므로,
아래 항목들은 "원본과 100% 동일"이 아니라 "문서화된 규칙으로 재현 가능한 한도
내의 근사치"임을 명시한다.

## 최종 라벨 목록 (2026-06-22 확정, everyday 제거까지 반영 — 아래 절들 참고)

**감정 14종**: happiness, calm, anticipation, affection, anger, fear, sadness,
disconnection, suffering, aversion, embarrassment, confidence, confusion,
yearning

**상황 7종**: conflict, danger, loss_absence, pressure, safety, teasing, vanity

이 14+7이 현재 유효한 최종 스키마다. 중간에 감정 15종(empathy 포함)→14종,
상황 7종→8종(everyday 포함)→다시 7종(everyday 제거)으로 두 차례씩 바뀌었는데,
그 경위는 바로 아래 절들에 시간순으로 남겨둔다.

## 2026-06-22 스키마 고정 — 감정 14종(empathy 제외) / 상황 8종(everyday 포함)

학습용 데이터셋(`context_emotion_train_dataset_v1.csv`)을 만들기 시작하면서,
사용자가 최종 스키마를 감정 14종/상황 8종으로 다시 고정했고, `emotion_mapping.py`
자체도 이 스키마로 맞추기로 했다 (이전의 15종/7종 버전은 폐기).

- **empathy 제외**: 바로 이전 단계에서 사용자 확인을 거쳐 "manual 전용 15번째
  감정 클래스"로 유지하기로 했었지만, 이번 14종 고정 스키마에는 empathy가 들어갈
  자리가 없다. `manual_images/empathy/`(4건)는 더 이상 emotion_class를 받지
  못하고, EMOTIC의 Engagement/Surprise와 동일하게 **스키마 외 라벨로 드롭**되어
  `manual_images_unresolved.csv`로 이동했다 (`emotion_mapping.NO_SCHEMA_SLOT_FOLDERS`).
  원본 폴더명(empathy)은 그대로 보존되어 있으니, 나중에 schema를 다시 넓히면
  복원 가능하다.
- **everyday 포함**: 이전 단계에서는 회고록의 "everyday 단독 라벨 행은 전부
  제거" 규칙을 따라 situation에서 everyday를 완전히 뺐었는데, 이번 고정
  스키마는 everyday를 정식 8번째 situation으로 되살렸다. 다만 **실제 데이터에는
  everyday로 매핑되는 행이 하나도 없다** — manual_images에 "everyday" 폴더가
  없고 EMOTIC 쪽은 situation 축 자체가 없기 때문이다. 스키마 정의에는 있지만
  건수는 0인 상태로, 향후 별도 데이터 보강이 필요하면 그때 채워야 한다.
- 이 변경으로 `captcha_bank_human_reviewed.csv`(emotic)는 영향 없음(26,466건
  그대로 — empathy/everyday 둘 다 emotic과 무관), `manual_images_labeled.csv`는
  228건 → **224건**으로, `manual_images_unresolved.csv`는 0건 → **4건**으로
  바뀌었다.

## 2026-06-22 everyday 재제거 — 상황 8종 → 7종

v1 학습셋(`context_emotion_train_dataset_v1.csv`)을 빌드해보니 everyday는
스키마에 정의만 있고 실제로 매핑되는 행이 0건이었다 (`SITUATION_CLASSES`에
넣었을 때부터 이미 예고된 상태). 사용자가 이 빈 슬롯을 스키마에서 다시
빼기로 확정해 **situation은 7종(conflict/danger/loss_absence/pressure/
safety/teasing/vanity)으로 돌아갔다**. 데이터 자체에는 영향이 없다 —
어차피 0건이었으므로 `captcha_bank_human_reviewed.csv`, `manual_images_labeled.csv`,
`context_emotion_train_dataset_v1.csv` 어느 것도 행 수가 바뀌지 않고,
`context_emotion_label_distribution_v1.md` / `context_emotion_label_mapping_v1.json`의
situation 목록에서만 everyday 항목이 빠진다.

## 2026-06-22 수정 — 사용자가 제공한 "최종 감정 풀 - 14종" 표로 교정

사용자가 회고록과는 별도로 통합 원본 라벨까지 명시한 최종 표를 제공해, 그
표를 기준으로 아래 3가지를 바로잡았다 (이 표가 회고록 텍스트보다 우선하는
권위 있는 소스로 취급):

- **Esteem, Sympathy**: 원래 calm으로 잘못 매핑했었는데, 표에는
  `affection ← affection, sympathy, esteem`으로 명시되어 있어 **affection**으로
  수정. calm은 이제 `Peace`만 남는다.
- **Sensitivity**: 원래 confusion으로 매핑했었는데(Day4 doubt_confusion
  그룹핑 추정에 근거한 불확실한 가정이었음), 표에는
  `aversion ← aversion, sensitivity`로 명시되어 있어 **aversion**으로 수정.
- **manual despair(4건)**: 원래 "Qwen 공격이 안 돌아서 미해결"로 보류했는데,
  표에는 `sadness ← sadness, despair`로 명시되어 있어 **sadness로 확정**.
  `manual_images_unresolved.csv`는 이제 0건이고, despair 4건은
  `manual_images_labeled.csv`에 emotion_class=sadness로 들어간다.

이 표는 14종(emotion)만 다루고 **empathy는 포함하지 않았다** — 이 시점에는
사용자가 일단 "EMOTIC 14종 + manual 전용 empathy로 15종 유지"를 확인했었으나,
그 결정은 위 "2026-06-22 스키마 고정" 절에서 **다시 뒤집혀 최종적으로 14종으로
확정**됐다. 마찬가지로 이때는 situation도 "7종 유지(everyday 제외)"로
확인했었지만, 이 역시 스키마 고정 절에서 **8종(everyday 포함)으로 뒤집혔다**.
이 단락은 그 중간 결정 과정의 기록으로 남겨두고, 현재 유효한 스키마는 항상
파일 상단의 "최종 라벨 목록" 절을 기준으로 한다.

표의 "대표적으로 함께 쓰는 상황" 컬럼(예: happiness ↔ everyday/safety/teasing)은
라벨-폴더 매핑 규칙이 아니라 **감정-상황 조합 가이드**(캡차 문항 구성 시 어떤
감정 이미지에 어떤 상황 컨텍스트를 붙일지 결정하는 용도로 보임)로 판단해,
`emotion_mapping.py`의 매핑 로직에는 반영하지 않았다. 필요하면 별도
co-occurrence 테이블로 만들 수 있다.

## 그 외 재구성 시 적용한 가정 (참고용)

- **EMOTIC Engagement, Surprise**: 14종 표에도 자리가 없고 병합 규칙에도
  언급이 없어 **드롭**했다 (excluded_pool.csv로 분리). Engagement는 원래도
  895건 중 845건이 사람 검수로 제외됐었으므로, "전부 제외"가 가장 근접한
  근사다. 다만 원래 50건은 사람이 직접 14종 중 하나로 재분류했었는데 그
  개별 판단(어떤 50건을 어떤 라벨로)은 복원할 수 없다.
- **manual concert 폴더 → conflict**: 폴더명은 "concert"인데 최종표의
  situation 라벨명은 "conflict"다. 두 값의 건수가 정확히 16건으로 일치해
  같은 폴더를 가리키는 것으로 판단해 매핑했다.
- **EMOTIC 규모**: 이번에 복원한 `captcha_bank_human_reviewed.csv`는 EMOTIC
  전체 코퍼스(26,466 person-instance, mscoco/framesdb/emodb_small/ade20k 전체)
  기준이다. Day5의 최종 수치(1,737건)는 `attack_candidates.csv`라는 별도
  후보 선별 단계를 거친 부분집합인데, 그 선별 파일 자체가 소실되어 동일한
  부분집합을 재현할 수 없다. 따라서 규모가 원본보다 훨씬 크며, 재선별이
  필요하다면 추가 작업이다.

## 검증

manual 쪽은 회고록에 기록된 최종 분포 숫자와 대조했을 때 거의 정확히
일치했다 (감정 99건 → 교정 후 despair 포함 103건, 상황 128건, 폴더별 건수
대부분 ±0~1건 차이). 이는 폴더명 → 라벨 매핑 역산이 합리적이라는 근거지만,
emotic 쪽은 애초에 비교 대상인 원본 1,737건 자체가 없으므로 같은 방식의
검증은 불가능하다.
