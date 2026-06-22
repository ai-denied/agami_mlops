# Context Emotion CAPTCHA 벤치마크 회고록 (Day 3)

**일자**: 2026-06-18
**담당**: agami MLOps
**선행 문서**: [RETROSPECTIVE_20260617_context_emotion.md](../2026-06-17/RETROSPECTIVE_20260617_context_emotion.md) (목적 재정의 → CAPTCHA 강도 분석 → 문제은행 구조화 → review_app.py 완성까지)

---

## 오늘 한 일: core 853건 사람 검수 실행

어제 만든 review_app.py(Streamlit)를 실제로 돌려서 core 풀 검수를 시작했다.
어제 next-step 우선순위 2번("core 853건 사람 검수")을 실행한 것.

검수는 captcha_bank_human_reviewed.csv에 실시간으로 저장되며, 세션 도중
확인한 스냅샷 기준 진행 상황은 다음과 같다(검수가 진행형이라 확인
시점마다 수치가 늘어났음):

| label_status | 건수 | 비율(전체 3,075건 기준) |
|---|---|---|
| reviewed | 89 | 2.9% |
| excluded | 99 | 3.2% |
| pending | 2,887 | 93.9% |

core 853건 기준으로는 약 22%(188/853)가 처리된 상태.

---

## 관찰 1 — Qwen 오답 판정이 사람 검수로도 대부분 확인됨

human_correct("Qwen이 사실은 맞았는지" 체크박스) 분포:

- **False 84건 / True 5건** (reviewed 89건 중)

즉 검수자가 직접 봐도 Qwen의 원래 답이 틀렸다고 확인한 비율이 94%(84/89)로
압도적이다. 원래 데이터셋이 "Qwen이 틀린 3,075건"에서 출발했으므로
당연한 결과이지만, 사람이 직접 봐도 명백히 틀린 경우가 대부분이라는 건
— CAPTCHA 문제로서 강도(사람은 맞고 공격모델은 틀림)가 실제로 성립할
가능성이 높다는 긍정적 신호.

True 5건(검수자가 보기엔 Qwen이 맞았다고 판단한 케이스)은 ground truth
자체가 의심스러운 항목으로, 추후 별도로 골라내 재검토할 필요.

human_label 분포(reviewed 89건):

| 클래스 | 건수 |
|---|---|
| disconnection | 37 |
| disquietment | 15 |
| doubt_confusion | 14 |
| anticipation | 13 |
| sadness | 4 |
| fear | 3 |
| calm | 3 |

어제 보고서에서 지적된 "core 풀 중 82.8%가 맥락의존(context-dependent)
클래스"라는 우려가 실제 검수에서도 그대로 재현됐다 — disconnection/
disquietment/doubt_confusion/anticipation 네 클래스가 reviewed 89건의
88%(79/89)를 차지. happiness/anger 같은 고각성 클래스는 거의 등장하지
않음. next-step의 "클래스 불균형 해소" 항목이 여전히 유효하며 오히려 더
시급해졌다.

---

## 관찰 2 — "Excluded로 저장" 시 이유가 기록되지 않고 있음

excluded 99건을 원래 tier 기준으로 보면:

| 원래 tier | 건수 |
|---|---|
| bias_dependent | 51 |
| robust | 45 |
| exclude_no_longer_hard | 3 |

주목할 점: core(robust) tier에서도 45건이나 검수 단계에서 제외되고
있다 — 자동 분류 tier가 핵심 후보로 분류한 것도 사람이 보면 상당수
부적합하다고 판단한다는 뜻으로, core 853건을 그대로 문제은행에 쓸 수
없다는 게 실제 검수로 재확인된 셈.

다만 **excluded 99건 전부 reviewer_note가 비어 있다.** review_app.py는
reviewer_note 입력칸을 제공하지만, 검수자가 "Excluded로 저장" 버튼을
누를 때 이유를 적지 않고 넘어가는 패턴이 굳어지고 있음. 지금은 빠른
처리에 유리하지만, 나중에 "왜 이 853건 중 이만큼이 제외됐는지" 패턴
분석(예: bbox 문제? 표정 모호? 라벨 자체가 이상함?)을 하려면 근거가
하나도 안 남는다. 검수를 더 진행하기 전에 제외 사유를 최소 카테고리(예:
드롭다운)로라도 강제하는 게 나을 듯.

---

## 다음 단계 (우선순위 갱신)

1. **core 853건 검수 계속**: 현재 188/853(22%) 처리. 나머지 약 665건.
2. **excluded 처리 시 사유 기록 강제화**: review_app.py에 제외 사유
   선택지(드롭다운 등) 추가 검토 — 지금처럼 빈 채로 쌓이면 사후 분석
   불가.
3. **human_correct=True 5건 별도 재검토**: ground truth가 의심되는
   케이스로 분리해 원본 EMOTIC consensus와 대조.
4. **클래스 불균형**: disconnection/disquietment/doubt_confusion/
   anticipation 편중이 검수 결과로도 재확인됨 — core 풀 보강 또는
   클래스별 상한 적용을 검수 완료 전에 결정해야 나중에 다시 손대지
   않음.
5. (선행 문서에서 이어짐, 아직 미시작) **GPT-4V/Gemini 교차검증**:
   cross_model_benchmark_template.csv 2,122건. bias_dependent 1,269건이
   진짜 강도인지 Qwen 버릇인지 가리는 핵심 단계 — 사람 검수와 별개로
   병행 가능.
6. (선행 문서에서 이어짐, 아직 미시작) **emotion_class 결측 95건 보완**:
   qwen_attack_results.csv actual_labels 매핑 반영.
