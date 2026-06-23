# context_emotion ops metrics — JSONL schema

운영 중 쌓이는 시계열 지표. **학습 데이터(`context_emotion_train_dataset_v2.csv`)와
완전히 분리된 별도 파일**이다 - 이유는 `MLOPS_OPERATION_DESIGN.md`의
"학습 데이터 vs 운영 평가 데이터" 섹션 참고. 한 줄 = 하루치(또는 배치당)
한 모델 버전에 대한 집계 1건.

저장 위치: `config/runtime_contract.yaml`의 `ops_metrics.events_path`
(기본값 `/workspace/data/context_emotion/ops_metrics/events.jsonl` - 코드
저장소가 아니라 데이터 디렉터리, 다른 `data/context_emotion/*` 산출물과
같은 관례).

## 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `date` | string (YYYY-MM-DD) | 집계 날짜 |
| `model_version` | string | 이 집계 구간에 운영 중이던 `current` 모델 버전 |
| `exposures` | int | CAPTCHA 노출 수 |
| `human_correct_rate` | float\|null | 사람 정답률 |
| `human_ambiguous_rate` | float\|null | "애매함" 선택 비율 (선택지 UX 나오기 전까지 null) |
| `class_selection_distribution` | object | `{emotion_class: count}` - 사람이 고른 답 분포 |
| `attacker_proxy_solve_rate` | float\|null | 공격 프록시 정답률 (프록시 미확정이면 null) |
| `attacker_proxy_error_types` | object\|null | 공격 프록시가 틀린 유형별 건수 |
| `excluded_question_rate` | float\|null | 운영 풀에서 이번 구간에 제외된 문항 비율 |
| `pending_review_rate` | float\|null | 재검수 큐로 넘어간 비율 |
| `recorded_at` | string (ISO datetime) | 기록 시각 |

## 채우는 시점

지금은 `record_daily_metrics()`를 호출하는 실제 운영 코드가 없다 (이건
서빙 쪽에서 호출해야 하는데, 그 코드는 이 스캐폴딩 범위 밖). null이
허용된 필드는 그 데이터 소스가 아직 준비되지 않았다는 뜻이고,
`evaluation/promotion_gate.py`는 그 게이트를 `not_configured`로 처리한다.
