# label_schema.json contract

매 candidate 패키지에는 `label_schema.json`이 들어가야 한다. 이 파일이
존재하는 이유: ONNX 모델의 출력 logit 순서는 학습 시점의 클래스 순서에
고정되는데, 그 순서가 운영 코드(`common/constants.py`)와 어긋나면
조용히 틀린 감정으로 응답하는 사고가 난다. 그래서 패키징/검증/스모크
테스트 전 단계에서 이 파일과 `common/constants.py`를 항상 비교한다.

## 필수 필드

```json
{
  "version": "v2",
  "emotion_classes": [
    "happiness", "calm", "anticipation", "affection", "anger", "fear",
    "sadness", "disconnection", "suffering", "aversion", "embarrassment",
    "confidence", "confusion", "yearning"
  ],
  "situation_classes": [
    "conflict", "danger", "loss_absence", "pressure", "safety",
    "teasing", "vanity"
  ]
}
```

- `emotion_classes`: ONNX `logits` 출력의 인덱스 순서와 1:1로 정확히
  일치해야 한다. 인덱스 i의 출력 = `emotion_classes[i]`.
- `situation_classes`: 지금 모델은 emotion 14종 단일 라벨만 다루므로
  (`MLOPS_OPERATION_DESIGN.md` "학습 데이터 vs 운영 평가 데이터" 참고)
  현재는 참고용 필드다. situation 분류 헤드가 추가되면 그때부터 순서
  강제를 시작한다.
- `version`: `metadata.json`의 `label_schema_version`과 정확히 같은
  문자열이어야 한다 (둘이 다르면 `validate_model_artifacts.py`가 막음).

## 검증 규칙 (코드로 강제되는 부분)

`deployment/model_store.py:validate_label_schema()`가 다음을 확인한다:

1. `emotion_classes`가 `context_emotion.common.constants.EMOTION_CLASSES`와
   **순서까지 완전히 동일**한가 (다른 순서로 같은 14개를 담아도 실패).
2. `metadata.json.label_schema_version == label_schema.json.version`.
3. `metadata.json.output_spec.shape`의 마지막 차원이
   `len(emotion_classes)`와 같은가.

지금 14클래스 스키마를 바꿀 계획이 없는 한, 모델팀은 이 파일을 그냥
`common/constants.py`에서 값을 복사해서 만들면 된다 - 별도로 새로
정의하지 말 것.
