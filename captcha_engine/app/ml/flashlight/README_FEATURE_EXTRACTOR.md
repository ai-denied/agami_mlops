# Sentient-CAPTCHA Mouse Feature Extractor

## 1. 목적

`mouse_feature_extractor.py`는 캡챠 엔진에서 수집한 raw mouse event 로그를 모델 입력 형식으로 변환하는 코드입니다.

모델 입력은 아래 두 개로 구성됩니다.

```txt
dynamic_features: GRU 입력, shape=(1, N, 7)
static_features: MLP 입력, shape=(1, 10)
```

## 2. 입력 raw log 예시

```json
[
  {"x": 100, "y": 200, "t": 0},
  {"x": 105, "y": 203, "t": 16},
  {"x": 110, "y": 208, "t": 32}
]
```

또는 브라우저 이벤트 형태도 지원합니다.

```json
{
  "events": [
    {"clientX": 100, "clientY": 200, "timeStamp": 0},
    {"clientX": 105, "clientY": 203, "timeStamp": 16}
  ]
}
```

지원 key:

| 의미 | 지원 key |
|---|---|
| x 좌표 | `x`, `clientX`, `pageX`, `screenX`, `offsetX` |
| y 좌표 | `y`, `clientY`, `pageY`, `screenY`, `offsetY` |
| 시간 | `t`, `time`, `timestamp`, `timeStamp`, `ts`, `elapsed`, `clientTime` |

## 3. 실행 방법

```bash
python mouse_feature_extractor.py \
  --raw "./raw_mouse_log.json" \
  --out "./sample_captcha_log.json"
```

timestamp가 초 단위면:

```bash
python mouse_feature_extractor.py \
  --raw "./raw_mouse_log.json" \
  --out "./sample_captcha_log.json" \
  --timestamp-unit s
```

현재 학습 모델은 ms 기준으로 학습되었으므로, 브라우저 `performance.now()`나 event `timeStamp`처럼 ms 단위를 쓰는 것을 권장합니다.

## 4. 추출되는 dynamic_features

| 피처 | 단위 | 설명 |
|---|---|---|
| `dx` | px | 이전 좌표 대비 x 이동량 |
| `dy` | px | 이전 좌표 대비 y 이동량 |
| `dt` | ms | 이전 이벤트와의 시간 차이 |
| `distance` | px | 두 좌표 사이 이동 거리 |
| `velocity` | px/ms | 이동 속도 |
| `acceleration` | px/ms² | 가속도 |
| `angle_change` | rad | 이동 방향 변화량 |

## 5. 추출되는 static_features

| 피처 | 단위 | 설명 |
|---|---|---|
| `duration` | ms | CAPTCHA 수행 총 시간 |
| `log_count` | count | raw mouse event 개수 |
| `total_distance` | px | 전체 이동 거리 |
| `straight_distance` | px | 시작점과 끝점 사이 직선 거리 |
| `distance_ratio` | ratio | 전체 이동 거리 / 직선 거리 |
| `avg_speed` | px/ms | 평균 속도 |
| `max_speed` | px/ms | 최대 속도 |
| `speed_std` | px/ms | 속도 표준편차 |
| `direction_changes` | count | 방향 전환 횟수 |
| `pauses` | count | 긴 정지 구간 횟수 |

## 6. 엔진 연동 순서

```txt
raw mouse event 수집
→ mouse_feature_extractor.py로 dynamic/static feature 생성
→ mouse_normalizer_params_v3_policy_tuned.json 기준 정규화
→ ONNX 모델 입력
→ bot_risk_score 출력
→ 3회 누적 정책 적용
→ 좌표 정답 매칭 결과와 AND 조건으로 최종 판정
```

## 7. 주의사항

- 이 코드는 좌표 정답 여부를 판단하지 않습니다.
- 정답 좌표 매칭은 캡챠 엔진에서 별도로 처리해야 합니다.
- 이 코드는 모델이 요구하는 행동 피처만 생성합니다.
- 최종 통과 조건은 `좌표 정답 매칭 성공 AND 모델 위험도 정책 통과`로 보는 것이 맞습니다.
