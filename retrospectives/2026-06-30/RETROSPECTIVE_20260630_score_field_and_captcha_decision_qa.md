# 회고록 — 2026-06-30

**주제**: /attempt score 필드 추가 (감정 그룹 기반 0.5) + captcha_decision.py 위젯팀 QA 대응

---

## 1. 오늘 한 일

| 항목 | 결과 |
|---|---|
| `/attempt` 응답에 `score` 필드 추가 | `schemas.py`, `app.py`, `attempt_logger.py` 수정 완료 |
| 0.5 판정 기준 재정의 | `aux_emotions` → 감정 그룹 일치 (`EMOTION_TO_GROUP`) |
| captcha_decision.py 위젯팀 QA 7개 항목 검토 | 5개 코드로 확인, 2개 설계 얼라인 필요 확인 |
| 라운드 정의 불일치 발견 | 모델팀: 단일 미션 3개 혼합 / 위젯팀: 얼굴+손 쌍 × 3 |

---

## 2. score 필드 추가

### 배경

위젯팀이 3회 누적 점수 정책(총점 ≥ 2.5 통과)을 구현하려면 문항별로 `1.0 / 0.5 / 0.0` 구분이 필요했다. 기존 `/attempt`는 `is_correct: bool`만 반환해 0.5를 줄 방법이 없었다.

### 0.5 기준 선택 과정

원래 의도는 `aux_emotions`(팀원이 직접 풀며 수집한 보조 정답)로 0.5를 판정하는 것이었다.
그런데 `aux_emotions`는 `final_emotion`과 너무 유사해 선택지에 넣으면 정상 사용자도 헷갈려 틀리는 문제가 있어, 이미 `generate_choices()`에서 선택지 풀로부터 제외(`excluded` 셋)해두었다.

선택지에 없는 레이블은 `validate_and_score()`에서 즉시 거부되므로, `aux_emotions` 기반 `points=0.5`는 구조적으로 발생 불가였다.

대안으로 **감정 그룹 일치**를 0.5 기준으로 채택했다:
- 선택지 구성 변경 없음 (aux 안 넣어도 됨)
- `EMOTION_TO_GROUP`은 서버 내부에만 존재 → 공격자 역이용 어려움
- 이미 선택지 오보기가 같은 그룹에서 나오므로 0.5 케이스가 실제로 발생함

### 변경 내용

```
attempt_logger.py: 0.5 판정을 aux_emotions → EMOTION_TO_GROUP 일치로 변경
schemas.py:        AttemptResponse에 score: float 필드 추가
app.py:            응답에 score=points 포함
```

### 채점 기준 (확정)

| 상황 | score | is_correct |
|------|-------|------------|
| final_emotion 정확 일치 | 1.0 | true |
| 같은 감정 그룹 선택 | 0.5 | false |
| 다른 그룹 선택 | 0.0 | false |

---

## 3. captcha_decision.py 위젯팀 QA

위젯팀이 "captcha_decision.py가 레포에 없다"며 7개 항목 확인을 요청했다.
실제로는 `ml-pipeline/facial_recognition/captcha_decision.py`에 이미 커밋돼 있었다.

### 항목별 결과

| # | 항목 | 결과 |
|---|------|------|
| 1 | captcha_decision.py 커밋 요청 | 이미 존재 — 블로커 아님 |
| 2 | 라운드 = 단일 미션? 쌍? | 코드 확정: 단일 미션 (`Literal["face","hand"]`) |
| 3 | 3라운드 구성 비율 | **설계 불일치 발견** (아래) |
| 4 | total_risk 공식·가중치·결측 기본값 | 코드 확정: `BAND_RISK_WEIGHT`, 결측=0.0 |
| 5 | face_detected/timeout 판정 기준 | **미결 — 위젯팀이 기준 직접 설정 필요** |
| 6 | mission_type enum 엄격성 | 코드 확정: `face`\|`hand`만, `pair` 불가 |
| 7 | spoof 없는 손 라운드 기본값 | 코드 확정: `spoof_score=0.0`, `face_detected=True` |

### 설계 불일치: 라운드 정의

- **모델팀 설계**: 단일 미션 3개 혼합 (face+hand+face 또는 hand+face+hand). `decide_three_round_captcha`가 정확히 3개 `MissionRound` 수신.
- **위젯팀 의도**: 얼굴+손 한 쌍을 3번 = 6개 미션.

현재 코드로는 6개 입력 시 `ValueError("CAPTCHA requires exactly 3 round results.")` 발생. 위젯팀과 구조 합의 필요 (6라운드로 함수 변경 vs. 쌍 집계 후 3개로 압축).

---

## 4. 아쉬운 점

- **captcha_decision.py 위치를 위젯팀이 몰랐음**: 파일이 `facial_recognition/` 하위에 있어 컨텍스트 모델 관련으로 찾다가 못 찾은 것으로 보임. 인터페이스 파일 위치를 명세서나 README에 명시해두었으면 왕복이 줄었을 것.
- **aux_emotions 설계 의도가 문서화되지 않았음**: `generate_choices()`에서 aux를 제외하는 이유가 주석에 있긴 하지만, "그래서 0.5 판정이 구조적으로 불가"라는 연결고리가 명시되지 않아 위젯팀이 "보조정답이 모델에 들어가 있다"고 오해했다.

---

## 5. 다음 액션

1. **라운드 정의 얼라인**: 위젯팀과 6라운드 방식 vs. 3라운드 혼합 방식 합의 후, 필요 시 `captcha_decision.py` 시그니처 변경.
2. **face_detected/timeout 판정 기준 전달**: 위젯팀이 어떤 조건에서 `face_detected=False`로 세팅해야 하는지 기준 제시 필요.
3. **push**: 오늘 커밋(`feat: /attempt 응답에 score 필드 추가`) 아직 로컬 상태 — 합의 완료 후 push.
