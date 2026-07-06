# 봇 데이터 생성 및 행동 패턴 설계 회고록

**일자**: 2026-04-29  
**담당**: agami MLOps

---

## 목표

실제 공격 상황을 가정한 봇 데이터 생성. 단순 랜덤 클릭 봇만으로는 현실적인 공격 패턴을 반영하기 어렵다고 판단하여 총 4가지 유형의 봇을 설계했다.

---

## 구현한 봇 유형

| 유형 | 행동 패턴 |
|------|-----------|
| `known_target_bot` | 정답 위치를 이미 알고 직선 이동 |
| `random_search_bot` | 랜덤 탐색 방식 |
| `grid_search_bot` | 화면을 지그재그 형태로 스캔 |
| `other_bot` | 속도 랜덤화 + 흔들림 기반 이동 추가 |

최종적으로 약 19,000개 이상의 봇 행동 데이터를 생성했다.

---

## 전처리 및 EDA

JSON 로그 데이터를 모델 학습용 형태로 변환. 추출한 주요 feature:

- `duration`
- `total_distance`
- `avg_speed`
- `max_speed`
- `speed_std`
- `direction_changes`
- `pauses`

EDA 분석을 통해 사람과 봇 데이터의 분포 차이를 확인했다.

---

## 트러블슈팅 — duration 값이 비정상적으로 일정한 문제

**원인**  
초기 코드에서 `duration = 900`처럼 고정값을 사용 → 모델이 시간값만 보고 봇이라고 판단하는 **shortcut learning** 문제가 발생할 가능성이 높았다.

**해결**  
```python
duration = random.randint(...)
interval = random.randint(...)
```
탐색 시간과 이동 간격이 랜덤하게 변하도록 수정하여 보다 현실적인 행동 패턴 데이터를 생성했다.
