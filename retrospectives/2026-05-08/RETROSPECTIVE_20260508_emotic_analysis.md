# EMOTIC 분석 및 AI 공격 검증 시작 회고록

**일자**: 2026-05-08  
**담당**: agami MLOps

---

## EMOTIC 분석 및 전처리

- dataset 구조 확인, train / val / test 구조 분석
- `Annotations.mat` 내부 구조 분석 — Python에서 MATLAB `.mat` 구조를 직접 파싱하는 코드 구현
- `.mat` 파일에서 이미지 경로, 감정 라벨, bbox, 성별/연령, valence/arousal/dominance 정보 추출

**결과**: 총 23,554개 데이터 CSV 변환 완료 → `emotic_annotations.csv` 생성

---

## CAPTCHA 목적 감정 후보 필터링

단순 표정만으로는 판단하기 어려운 감정을 중심으로 필터링:

- Disconnection
- Doubt/Confusion
- Embarrassment
- Fear
- Sadness
- Disquietment
- Sympathy
- Yearning

**결과**: CAPTCHA 후보 이미지 약 3,768개 추출 → `emotic_captcha_candidates.csv` 생성. 후보 CSV 기반으로 실제 이미지를 자동 복사하는 스크립트도 구현해 CAPTCHA 검증용 데이터셋 구성을 완료했다.

---

## AI 공격 테스트 파이프라인 구축

가장 중요했던 작업: **"AI가 실제로 이 CAPTCHA를 얼마나 잘 푸는가?"** 검증.

OpenAI GPT-4o-mini Vision 기반 자동 공격 테스트 파이프라인 구축.
구조: 이미지 입력 → 감정 추론 수행 → 실제 라벨과 비교 → 결과 CSV 저장

### 1차 AI 검증 테스트 결과

EMOTIC 샘플 30장 대상 테스트.

| 결과 | 수치 |
|------|------|
| 정답 | 15개 |
| 오답 | 15개 |
| 정답률 | 약 50% |

---

## 분석 결과

AI는 여전히 표정 중심으로 판단하는 경향이 강했다. 특히 Disconnection, Embarrassment, Doubt/Confusion 같은 감정에서 혼동이 많이 발생했다. 일부 이미지는 상황 맥락 해석 자체를 실패하는 모습도 확인했다.

---

## 가장 중요했던 의미

이번 검증을 통해 **AI가 실제로 헷갈리는 상황형 감정 데이터가 존재한다**는 점을 확인했다. 단순 이미지 분류 CAPTCHA가 아니라 맥락 이해 기반 캡챠 가능성을 확인한 중요한 과정이었다.

## 미해결 / 다음 단계

- AI가 잘 맞추는 데이터 제거
- AI가 헷갈리는 데이터 강화
- 인간 행동 패턴 + 상황 이해 결합
