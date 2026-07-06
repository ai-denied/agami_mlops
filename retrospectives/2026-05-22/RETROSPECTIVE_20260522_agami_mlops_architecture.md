# AGAMI 전체 MLOps 구조 설계 회고록

**일자**: 2026-05-22  
**담당**: agami MLOps

---

## 설계 목표

AGAMI 서비스 전체의 MLOps 구조를 설계했다. 핵심은 CAPTCHA 유형이 여러 개로 나뉘더라도 **공통 MLOps 파이프라인 구조는 동일하게 가져가는 것**이었다.

캡챠마다 사용하는 feature나 모델은 다르다.
- 손전등 CAPTCHA: 마우스 행동 로그 중심
- 감정 CAPTCHA: 이미지/라벨/AI 공격 검증 중심
- 실시간 행동 CAPTCHA: 얼굴/손 landmark 중심

이지만, 전체 ML 운영 흐름은 공통화할 수 있다고 판단했다.

---

## 공통 MLOps 파이프라인 구조

```
Behavior Log Collection → Raw Data Storage → Preprocessing Pipeline
→ Training Pipeline → Model Validation → Model Registry
→ Model Serving API → Human/Bot Detection
```

이 구조를 기준으로 각 CAPTCHA 유형별 세부 전처리와 모델만 다르게 적용하는 방향으로 정리했다.

---

## 자동 재학습 흐름 설계

```
Monitoring → Trigger → Automated Retraining → Model Serving
```

서비스 운영 중 탐지 성능이 떨어지거나 새로운 공격 패턴이 발견되면, 이를 트리거로 삼아 재학습을 수행하는 구조다.

---

## 데이터 저장 전략 정리

장기 보관이 필요한 전체 데이터는 Google Drive에 저장하고, GPU 서버에는 최신 학습 데이터와 대표 샘플 데이터만 유지하는 방향으로 결정했다. GPU 서버의 저장 공간 부담을 줄이면서도 필요한 학습 데이터는 빠르게 접근할 수 있다.

---

## 손전등 CAPTCHA 기준 프로젝트 구조 재설계

파이프라인 단계별 폴더:
- validation
- preprocessing
- training
- evaluation
- inference
- configs

데이터 폴더(역할별 분리):
- raw
- processed
- generated
- labels
- mouse_logs

**preprocessing 코드 역할 정리**
- `extract_features`
- `extract_dynamic_features`
- `merge_sampled_data`
- `split_dataset`

**GRU 기반 행동 분석 모델 구조**
- common
- data
- models
- training
- evaluation
- inference

이를 통해 단순 실험 코드가 아니라 실제 ML 프로젝트 구조에 가까운 형태로 리팩터링하는 방향을 잡았다.

---

## 이번 주 전체 회고

이번 주는 AGAMI 프로젝트가 단순 CAPTCHA 구현을 넘어서, 실제 공격 검증과 MLOps 구조 설계 단계로 확장된 주였다.

특히 중요한 변화는 Playwright 기반 매크로 테스트를 통해 "봇 입장에서 실제 CAPTCHA를 어떻게 공격할 수 있는지"를 직접 검증했다는 점이다. 이를 통해 현재 CAPTCHA 구조가 어떤 부분에서 안전하고, 어떤 부분에서 테스트용 노출이나 좌표 처리에 주의해야 하는지 확인할 수 있었다.

또한 손전등 CAPTCHA를 기준으로 ML 파이프라인과 프로젝트 구조를 다시 정리하면서, 앞으로 모델 학습과 추론 API 연결을 더 체계적으로 진행할 수 있는 기반을 마련했다.

**이번 주 핵심 성과**
- 실시간 행동 기반 CAPTCHA 구조 설계
- 얼굴/손 landmark 기반 feature 정의
- Playwright 기반 매크로 봇 테스트 환경 구축
- known-target bot 동작 검증
- 좌표 정규화 문제 해결
- 대시보드 구조 및 API 명세서 작성
- AGAMI 공통 MLOps 파이프라인 구조 설계
- 손전등 CAPTCHA 기준 프로젝트 구조 재정리
- GRU 기반 행동 분석 모델 구조 설계

결과적으로 이번 주는 "CAPTCHA를 만드는 단계"에서 "CAPTCHA를 공격해보고, 그 결과를 바탕으로 운영 가능한 ML 시스템으로 정리하는 단계"로 넘어간 시기였다고 볼 수 있다.
