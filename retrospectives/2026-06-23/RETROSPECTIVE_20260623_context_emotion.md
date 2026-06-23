# Context Emotion CAPTCHA MLOps 운영 구조 설계 및 검증 회고록

## 1. 오늘의 목표

감정추론 CAPTCHA 모델링은 다른 팀원이 진행 중이므로, 실제 모델 학습 결과가 나오기 전 단계에서 **모델을 안전하게 평가·패키징·승격·롤백할 수 있는 MLOps 운영 구조를 먼저 구축**하는 것을 목표로 했다.

기존 Flashlight CAPTCHA에서 사용 중인 `package → compare → promote` 운영 패턴을 참고하되, 감정추론 CAPTCHA의 특성인 **클래스별 성능 회귀**, **사람의 애매함**, **공격 모델의 정답률 변화**를 반영하는 별도 게이트를 설계했다.

---

## 2. 운영 구조 설계

감정추론 모델의 기본 운영 흐름을 아래와 같이 정의했다.

```text
학습 결과물
→ artifact 검증
→ candidate 패키징
→ 후보 모델 평가
→ current 모델과 비교
→ promotion gate 판정
→ promote 또는 reject/manual_review
→ 이전 current archive 보관
→ 배포 후 smoke test
→ 운영 지표 기록
```

모델 저장소는 `current`, `candidates`, `archive` 구조로 분리했다.

* `candidates/{version}`: 평가와 승격 대기 중인 후보 모델
* `current`: 현재 운영 중인 모델
* `archive`: 승격 또는 롤백 과정에서 보관되는 이전 모델
* `ops_metrics`: 학습 데이터와 분리된 실제 운영 지표 기록 영역

재학습은 정기 Cron 기반이 아니라, 운영 신호가 누적될 때 검토하는 방식으로 설계했다. 주요 신호는 공격 프록시 정답률 상승, 사람 애매함 비율 상승, 특정 클래스 편향 증가, 문제 제외 비율 증가 등이다.

---

## 3. 생성한 MLOps 구성

### 설계 및 계약 문서

* `MLOPS_OPERATION_DESIGN.md`

  * 전체 운영 흐름
  * artifact 계약
  * model-store 구조
  * 학습 데이터와 운영 데이터 분리
  * 신호 기반 재학습 원칙
  * 모델팀 산출물 연결 체크리스트
  * 향후 정책 결정이 필요한 TODO 항목 정리

* `contracts/`

  * `model_metadata.schema.json`
  * `evaluation_result.schema.json`
  * `promotion_decision.schema.json`
  * `label_schema_contract.md`

모델 metadata, 평가 결과, 승격 판정 결과에 필요한 필수 필드와 형식을 계약으로 분리했다.

### 운영 정책 및 런타임 계약

* `config/promotion_policy.yaml`

  * 전체 성능 기준
  * 중요/희소 클래스 회귀 허용 기준
  * 사람 애매함 비율
  * 공격 프록시 정답률
  * 신호 기반 재학습 트리거
  * 현재 확정되지 않은 수치는 TODO 또는 placeholder로 유지

* `config/runtime_contract.yaml`

  * 운영 모델 필수 파일
  * ONNX 입출력 이름
  * 모델 입력 shape
  * 운영 지표 저장 경로 정의

### 배포 및 모델 저장소 관리

* `deployment/model_store.py`

  * candidate/current/archive 경로 관리
  * artifact 계약 검증
  * 버전 및 해시 검증 공통 로직

* `deployment/promote_model.py`

  * candidate 모델을 current로 원자적으로 교체
  * 실패 시 기존 current 자동 복원
  * 이전 current archive 보관

* `deployment/rollback_model.py`

  * archive 모델 목록 조회
  * 특정 archive 버전을 current로 복구
  * rollback 대상의 실제 모델 버전 및 승격 시각 표시

### 평가 및 승격 게이트

* `evaluation/evaluate_candidate.py`

  * 후보 모델 평가 결과 생성
  * accuracy, precision, recall, F1 계산
  * 평가 대상 ONNX의 SHA-256 기록

* `evaluation/compare_candidate.py`

  * current와 candidate 평가 결과 비교
  * promotion decision 생성

* `evaluation/class_regression_check.py`

  * 중요/희소 클래스의 성능 하락 검사

* `evaluation/attacker_proxy_eval.py`

  * 공격 프록시 모델 기준 CAPTCHA 정답률 평가 인터페이스

* `evaluation/promotion_gate.py`

  * 전체 품질
  * 클래스 회귀
  * 사람 애매함
  * 공격 프록시 방어력
  * artifact 무결성
  * 총 5개 게이트의 결과를 `promote`, `reject`, `manual_review`로 판정

### 스크립트 및 운영 지표

* `scripts/package_emotion_model.py`

  * staging 영역에서 artifact 검증 완료 후 candidate 생성
  * 검증 실패 시 candidate 디렉터리가 생성되지 않도록 구성

* `scripts/validate_model_artifacts.py`

  * 모델팀 산출물의 파일 존재 여부, 버전 일치, 계약 충족 여부 확인

* `scripts/smoke_test_model.py`

  * 배포 직후 ONNX 로딩 및 기본 추론 확인

* `scripts/run_model_update_pipeline.py`

  * 검증부터 package, compare, promote, smoke test까지 연결하는 오케스트레이터

* `ops_metrics/recorder.py`

  * 운영 지표를 JSONL 형태로 기록·조회
  * 학습 데이터 경로와 운영 지표 경로가 동일하게 설정되는 실수 방지

---

## 4. 핵심 트러블슈팅 및 보강

초기 스캐폴딩 검토 과정에서 실제 운영 사고로 이어질 수 있는 위험 요소를 발견하고 보강했다.

### 4-1. 평가 결과와 ONNX 파일 불일치 위험

기존에는 ONNX 파일과 `evaluation_result.json`이 별도 인자로 전달되어, 새 ONNX에 이전 모델의 좋은 평가 결과를 붙여도 통과할 수 있었다.

이를 방지하기 위해 평가 시 실제 ONNX 파일의 SHA-256을 계산해 `evaluation_result.json`에 기록하도록 수정했다. 이후 package 및 promote 단계에서도 ONNX 파일의 해시, metadata, manifest, evaluation result의 해시가 일치하는지 재검증하도록 보강했다.

이를 통해 평가 이후 candidate 파일이 변경되거나, 다른 모델의 평가 결과가 섞이는 상황을 차단할 수 있게 했다.

### 4-2. ONNX 입력 shape 하드코딩 문제

`promote_model.py`에서 ONNX 입력 shape을 `[1, 3, 224, 224]`로 고정하고 있어, 실제 모델 입력 크기가 다르거나 dynamic shape 모델일 경우 잘못된 검증이 이루어질 수 있었다.

이를 metadata 및 runtime contract 기반으로 입력 shape을 읽도록 수정했다. 실제 모델이 도착한 뒤에는 ONNX의 실제 입출력 이름과 shape에 맞춰 최종 확인이 필요하다.

### 4-3. 모델 버전 불일치 검증 누락

metadata, evaluation result, candidate 디렉터리 이름의 버전이 서로 달라도 통과할 수 있는 문제가 있었다.

artifact 검증 단계에 expected version 검사를 추가해 다음 값들이 일치하지 않으면 package 또는 promote가 실패하도록 수정했다.

* candidate 디렉터리 버전
* `metadata.json`의 version
* `evaluation_result.json`의 version
* manifest의 version

### 4-4. 승격 사유가 불명확한 문제

기존 promotion decision은 `macro_f1 >= min_macro_f1`처럼 규칙명만 남아, 실제 어떤 수치 때문에 통과 또는 실패했는지 확인하기 어려웠다.

각 게이트 결과에 실제 후보 수치, current 수치, 기준값, 차이값을 함께 기록하도록 수정했다. 이를 통해 승격 실패 시 원인을 사람이 바로 확인할 수 있게 했다.

### 4-5. 공격 프록시 미설정 상태의 가시성 부족

공격 프록시 평가가 아직 설정되지 않은 경우, 자동 승격을 막지는 않더라도 "검증되지 않음" 상태가 눈에 띄지 않을 수 있었다.

`not_configured` 상태를 warning으로 명시하고, promotion decision에 공격 프록시 평가가 수행되지 않았다는 사실이 남도록 보강했다. 향후 공격 프록시 평가가 준비되면 해당 게이트를 필수 조건으로 전환할 수 있다.

### 4-6. 파일 시스템 및 rollback 안정성 보강

다음 운영 안정성 문제도 함께 수정했다.

* archive/staging 디렉터리 타임스탬프를 마이크로초 단위로 변경하여 충돌 방지
* candidate 최종 복사 중 실패할 경우 반쪽 디렉터리 자동 정리
* rollback 목록에서 디렉터리 이름뿐 아니라 실제 모델 버전과 승격 시각 표시
* current가 없는 첫 promote, 기존 current가 있는 promote, promote/rollback 중간 실패 시 current 보호 동작 확인

---

## 5. 검증 결과

총 20개의 unittest를 통해 아래 흐름을 검증했다.

* artifact 검증 성공 및 실패
* version 불일치 차단
* package staging 검증
* candidate 생성
* current가 없는 첫 promote
* current가 존재하는 상태에서 archive 생성
* promote 실패 시 기존 current 복원
* rollback 대상 조회 및 복구
* candidate ONNX 변경 후 해시 불일치로 promote 차단
* 운영 지표 경로와 학습 데이터 경로 충돌 방지

또한 mock artifact를 사용해 다음 흐름을 직접 실행했다.

```text
package(v1)
→ promote(v1)
→ package(v2)
→ promote(v2)
→ rollback 목록 확인
→ rollback 실행
```

실제 모델이 없는 상태이므로 ONNX Runtime 기반의 실제 추론 및 smoke test는 아직 검증하지 못했다. 현재는 mock artifact와 계약 검증 중심으로 운영 안정성을 확인한 상태다.

---

## 6. 현재 상태

감정추론 모델이 아직 완성되지 않았더라도, 모델팀 결과물이 전달되면 다음 절차로 바로 연결할 수 있는 상태다.

```text
ONNX + metadata + label schema + preprocessing config
→ validate_model_artifacts
→ evaluate_candidate
→ package_emotion_model
→ compare_candidate
→ promote 또는 manual_review/reject
→ smoke_test
```

초기 운영에서는 자동 promote를 바로 활성화하지 않고, 첫 몇 개 후보 모델은 `manual_review` 중심으로 승격 결과를 검토하는 방식이 적절하다.

---

## 7. 향후 할 일

### 모델팀 산출물 수령 후

1. 실제 ONNX 파일의 입출력 이름과 입력 shape 확인
2. 실제 preprocessing config 및 label schema 연결
3. 실제 평가셋으로 `evaluation_result.json` 생성
4. 실제 ONNX Runtime smoke test 실행
5. 모델별 package → compare → promote → rollback 전체 흐름 검증

### 운영 정책 확정 필요

* 최소 macro F1 및 accuracy 기준
* 중요/희소 클래스 목록
* 클래스별 성능 하락 허용치
* 사람 애매함 비율 기준
* 공격 프록시 정답률 기준
* 공격 프록시 및 사람 애매함 게이트 활성화 시점
* candidate/archive 보관 기간 및 개수
* 신호 기반 재학습 트리거 임계치

---

## 8. 결론

오늘은 감정추론 CAPTCHA 모델을 단순히 학습·배포하는 구조가 아니라, **모델 성능뿐 아니라 CAPTCHA 방어력과 사용자 경험까지 함께 검증한 뒤에만 운영 모델을 교체하는 구조**를 설계했다.

특히 artifact 해시 검증, 버전 일치 검증, 원자적 promote/rollback, 클래스별 회귀 검사, 공격 프록시 평가 인터페이스를 추가하면서 잘못된 모델 또는 검증되지 않은 모델이 운영 환경에 반영되는 위험을 줄였다.

모델링이 완료되면 실제 ONNX와 평가 결과를 이 구조에 연결해, 후보 모델을 안전하게 검증하고 운영 모델로 승격할 수 있다.
