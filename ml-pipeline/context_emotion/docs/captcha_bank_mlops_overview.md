# 감정추론 CAPTCHA MLOps 흐름 — 한 장짜리 개요

> 마지막 업데이트: 2026-06-26

---

## 전체 흐름 (두 개의 독립 파이프라인)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE A — EfficientNet-B0 감정 분류 모델 업데이트 (context_emotion/)     │
│                                                                             │
│  학습 데이터       모델 학습            MLOps                               │
│  (EMOTIC + 수동)  ──→ train_model.py ──→ package → compare → promote       │
│                        (PyTorch/ONNX)      model-store/context_emotion/     │
│                                                     ↓                      │
│                                           FastAPI /predict 서비스 (port 8082) │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE B — captcha_bank: CAPTCHA 풀 보안 평가 + 어태커 프록시 모델       │
│              (context_emotion/captcha_bank/)  ← 이 문서의 주제              │
│                                                                             │
│  ① 이미지 수집·전처리                                                      │
│     EMOTIC / 수동 수집 이미지 → preprocessing/ → emotic 데이터셋           │
│                                                                             │
│  ② VLM 공격 실행  (build_review_queue.py)                                  │
│     Qwen2.5-VL-3B, SmolVLM2-2.2B, Qwen2-VL-2B가 각 이미지에              │
│     감정 레이블을 예측 → 공격 성공/실패 기록 → JSONL 저장                  │
│                                                                             │
│  ③ 인간 검수  (human_review_server.py, port 8765)                          │
│     검수자가 VLM 예측 검토 → final_emotion 확정 → 검수 완료 표시           │
│                                                                             │
│  ④ 풀 내보내기  (export_captcha_pool.py)                                   │
│     approved 문항만 추출 → captcha_pool.csv 생성                           │
│                                                                             │
│  ⑤ MLOps 파이프라인 (8단계, Argo Workflows 또는 run_pipeline.py)           │
│                                                                             │
│   STEP 1: validate-captcha-pool  풀 CSV 구조 · 클래스 분포 검증            │
│        ↓                  ↓                                                │
│   STEP 2: train-model     STEP 4: build-choice-policy-report               │
│      (sklearn RF)           (4지선다 정책 분석 · choice_policy_report.md)  │
│        ↓                  ↓                                                │
│   STEP 3: evaluate-model ─→ (완료 대기) ─────────────────────────────────  │
│      attacker_pass_rate, robust_rate, 3-Q pass rate 계산                   │
│                                       ↓                                    │
│                              STEP 5: package-model                         │
│                                model.joblib + metadata.json                │
│                                + evaluation_result.json                    │
│                                + captcha_pool.csv                          │
│                                + choice_policy_report.md                   │
│                                → candidates/{version}/                     │
│                                       ↓                                    │
│                              STEP 6: compare-model                         │
│                                게이트 7개 확인                              │
│                                (pass_rate ≤ 0.35, pool_size ≥ 200 등)     │
│                                       ↓ (통과 시)                         │
│                              STEP 7: promote-model                         │
│                                candidates/{version}/ → current/            │
│                                이전 current/ → archive/                    │
│                                       ↓                                    │
│                              STEP 8: smoke-test                            │
│                                model.joblib 로드 · 5개 샘플 추론 확인      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## model-store/captcha_bank/ 구조

```
model-store/captcha_bank/
├── current/                  ← 운영 중인 어태커 프록시 모델
│   ├── model.joblib          sklearn (emotion_attacker + security_ranker)
│   ├── metadata.json         버전, 학습 날짜, 풀 크기, promoted_at
│   ├── evaluation_result.json 보안 지표 전체
│   ├── captcha_pool.csv      이 버전의 CAPTCHA 풀
│   └── choice_policy_report.md 4지선다 정책 분석
├── candidates/
│   └── {version}/            패키징 완료 후 승격 대기
└── archive/
    └── {timestamp}_{version}/ 교체된 이전 버전 백업
```

---

## 핵심 보안 지표 (evaluation_result.json)

| 지표 | 의미 | 기준값 (gate) |
|------|------|--------------|
| `attacker_pass_rate` | sklearn 어태커의 단일 문항 정답률 | ≤ 0.35 |
| `robust_rate` | 1 - attacker_pass_rate | ≥ 0.65 |
| `choice_policy_pass_rate` | 어태커의 3문제 챌린지 통과율 | ≤ 0.10 |
| `ambiguous_rate` | 모호 문항 비율 | ≤ 0.20 |
| `pool_size` | 풀 전체 문항 수 | ≥ 200 |
| `macro_f1_attacker` | 어태커 모델의 macro F1 | (참고용) |

`choice_policy_pass_rate`는 `pass_score=2.5` 기준 3문제 이진 점수 합산 통과율.  
sklearn attacker_pass_rate=0.30 → 3-Q pass rate ≈ 0.30³ = 2.7%

---

## 승격 정책 수정

`captcha_bank/config/promotion_policy.yaml` 에서 임계값을 조정하면  
다음 파이프라인 실행 시 새 정책이 자동 적용된다.

---

## 주요 파일 위치

| 역할 | 경로 |
|------|------|
| VLM 공격 실행 | `captcha_bank/build_review_queue.py` |
| 인간 검수 UI | `captcha_bank/human_review_server.py` |
| 풀 내보내기 | `captcha_bank/export_captcha_pool.py` |
| 선택지 생성 | `captcha_bank/choice_generation.py` |
| 피처 추출 | `captcha_bank/training/features.py` |
| 모델 학습 | `captcha_bank/training/train_attack_model.py` |
| 평가 | `captcha_bank/evaluation/run_attack_eval.py` |
| 파이프라인 오케스트레이터 | `captcha_bank/scripts/run_pipeline.py` |
| Argo YAML | `k8s/argo-workflows/captcha-bank-pipeline.yaml` |
| 승격 정책 | `captcha_bank/config/promotion_policy.yaml` |
| 운영 가이드 | `captcha_bank/CAPTCHA_BANK_OPS.md` |

---

## TODO (미구현 항목)

- [ ] **VLM 어태커 통계**: `metrics.py`의 `vlm_attacker_stats` 계산은 풀 CSV에  
      `qwen_emotion`, `smolvlm_emotion`, `self_attack_emotion` 열이 있을 때만 활성화  
      → `build_review_queue.py` 실행 후 생성되는 열에 의존
- [ ] **PIPELINE A 연동**: 감정 분류 ONNX 모델 업데이트 후 CAPTCHA 풀 재평가 자동 트리거
- [ ] **human_review_server 자동 종료**: 검수 완료 후 `export_captcha_pool.py` 자동 실행
- [ ] **captcha_bank-workflow-sa 적용**: RBAC YAML 클러스터 실제 적용 후 SA 토큰 확인
