# context_emotion MLOps 운영 설계

이 문서는 **실제 emotion classifier 모델이 아직 없는 상태**에서 만들어졌다.
모델팀이 무엇을 만들든 (백본이 EfficientNet이든 ViT든, freeze 전략이
무엇이든) 아래 흐름과 contract만 지키면 운영 파이프라인에 그대로
꽂힌다는 게 이 문서와 스캐폴딩 코드의 목적이다. 숫자(threshold)는 모델이
한 번도 안 나온 시점에서 정하면 근거 없는 임의값이 되므로, 전부 TODO로
남겨둔다 (8장).

참고 패턴: `flashlight/scripts/{package_for_captcha_engine,
compare_candidate, promote_model, run_model_update_pipeline}.py`. 이
설계는 그 구조를 기본으로 가져오되, emotion CAPTCHA 고유의 신호(사람
애매함, 공격 모델 정답률, 클래스별 회귀)를 게이트에 추가한다.

---

## 1. 전체 운영 흐름

```
[모델팀: 학습 (이 문서 범위 밖)]
        |
        v  checkpoint(.pt) + ONNX export
[evaluate_candidate.py]            <- 학습된 후보를 held-out eval csv로 실제 추론/채점
        |  evaluation_result.json
        v
[package_emotion_model.py]         <- 6개 파일을 candidates/{version}/로 패키징 + contract 검증
        |  검증 실패 시 후보 자체가 생성되지 않음
        v  candidates/{version}/
[compare_candidate.py]             <- current(있다면)와 evaluation_result.json 비교
        |  promotion_gate.py가 5개 게이트 판정 -> promotion_decision.json
        v
   final_decision?
     |-- reject         -> 중단, 이유는 promotion_decision.json에 기록됨
     |-- manual_review   -> 중단, 사람이 promotion_decision.json 보고 판단
     '-- promote   (※ required=false 게이트가 not_configured/fail이어도 promote될 수 있음 -
                       이 경우 promotion_decision.json.warnings에 어떤 게이트가
                       검증 안 됐는지 항상 남는다. attacker_proxy_resistance가
                       대표적 - 지금은 required=false라 매번 promote에 끼어 있을 것)
              v
       [promote_model.py]         <- current를 archive/{timestamp}_{old_version}/로 백업
              |                       후 candidates/{version}/을 staging -> current로 원자적 교체
              v  current/
       [smoke_test_model.py]      <- 승격 직후, 실제로 ONNX가 로딩/추론되는지 헬스체크
              v
       [ops_metrics 기록]          <- "이 버전이 이 날짜에 배포됨" 마커 (운영 노출 집계는 서빙 측이 채움)
```

위 전체를 한 번에 실행하는 오케스트레이터가
`scripts/run_model_update_pipeline.py`다. `compare_candidate.py`가
`promote`가 아니면 그 다음 단계(promote/smoke_test/record)는 전부
건너뛴다 - flashlight의 "compare FAIL이면 promote 안 함" 원칙을 5개
게이트로 확장한 것.

문제가 생기면 `deployment/rollback_model.py`로 가장 최근(또는 지정한)
archive를 current로 되돌린다. flashlight에는 없던 부분 - 운영 중
사고가 나면 사람이 기다릴 시간이 없다.

---

## 2. 단계별 입력/출력/artifact 계약

| 단계 | 입력 | 출력 | 실패 시 |
|---|---|---|---|
| `evaluate_candidate.py` | onnx, label_schema.json, preprocessing_config.json, eval csv(+split), image_root, version | `evaluation_result.json` | 의존성/파일 없으면 즉시 에러 종료 (가짜 숫자 출력 안 함) |
| `package_emotion_model.py` | onnx, metadata.json, label_schema.json, preprocessing_config.json, evaluation_result.json, version | `candidates/{version}/` (6개 파일 + manifest.json) | contract 위반 시 **candidates/{version}/이 생성되지 않음** (임시 staging에서 검증 후에만 최종 이동) |
| `compare_candidate.py` | `current/evaluation_result.json`(있다면), `candidates/{version}/evaluation_result.json` | `candidates/{version}/promotion_decision.json` | current가 없으면(첫 모델) candidate 단독으로만 평가, 에러 아님 |
| `promote_model.py` | `--version` | `current/` 교체 + `archive/{timestamp}_{old_version}/` | 검증/스테이징/교체 중 어디서 실패해도 **current는 교체 전 상태 유지** |
| `smoke_test_model.py` | `--dir` (보통 `current/`) | 콘솔 출력 (pass/fail) | onnxruntime 없거나 출력에 NaN/차원 불일치 시 실패로 종료 |
| `rollback_model.py` | `--to-archive` 또는 최신 archive | `current/` 복원 + rollback 이전 current도 다시 archive에 백업 | archive가 비어 있으면 에러 |

---

## 3. 모델팀이 최종적으로 제공해야 하는 산출물 계약

`package_emotion_model.py --onnx --metadata --label-schema
--preprocessing-config --evaluation-result`로 넘기는 5종 (manifest.json은
패키징 스크립트가 자동 생성하므로 모델팀이 만들 필요 없음):

1. **checkpoint** (`.pt` 등) - model-store에는 들어가지 않는다.
   `metadata.json.checkpoint_source`에 경로만 적어 추적성만 남김
   (flashlight도 동일 - normalizer/onnx만 패키징, .pt는 안 둠).
2. **ONNX** (`model.onnx`) - `config/runtime_contract.yaml`의
   `onnx_contract` (입력명 `image`, 출력명 `logits`)를 따를 것.
3. **label schema** (`label_schema.json`) - `contracts/label_schema_contract.md`.
   `common/constants.py`의 `EMOTION_CLASSES`를 순서까지 그대로 복사.
4. **preprocessing config** (`preprocessing_config.json`) - 최소
   `image_size`, `normalize_mean`, `normalize_std`,
   `crop_to_bbox_for_emotic` 4개 키. 추론 시 이미지를 어떻게 전처리해야
   하는지가 모델 코드가 아니라 이 파일에 선언되어 있어야, 운영 쪽
   추론기가 모델 내부 구현을 몰라도 정확히 같은 전처리를 재현할 수 있다.
5. **metadata** (`metadata.json`) - `contracts/model_metadata.schema.json`.
   특히 `version`, `label_schema_version`, `input_spec`/`output_spec`은
   다른 모든 단계가 그대로 읽는 필드라 빠지면 패키징이 막힌다.
6. **evaluation result** (`evaluation_result.json`) - `evaluate_candidate.py`
   실행 결과 그대로. 직접 만들어서 넘기지 말 것 - 숫자가 실제 추론에서
   나온 게 아니면 compare 단계가 의미 없어진다.

### evaluation_result.json을 그냥 믿지 않는다

evaluation_result.json은 `--onnx`/`--evaluation-result`로 **따로따로**
넘기는 두 파일이라, 둘이 실제로 같은 모델을 가리키는지는 아무도 보장해
주지 않는다 - 예전 모델의 evaluation_result.json을 새 onnx와 같이
넘기는 실수가 가능했다. 그래서 `package_emotion_model.py` /
`promote_model.py` / `compare_candidate.py`는 매번
(`deployment/model_store.py`를 통해) 두 가지를 같이 검증한다:

- **`onnx_hash_consistency`**: `evaluate_candidate.py`가 평가에 쓴 정확히
  그 onnx 파일의 sha256(`evaluation_result.json.onnx_sha256`)이 지금
  패키징/승격하려는 `model.onnx`의 sha256과 같은지.
- **`version_consistency`**: `metadata.json.version` /
  `evaluation_result.json.version`이 candidates 디렉터리 이름
  (`--version`)과 모두 일치하는지.

둘 중 하나라도 안 맞으면 후보 생성/승격 자체가 막힌다 (`tests/test_package_contract.py`의
`test_onnx_hash_mismatch_blocks_candidate_creation`,
`test_version_mismatch_blocks_candidate_creation` 참고).

`scripts/validate_model_artifacts.py --dir <폴더> [--expected-version <버전>]`로
패키징 전에 미리 자가 점검 가능 (`--expected-version`을 주면 위 두
검증까지 같이 돈다).

---

## 4. model-store 디렉터리 구조

```
ml-pipeline/model-store/context_emotion/
├── candidates/
│   └── {version}/
│       ├── model.onnx
│       ├── metadata.json
│       ├── label_schema.json
│       ├── preprocessing_config.json
│       ├── evaluation_result.json
│       ├── manifest.json              # package_emotion_model.py가 생성 (sha256 + source 경로)
│       └── promotion_decision.json    # compare_candidate.py가 생성
├── current/                            # 운영 중인 단 하나의 버전. package로 직접 못 건드림
│   └── (candidates/{version}/과 동일한 6개 파일 + promoted_at, promoted_from_candidate가 추가된 metadata.json)
└── archive/
    └── {timestamp}_{old_version}/      # promote 직전 current의 스냅샷. rollback의 복원 대상
```

- **candidates/**: 후보 보관소. 패키징 시 contract를 통과한 것만 존재한다
  (실패하면 디렉터리 자체가 안 생김). 승격 안 된 후보도 계속 남아있음 -
  지우는 자동화는 의도적으로 만들지 않았다 (어떤 후보를 지워도 되는지는
  운영 정책 문제, 8장 TODO).
- **current/**: 서빙이 읽는 단 하나의 경로. `promote_model.py`의 원자적
  교체로만 바뀐다.
- **archive/**: 매 승격 + 매 rollback 직전에 자동으로 쌓인다. 보존 기간/
  삭제 정책 없음 (8장 TODO).

### 원자적 교체 + 롤백 방식

`os.replace`는 디렉터리를 원자적으로 바꿔주지 않으므로 (flashlight와
동일한 이유로) 2단계 rename을 쓴다:

1. `current/` -> `_current_old`로 rename (현재 버전을 임시로 빼둠)
2. `current_staging_{ts}/` (candidate 복사본) -> `current/`로 rename
3. 2번이 실패하면 `_current_old`를 다시 `current/`로 rename해서 즉시 복원
4. 2번이 성공하면 `_current_old` 삭제

이 절차는 `promote_model.py`와 `rollback_model.py`가 거의 동일하게
공유한다 (rollback은 "staging의 출처가 archive"라는 점만 다름). 두
스크립트 모두 **교체 전에 먼저 백업을 만든다** - 즉 rollback도 흔적을
남기므로, rollback한 걸 또 되돌릴 수 있다.

---

## 5. 학습 데이터 vs 운영 평가 데이터

이건 [이전 대화]에서 나온 핵심 우려(공격모델이 풀어내는 이미지를 계속
빼면 데이터가 고갈된다)를 운영 구조에 반영한 부분이다. **두 데이터는
절대 같은 파일/같은 정리 정책을 공유하지 않는다.**

| | 학습 데이터 | 운영 평가 데이터 |
|---|---|---|
| 위치 | `/workspace/data/context_emotion/processed/context_emotion_train_dataset_v2.csv` | `ops_metrics/events.jsonl` (스키마는 `ops_metrics/schema.md`) + (TODO) 별도 공격 평가 풀 |
| 만드는 주체 | `preprocessing/build_train_dataset_v2.py` (정적, 배치성) | 운영 중 실시간/배치로 계속 쌓임 |
| 정리 정책 | `build_train_dataset_v2.py`를 다시 돌릴 때만 바뀜 - 어태커가 맞혔다고 행을 빼지 않음 | 노출 빈도/애매함 비율 등으로 "운영 풀" 노출 가중치를 조절 (이 스캐폴딩 범위 밖 - 서빙 쪽 구현 필요) |
| 누가 읽나 | `training/*`, `evaluate_candidate.py`의 `--eval-csv` | `evaluate_candidate.py`의 `human_ambiguity` 채우기, `promotion_gate.py`의 사람 애매함/공격 프록시 게이트 |

`evaluate_candidate.py`는 학습 csv의 `test` split으로 **legitimate-user
accuracy**(사람 입장에서 정답률)만 본다. **공격 저항력은 절대 같은
지표가 아니다** - 그건 `attacker_proxy_eval.py`가 별도 평가 풀로 본다
(아직 미구현, 6장).

---

## 6. 운영 데이터에서 추적할 지표

`ops_metrics/schema.md`에 전체 스키마. 요약:

- 노출 수, 사람 정답률, 사람 애매함/이의제기 비율
- 클래스별 선택 분포
- 공격 프록시 정답률, 공격 프록시 오답 유형
- 제외된 문항 비율, 재검수 필요 비율

기록 함수는 `ops_metrics/recorder.py:record_daily_metrics()`. **이걸
호출하는 실제 운영(서빙) 코드는 아직 없다** - 이 스캐폴딩은 스키마와
기록/조회 함수만 만들었다. `run_model_update_pipeline.py`는 승격 성공
직후 `exposures=0`짜리 "배포됨" 마커 한 줄만 남긴다 (실제 운영 지표
아님 - 그냥 "이 버전이 이 날짜부터 떠 있었다"는 사실 기록).

---

## 7. 재학습 트리거 원칙: 신호 기반, 캘린더 기반 아님

`config/promotion_policy.yaml`의 `retrain_trigger.signal_based: true`로
명시. **이 레포의 어떤 스크립트도 정기 cron으로 재학습을 트리거하지
않는다** - 트리거 후보 신호는:

- 공격 프록시 정답률이 임계치 이상 상승 (방어력 저하)
- 사람 애매함 비율이 임계치 이상 상승 (라벨/문항 품질 저하)
- 특정 클래스 노출/정답 편향이 임계치 이상 증가
- 운영 풀에서 제외되는 문항 비율이 너무 빠르게 증가 (풀 고갈 신호)

이 신호들을 자동으로 보고 재학습을 **시작하는 자동화는 만들지
않았다** - `ops_metrics`를 보는 사람/알림이 신호로 참고해서 재학습을
"검토"하는 단계까지만이고, 실제 재학습 실행은 사람이 트리거한다. 임계치
숫자는 전부 8장 TODO.

---

## 8. 아직 결정 안 된 것 (TODO)

전부 `config/promotion_policy.yaml`에 `null`로 들어가 있다. 코드는
`null`인 게이트를 `not_configured`로 처리하고, `not_configured`인
**required 게이트**가 하나라도 있으면 최종 판정은 절대 `promote`가 될
수 없다(`manual_review`로 막힘) - 그러니 아래가 비어 있어도 사고로
이어지진 않지만, 모델팀 첫 candidate가 나온 후에는 사람이 직접 정해야
승격이 가능해진다.

- [ ] `overall_quality_gate`: `min_macro_f1` / `min_accuracy` / `max_macro_f1_drop_vs_current` 숫자
- [ ] `class_regression_gate.critical_classes`: 지금은 label_distribution_v2.md의
      low_resource 3종(`embarrassment`, `aversion`, `sadness`) + `confusion`
      임시 지정 - 운영팀이 "운영상 중요한" 클래스 목록을 별도로 확정해야 함
- [ ] `class_regression_gate`: `max_f1_drop_absolute` vs `max_f1_drop_relative_pct` 중 뭘 쓸지, 숫자
- [ ] `human_ambiguity_gate`: "애매함" 선택지 UX 자체가 아직 없음 - 나오면
      `required: true`로 전환 + `min_exposures_required`/`max_ambiguous_rate` 숫자
- [ ] `attacker_proxy_gate`: 어태커 프록시 모델 자체가 미선정, 별도
      adversarial eval pool도 미구축 - 나오면 `required: true`로 전환 +
      `max_attacker_solve_rate`/`max_attacker_solve_rate_increase` 숫자
- [ ] `retrain_trigger`의 4개 threshold 전부
- [ ] candidates/ 보존 정책 (오래된/승격 안 된 후보를 언제 지울지) - 지금은 무기한 보관
- [ ] archive/ 보존 정책 (디스크 무한정 쌓임) - 지금은 무기한 보관
- [ ] `Dockerfile.training`이 아직 `flashlight/`만 COPY함 - context_emotion을
      실제 학습/평가 컨테이너에 넣으려면 별도 작업 필요 (이 스캐폴딩 범위 밖)

---

## 9. 모델팀 핸드오프 체크리스트

모델팀이 결과물을 넘긴 뒤, 다음 순서로 실행하면 끝까지 연결된다
(번호는 1장 다이어그램과 동일):

1. `python -m context_emotion.scripts.validate_model_artifacts --dir <onnx/metadata/label_schema/preprocessing_config가 모인 폴더>`
   로 미리 자가 점검 (이 단계에서 evaluation_result.json/manifest.json은
   아직 없어도 됨 - required_files 항목만 FAIL로 나올 것, 나머지
   metadata/label_schema/preprocessing_config 검증은 그대로 통과해야 함)
2. `python -m context_emotion.evaluation.evaluate_candidate --onnx ... --label-schema ... --preprocessing-config ... --eval-csv ... --image-root ... --version ... --out evaluation_result.json`
3. `python -m context_emotion.scripts.package_emotion_model --onnx ... --metadata ... --label-schema ... --preprocessing-config ... --evaluation-result evaluation_result.json --version ...`
4. `python -m context_emotion.evaluation.compare_candidate --version ...` 로 `promotion_decision.json` 확인
5. `final_decision == "promote"`면 `python -m context_emotion.deployment.promote_model --version ...`
6. `python -m context_emotion.scripts.smoke_test_model --dir model-store/context_emotion/current`

또는 2~6번을 한 번에: `python -m context_emotion.scripts.run_model_update_pipeline --version ... --onnx ... --metadata ... --label-schema ... --preprocessing-config ... --eval-csv ... --image-root ...`

문제가 생기면: `python -m context_emotion.deployment.rollback_model --list` 로
archive 확인 후 `--to-latest-archive` 또는 `--to-archive <name>`.
