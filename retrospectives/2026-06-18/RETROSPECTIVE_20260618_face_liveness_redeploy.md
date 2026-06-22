# 회고록 — face_liveness 모델 재배포 (gru_v3_aspectfix_w8)

**세션 기간**: 2026-06-17 ~ 2026-06-18
**배포 시각(promoted_at)**: 2026-06-18 00:38:23
**작성**: Claude (배포 작업 회고)

---

## 1. 타임라인

| 시각 | 단계 | 내용 |
|---|---|---|
| 2026-06-16 07:45 | (기존 운영) | 이전 버전이 current/에서 운영 중 (이후 archive로 백업됨) |
| 2026-06-17 06:05 | 재학습 1차 | gru_h32_lr0005_aspectfix — 종횡비 보정 반영 첫 재학습 |
| 2026-06-17 06:12 | 재학습 2차 | gru_h32_lr0005_v3_aspectfix — v3(시간정규화) 데이터셋 적용 |
| 2026-06-17 06:17~06:19 | source_weight 스윕 | train_source_weighted.py로 R_live_clip 가중치 3/5/8/10 비교 (gru_v3_aspectfix_w3/w5/w8/w10) |
| 2026-06-17 (분석) | replay/R_live 분포 분석 | analyze_replay_rlive_overlap.py로 w8 후보의 spoof_score 분포 검증 |
| 2026-06-18 00:37:23 | export + 패키징 | gru_v3_aspectfix_w8 체크포인트 → ONNX 변환, candidates/v3_w8_aspectfix_20260618/ 패키징 |
| 2026-06-18 00:38:23 | promote | promote_model.py로 current/ 원자적 교체, 기존 버전은 archive/20260618_003823_unknown/로 백업 |
| 2026-06-18 06:12~06:16 | 인계 문서화 | deployment_handoff_v3_w8/ 패키지 작성 (README, 코드 사본, K8s 매니페스트 포함) |

---

## 2. 장애/문제 정의

**증상**: 운영 중이던 모델("v1, threshold=0.21"로 알려졌던 버전)에서
**R_live_clip(실제 사용자 클립) FRR이 95%**까지 치솟음 — 즉 정상 사용자
대부분이 스푸핑으로 오탐되어 차단됨.

**근본 원인**: feature extraction 단계의 **종횡비(aspect ratio) 처리
버그**. aspect_corrected_landmarks()가 누락/오적용되어 비정방형 비디오
프레임에서 landmark 좌표가 왜곡됨 → 정상 사용자의 얼굴 움직임 패턴이
비정상으로 학습/추론됨.

---

## 3. 원인 분석 → 조치 과정

1. **버그 위치 특정**: preprocessing/extract_image_features.py,
   extract_video_features.py, extract_features_time_norm.py에
   aspect_corrected_landmarks(landmarks, width, height) 추가 —
   lm.y / aspect_ratio로 y좌표 보정.
2. **데이터셋 재생성**: 보정된 추출 로직으로 v3 데이터셋
   (dataset_v3_time_norm.npz, 1013 clips) 재생성. 기존 v2 대비
   frame_interval 기반 시간정규화(velocity/displacement를
   frame_interval로 나눔)도 함께 적용.
3. **재학습 반복**: 단순 aspectfix만 적용한 1차 버전
   (gru_h32_lr0005_aspectfix) → 시간정규화까지 반영한 v3 버전
   (gru_h32_lr0005_v3_aspectfix) → R_live_clip 클래스가 여전히 적어
   (93개) 과소 학습되는 문제 확인.
4. **source_weight 스윕**: R_live_clip 샘플에 가중치 3/5/10을 줘보고
   비교한 결과, **w8**이 R_live FRR과 공격 탐지율(recall) 사이에서
   가장 균형 잡힌 후보로 선정 (analyze_replay_rlive_overlap.py로
   spoof_score 분포 시각 검증).
5. **threshold 재산정**: 학습 스크립트가 자동 산출하는 best_f1
   threshold(0.18)는 F1만 최적화 → R_live FRR이 71%까지 치솟는
   부작용 확인. **운영 threshold를 0.65로 수동 override**해서 R_live
   FRR을 14.29%(test)/8.33%(valid)까지 낮춤. 단, 이 결정으로 공격
   탐지 recall은 58.7%로 낮아지고 replay 공격 절반가량은 통과하게
   됨 — "사람을 덜 차단하는 쪽"을 의도적으로 선택.
6. **검증 후 승격**: ONNX 변환 시 export 전후 score 일치(diff=0.0)
   확인 → promote_model.py가 필수 파일 체크, ONNX 더미 추론 검증,
   기존 current/ 백업, 원자적 교체(os.rename, 실패 시 자동 롤백)
   순으로 안전하게 승격.

---

## 4. 의사결정 기록 (Why)

| 결정 | 이유 |
|---|---|
| threshold 0.18 → 0.65 수동 override | best_f1는 F1만 보고 실사용자 차단(FRR)을 직접 최적화하지 않음. 운영 정책상 "정상 사용자 차단 최소화"가 "공격 탐지율"보다 우선 |
| source_weight=8 선택 (3/5/10 대비) | w8이 R_live FRR/공격 recall 트레이드오프에서 가장 균형적. w10 이상은 과적합으로 다른 source_group 성능 저하 우려 (정확한 수치는 runs/gru_v3_aspectfix_w10 결과와 비교 필요 — 본 회고록 작성 시점에는 README/메타데이터에 수치가 남아있지 않아 추정) |
| ONNX 더미 검증을 promote 단계에 포함 | 변환 과정에서 발생할 수 있는 silent breakage(예: 입력 shape mismatch)를 배포 직전에 한 번 더 차단 |
| archive 백업 후 원자적 rename | 승격 실패 시 current/가 깨진 상태로 남는 것을 방지, 즉시 롤백 가능하게 설계 |

---

## 5. 좋았던 점 (Went Well)

- **재현 가능한 파이프라인**: run_model_update_pipeline.py가 train →
  export → promote를 단일 커맨드로 묶어서, 다음 재학습 시에도 같은
  절차를 그대로 반복할 수 있음.
- **안전한 승격 메커니즘 (설계상)**: dry-run 지원, 원자적 교체, 자동
  백업/롤백 로직이 모두 코드에 갖춰져 있음.
- **학습-서빙 parity 검증**: 오프라인 전처리(extract_features_time_norm.py)와
  실시간 feature extractor(code/face_feature_extractor.py) 출력이 20개
  피처 전부 1e-6 수준으로 일치함을 실제 R_live 클립으로 확인 — 배포 후
  "학습 때와 다른 입력이 들어와서 성능이 다르다"는 흔한 실패 유형을
  사전에 차단.
- **인계 문서화**: deployment_handoff_v3_w8/README.md에 "이전에 v1로
  알던 버전이 아니다"라는 경고를 명시해, 다음 사람이 구버전 정보를
  그대로 신뢰하는 실수를 방지.

---

## 6. 아쉬운 점 / 리스크 (Could Be Better)

- **🔴 실제 백업이 비어 있음 — 롤백 불가능한 상태**:
  archive/20260618_003823_unknown/을 확인한 결과 **파일이 0개**다.
  _read_current_version()이 old_version="unknown"을 반환한 것으로
  보아, promote 직전 current/에 metadata.json조차 없었던 것으로
  보인다. model-store 전체를 검색해도 "v1, threshold=0.21"로 알려졌던
  이전 운영 모델의 실제 파일이 어디에도 남아있지 않다 — 이번 promote가
  model-store에 대한 **사실상 첫 실제 배포**였고, "v1"은 파일로 존재한
  적 없는 구전 정보였을 가능성이 있다. 즉 gru_v3_aspectfix_w8에 문제가
  생겨도 **돌아갈 이전 버전이 실제로 없다.** promote_model.py의
  백업/롤백 로직 자체는 정상 동작했지만(빈 디렉토리를 그대로 백업했을
  뿐), 애초에 백업할 실체가 없었던 셈이다.
- **R_live_clip 샘플 수가 절대적으로 적음(93개)**: source_weight로
  가중치를 줘서 우회했지만, 근본적으로는 실사용자 데이터 자체를 더
  모아야 함. 가중치 기반 보정은 분포 자체를 늘리는 것이 아니라
  과적합 위험을 동반.
- **threshold 0.65 override는 수동 결정 — 재현성 약함**: 코드상으로
  "왜 0.65인지"가 사람이 README를 읽어야만 알 수 있고, 자동
  파이프라인(run_model_update_pipeline.py)에는 이 override 로직이
  없음. 다음 재학습 시 동일한 운영 철학(FRR 우선)을 자동으로 반영할
  장치가 없어, 또 누군가 best_f1 threshold를 그대로 승격시킬 위험이
  있음.
- **archive 디렉토리 명명이 unknown**: archive/20260618_003823_unknown/에서
  old_version이 unknown으로 기록됨 — 직전 배포본의 metadata.json에
  version 필드가 없었던 것으로 보임. 롤백 시 "무엇으로 되돌리는지"
  이름만으로 알 수 없어 운영 부담 증가.
- **replay 공격 절반 통과는 의도된 트레이드오프지만 모니터링 미흡**:
  FRR을 낮추는 대신 recall을 낮춘 결정이 운영 중 실제로 악용되는지
  (replay 공격 증가) 추적할 대시보드/알람이 현재 코드베이스에는
  보이지 않음.
- **w3/w5/w10 비교 수치가 보존되지 않음**: 4개 후보를 학습했지만 최종
  인계 문서에는 w8의 결과만 남고 나머지 비교표가 어디에도 정리되어
  있지 않아, "왜 w8인지"를 데이터로 재확인하기 어려움.

---

## 7. 다음 액션 제안

1. **(긴급) 현재 current/의 gru_v3_aspectfix_w8을 별도 위치에 즉시
   안전 백업**할 것 — 지금 망가지면 되돌릴 버전이 없다.
2. run_model_update_pipeline.py에 --operational-threshold 같은 옵션을
   추가해, "best_f1 자동값을 그대로 promote하지 않는다"는 운영 철학을
   코드에 강제하기.
3. R_live_clip 실사용자 데이터를 추가 수집해 source_weight 의존도를
   낮추기.
4. promote 시 metadata.json에 version 필드 필수화(현재 unknown으로
   남는 케이스 방지) — promote_model.py의 _validate_candidate에 체크
   추가.
5. replay 공격 통과율을 추적할 운영 모니터링/알람 추가 검토.
6. w3/w5/w10 비교 결과를 model/retrain_aspectfix/에 요약 문서로 남기기
   (사후 추적 가능하도록).
