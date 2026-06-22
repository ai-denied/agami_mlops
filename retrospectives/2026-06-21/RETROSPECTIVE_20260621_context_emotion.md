# Context Emotion CAPTCHA 벤치마크 회고록 (Day 5)

**일자**: 2026-06-21
**담당**: agami MLOps
**선행 문서**: [RETROSPECTIVE_20260619_context_emotion.md](../2026-06-19/RETROSPECTIVE_20260619_context_emotion.md) (사람 검수 중단 → EMOTIC 원본 라벨 복원 전환, situation_class 폐지, 442건 unmapped 발생)

---

## 오늘의 흐름

1. Engagement/unmapped 검수 마무리 → 메인 풀을 1,737건으로 확정
2. **"emotic 원본 그대로 써달라"는 요청이 실제로는 지켜지지 않았던 걸
   발견 → 재복원**
3. manual 폴더명 vs 실제 이미지 매핑 무결성 검증 → 34건 누락 발견 →
   30건 정상 사유 확인 후 재투입
4. 최종 라벨 체계 확정: **감정 14종 + 상황 7종**
5. Qwen 재검증용 4지선다 MCQ 큐 생성(1,961문제)
6. 로컬 Qwen2.5-VL-3B 실행 환경 구성 시도 → 디스크 용량 한계로 모델
   다운로드 단계에서 보류

---

## 1. Engagement/unmapped 검수 마무리

어제까지 만든 engagement_review_app.py(8502)로 Engagement 과반 895건을
사람이 직접 라벨링/제외 검수했다. 진행 중 **메인 풀과의 동기화 버그**를
발견 — Engagement 앱에서 개별로 "Excluded" 누른 179건이 메인 풀
(captcha_bank_human_reviewed.csv)에서 안 빠지고 남아있었음(163건은
reviewed로, 8건은 pending으로 남아 메인 검수 화면에 다시 나타남). 전부
찾아서 excluded_pool.csv로 이동 처리.

이후 끝까지 라벨 매핑이 안 되고 결정도 안 난 **pending 99건**을 메인
풀에서 제거(완전 삭제가 아니라 excluded_pool.csv에
unmapped_never_resolved_dropped 사유로 보관)하면서 검수를 마무리.

**메인 풀 최종**: 1,737건 (reviewed 1,668 / excluded 69), pending 0.
**excluded_pool.csv 누적**: 1,144건.

---

## 2. "EMOTIC 원본 그대로"가 실제로는 압축돼 있었던 문제

2026-06-19에 restore_original_emotic_labels.py로 "원본 라벨 복원"을
했다고 기록했지만, 실제로는 원본 annotator 과반 카테고리를 다시
찾아온 뒤 **EMOTIC_TO_EMOTION 10클래스 매핑표를 거쳐 저장**하고
있었다 — 사용자가 직접 데이터를 확인하다가 발견.

재처리 기준:

- **1,606건**: 진짜 원본 EMOTIC 카테고리명(Disconnection,
  Doubt/Confusion 등 고유 표기, 멀티레이블)으로 다시 복원
- **131건**: 사람이 직접 검수한 것(unmapped 82건 + engagement 50건,
  중복 1건 제외) → 그대로 유지, 손 안 댐

---

## 3. manual 데이터 무결성 검증 — 34건 누락 발견

"manual 폴더명과 실제 이미지가 다 맞게 매핑됐는지" 확인 요청에 따라
디스크 실제 파일 수 vs manual_images_labeled.csv 행 수를 전수 대조:

- 디스크 228장 vs 라벨링 194건 → **34건 누락**
- 매핑이 "틀린" 건 0건(있는 건 다 정확) — 애초에 일부 이미지이
  파이프라인에 안 들어갔던 것

원인 추적(qwen_attack_results.csv 대조):

- **30건**: Qwen이 이미 맞혔던 케이스(correct=True) — CAPTCHA
  전제("Qwen 틀림")가 안 맞아서 attack_candidates_filtered.csv
  단계에서 정상적으로 제외된 것. 사용자가 "그때 라벨이 너무 압축돼서
  Qwen이 맞히기 쉬웠을 수 있다"고 판단해 **재투입 결정** →
  bank_tier='excluded', tier='exclude_no_longer_hard',
  label_status='pending'으로 표시해 다시 검토 가능하게 추가.
- **despair 4건**: qwen_attack_results.csv에 한 번도 없음(Qwen
  공격을 안 돌린 듯) — **아직 미해결**, Qwen 공격을 새로 돌려야 함.

이 과정에서 empathy 폴더(4건, 기존 매핑표에 없던 단어)는 임시로
calm에 넣었다가, 사용자 요청으로 **독립 라벨로 분리**.

---

## 4. 최종 라벨 체계 확정 — 감정 14종 + 상황 7종

전체 라벨(emotic 원본 26+α종 + manual 폴더명 28종)을 다 나열해서
동의어 묶기를 논의한 끝에, 사용자가 최종 표를 제시:

**감정 14종**: happiness, calm, anticipation, affection, anger, fear,
sadness, disconnection, suffering, aversion, embarrassment, confidence,
confusion, yearning

**상황 8종(표 기준) → 실제 7종**: conflict, danger, everyday,
loss_absence, pressure, safety, teasing, vanity (단, everyday는 아래
규칙으로 전부 제거됨)

적용 세부사항:

- Fatigue가 기존엔 disconnection 묶음이었는데 **suffering으로
  이동**(Pain/Suffering/Fatigue)
- Confidence/Embarrassment가 독립 클래스로 분리(기존엔 calm/
  doubt_confusion에 묶여있었음)
- Disquietment→fear, Doubt/Confusion→confusion으로 흡수
- teasing, vanity는 감정에서 상황으로 전환(기존엔 emotion 매핑이
  있었음) — 해당 폴더의 감정 필드는 비움
- social_gathering(1건 출처 포함 16건), superiority(1건)는 최종
  표에 없어서 기본값으로 각각 everyday, pressure에 합침
- **everyday가 단독 라벨인 행은 전부 제거**(멀티레이블이면 유지하라고
  했으나 현재 전부 단독이라 96건 전부 제거) — "의미가 없기 때문"

**최종 분포**:

- emotic 풀(1,737건): disconnection 720·yearning 319·fear 238·
  confusion 218·suffering 211·sadness 172·happiness 130·anticipation
  108·calm 107·affection 107·embarrassment 62·anger 52·confidence
  47·aversion 45
- manual 감정(99건): happiness 27·fear 25·anger 16·disconnection
  12·yearning 6·calm 6·empathy 4·confusion 3
- manual 상황(128건): safety 55·danger 43·conflict 16·teasing 8·
  loss_absence 3·pressure 2·vanity 1

---

## 5. Qwen 재검증용 MCQ 큐 생성

다음 단계 논의("공격모델 노이즈 vs 다른 방안") 끝에, **오답 보기
설계를 정교화**하는 방향으로 결정 — build_qwen_recheck_queue.py 작성.

- 문제당 4지선다: 정답 1개(멀티레이블이면 무작위 1개 보장) + 같은
  축 라벨 풀에서 오답 3개, 순서 셔플
- emotic은 emotion 축만, manual은 emotion/situation 축 분리(19건은
  둘 다 있어서 문제 2개)
- **출력**: qwen_recheck_queue.csv, 총 1,961문제(emotion 1,833 +
  situation 128), 이미지 1,942장 기준

---

## 6. 로컬 Qwen2.5-VL-3B 실행 시도 — 디스크 용량으로 보류

API 키 없이 로컬 GPU에서 Qwen/Qwen2.5-VL-3B-Instruct를 돌리자는
요청으로 환경 조사 및 구성을 시도:

- **기존 흔적 없음**: 원래 Qwen 공격 검증은 Google Colab에서 돌렸던
  것(image_path가 /content/drive/MyDrive/agami/...), 이 저장소엔
  호출 스크립트/모델 캐시 전혀 없음
- **GPU 없음**: nvidia-smi 자체가 없고 torch.cuda.is_available()=False
  — 이 컨테이너는 CPU 전용
- 패키지 설치: transformers 5.12.1, accelerate 1.14.0, qwen-vl-utils
  0.0.14 설치, torch를 2.0.1(CUDA) → 2.12.1+cpu로 교체
  (transformers>=torch 2.4 요구사항 때문), triton/torchvision/
  torchaudio 버전 충돌까지 해결해서 Qwen2_5_VLForConditionalGeneration
  import까지는 성공
- **모델 다운로드 직전 중단**: Qwen/Qwen2.5-VL-3B-Instruct 전체 용량
  7.52GB vs 디스크 여유 7.8~8.1GB — 안전 마진이 없음
- conda clean/pkgs 캐시 삭제(5.7GB) 시도했지만 **df 수치가 거의 안
  바뀜** — 이 컨테이너의 오버레이FS 용량 통계가 호스트 노드의
  containerd 스냅샷 전체를 반영하는 구조라, 컨테이너 안에서 추가로
  비울 수 있는 게 거의 없다는 게 확인됨
- 사용자 판단: **여기서 다운로드는 포기, 다른 목표로 전환**(설치된
  패키지들은 남겨둠 — 나중에 디스크 여유 생기면 모델 다운로드부터
  이어가면 됨)

---

## 다음 단계 (우선순위 후보)

1. **despair 4건**: Qwen 공격이 안 된 상태라 별도로 처리 필요(공격
   재실행 또는 제외 결정)
2. **Qwen 재검증 실행 방법 결정**: 로컬 GPU 불가로 확인됨 — (a) 실제
   GPU 서버 확보, (b) API 키 기반 호출로 전환, (c) 더 작은 모델로
   전환 중 선택 필요
3. qwen_recheck_queue.csv(1,961문제)는 생성 완료 상태로 대기 중 —
   실행 환경만 정해지면 바로 사용 가능
4. (선행 문서에서 계속 미해결) GPT-4V/Gemini 교차검증 — 아직 미착수

---

## 산출물 정리

| 파일 | 상태 |
|---|---|
| captcha_bank_human_reviewed.csv | 1,737건 (reviewed 1,668 / excluded 69). emotic 원본 라벨(대부분) + 14클래스 최종 정리 |
| manual_images_labeled.csv | 224건(원래 194 + Qwen 정답이었던 30건 재투입). 감정 8종(empathy 포함)+상황 7종 |
| excluded_pool.csv | 1,144건 누적 보관 |
| engagement_review_queue.csv | 895건 전부 처리 완료(labeled 50 / excluded 845) |
| build_engagement_review_queue.py, engagement_review_app.py | Engagement 전용 검수 도구 |
| restore_original_emotic_labels.py | 원본 라벨 복원 스크립트(1차, 이후 추가 수정은 즉석 스크립트로 처리) |
| build_qwen_recheck_queue.py, qwen_recheck_queue.csv | Qwen 재검증용 4지선다 MCQ 생성기 + 결과(1,961문제) |
| review_app.py | 4개 축(emotion_class/EMOTIC raw/situation/manual) 멀티셀렉트로 확장, bbox 표시 제거 |
