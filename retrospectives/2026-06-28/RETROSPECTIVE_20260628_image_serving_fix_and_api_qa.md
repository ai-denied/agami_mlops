# 회고록 — 2026-06-28

**주제**: context-emotion-api 이미지 서빙 버그 수정 × 2 + 위젯팀 API QA 대응

---

## 1. 오늘 한 일

| 항목 | 결과 |
|---|---|
| 감정 라벨 집합 확인 및 전달 | 14종 영어 레이블 확정, 한국어 매핑 테이블 제공 |
| solved 마크 메커니즘 확인 | 인메모리 세션 pop (Redis/JWT/DB 없음) 확인 |
| 버그 1: image_url 경로 손실 | `os.path.basename()` 제거 → 상대경로 전체 사용으로 수정 |
| 버그 2: IMAGE_BASE_DIR 이중 경로 | PVC 내부 `context_emotion/` 중첩 발견 → env 경로 수정 |
| API 명세서 보정 | `image_url` 실제 형식, `choices` 언어(영어) 명시 |
| 이미지 서빙 최종 검증 | `status: 200` 확인 |

---

## 2. 버그 1 — image_url 서브디렉토리 손실

### 현상

```
POST /context-emotion/challenge
→ image_url: "/static/images/frame_kmzni9cc6gdddr8t.jpg"
GET  /static/images/frame_kmzni9cc6gdddr8t.jpg → 404
```

### 원인

`challenge_sampler.py`에서 `os.path.basename(image_path)`를 사용해 파일명만 추출했다.
CSV의 `image_path`는 `DATA_ROOT`(`/workspace/data/context_emotion`) 기준 상대경로(`emotic/framesdb/framesdb/images/frame_xxx.jpg`)인데, basename으로 자르면 서브디렉토리가 사라진다.

### 수정

```python
# 수정 전
filename = os.path.basename(image_path) if image_path else "unknown.jpg"
image_url = f"{_IMAGE_BASE_URL}/{filename}"

# 수정 후
image_url = f"{_IMAGE_BASE_URL}/{image_path}" if image_path else ...
```

StaticFiles는 `/data/context_emotion`을 루트로 서빙하므로 상대경로 전체를 그대로 붙이면 경로가 일치한다.

### 배운 점

basename을 쓰는 게 "단순화"처럼 보여도, 경로 구조에 의미가 있는 경우 경로 전체를 보존해야 한다.

---

## 3. 버그 2 — IMAGE_BASE_DIR 이중 경로 중첩

### 현상

버그 1 수정 후에도 여전히 404. 실제 반환 경로:
```
/static/images/emotic_dataset/emotic/emodb_small/images/bhapb62rfgt94tcs4f.jpg → 404
```

### 원인 파악 과정

```bash
kubectl exec $P -- ls /data/context_emotion
# 결과: context_emotion
```

`/data/context_emotion` 안에 또 `context_emotion/` 디렉토리가 있었다.

**원인**: 데이터 PVC를 파이프라인 pod(마운트: `/data`)에서 wget으로 채울 때 데이터를 `/data/context_emotion/` 경로에 저장했다. 그 결과 PVC 내부 구조가 `context_emotion/{emotic,emotic_dataset,manual_images,processed}/...`가 됐다. API pod에서 같은 PVC를 `/data/context_emotion`에 마운트하면 실제 이미지 경로는 `/data/context_emotion/context_emotion/emotic_dataset/...`이 된다.

### 수정

`deployment.yaml`의 `IMAGE_BASE_DIR` 환경변수 한 단계 추가:

```yaml
# 수정 전
- name: IMAGE_BASE_DIR
  value: "/data/context_emotion"

# 수정 후
- name: IMAGE_BASE_DIR
  value: "/data/context_emotion/context_emotion"
```

### 배운 점

- PVC를 채운 pod의 마운트 경로와 서빙 pod의 마운트 경로가 다를 때 경로 중첩이 생긴다.
- 볼륨 마운트 경로는 "PVC 루트의 내용이 무엇인지"를 기준으로 맞춰야 하며, 설정만 보고 가정하지 말고 `kubectl exec -- ls`로 실제 구조를 확인해야 한다.
- PVC에 데이터를 적재할 때 루트 바로 아래에 데이터를 넣는지 서브디렉토리에 넣는지 문서화해두는 게 필요하다.

---

## 4. 위젯팀 QA 대응

### 감정 라벨 집합

위젯팀이 "7-enum과 1:1 매핑 가능한지" 물었다. 코드 확인 결과:

- 우리 시스템은 **14종** 레이블 사용 (`constants.py`의 `EMOTION_CLASSES`)
- API `choices`는 **영어** 반환 (한국어 매핑은 `human_review_server.py` 내부 전용)
- Ekman 7종과 1:1 매핑 불가 — 별도 스키마
- 한국어 표시가 필요하면 클라이언트에서 직접 매핑

전체 14종 + 한국어 대응표를 위젯팀에 전달.

### solved 마크 메커니즘

`is_correct=True` 시 `challenge_sampler.invalidate(challenge_id)` 호출 → `_sessions.pop(challenge_id)`. Redis/JWT/DB 없이 Python 프로세스 인메모리. 브리지가 `is_correct: true` 응답을 받는 것이 유일한 트리거 신호이며, solved 토큰 관리는 브리지 책임.

주의: pod 재시작 시 진행 중인 challenge 세션 전부 소멸.

---

## 5. 아쉬운 점

- **터미널 63자 제약이 이번 세션도 발목**: `wget` 옵션, `python3 -c`, `kubectl exec` 멀티라인 커맨드가 줄바꿈으로 계속 깨졌다. 스크립트를 파일로 작성하는 패턴(`echo` 라인별 작성 → `python3 /tmp/t.py`)이 가장 안정적이었다.
- **같은 유형의 마운트 경로 버그가 두 번**: 이중 경로 문제는 이전에 `attempt-logs` 중첩 마운트에서도 겪었다. 볼륨 마운트 설정 시 `kubectl exec -- ls` 검증을 루틴화해야 한다.
- **명세서 예시 경로가 실제와 달랐음**: 이미지 서빙 경로를 실제로 검증하지 않고 가정해서 명세서 예시를 썼다가 위젯팀에 혼란을 줬다.

---

## 6. 다음 액션

1. **PVC 데이터 적재 방식 문서화**: 파이프라인 pod 마운트 경로 vs API pod 마운트 경로 불일치 재발 방지.
2. **이미지 서빙 E2E 테스트 추가**: challenge 발급 → 이미지 GET 200 확인하는 smoke test를 파이프라인 또는 배포 후 검증에 포함.
