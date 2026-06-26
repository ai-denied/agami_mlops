# 회고록 — context-emotion-api CD 구축 + captcha-bank 파이프라인 디버깅

**세션 기간**: 2026-06-26 (단일 세션)  
**작성**: Claude (작업 회고)  
**관련 커밋**: `ed64348a` ~ `8499a232` (이 세션의 커밋 범위)

---

## 1. 오늘의 목표

1. `context-emotion-captcha-api` ArgoCD Application을 `context-emotion-api`로 이름 변경하고 GitOps CD 파이프라인을 완성한다.
2. `captcha-bank-pipeline` Argo Workflow를 실행해 `captcha-bank-model-store-pvc`에 모델을 적재하고, API가 `pool_loaded: true` 상태가 되도록 한다.

---

## 2. 한 일

### 2-1. ArgoCD Application 이름 변경

기존 `context-emotion-captcha-api` Application을 삭제하고 `context-emotion-api`로 새로 생성했다.

- `argocd/apps/context-emotion-api.yaml` 신규 작성 (기존 파일 대체)
- `manifests/context-emotion-api/` 디렉토리로 이동

처음에는 **ArgoCD Image Updater** 도입을 시도했으나, Helm 레포 404 + GitHub raw URL 터미널 라인-랩 문제로 설치가 번번이 실패했다. 결국 flashlight와 동일한 **manifest 커밋 방식**으로 전환했다.

#### manifest 커밋 패턴 (현재 채택)
```
CI(빌드) → Harbor 푸시 → deployment.yaml 이미지 태그 수정 → git commit/push → ArgoCD 자동 감지 → Rollout
```

`.github/workflows/build-captcha-bank.yml`에 `update-manifest` job 추가:
```yaml
- name: Update image tag in manifest
  run: sed -i "s|image: .../agami-captcha-bank:.*|image: .../agami-captcha-bank:${TAG}|g" manifests/.../deployment.yaml
- name: Commit and push manifest
  run: |
    git commit -m "chore: update captcha-api image → ${TAG} [skip ci]"
    git pull --rebase origin main && git push
```

---

### 2-2. Deployment 4연속 수정

Pod가 `1/1 Running`이 되기까지 4개의 문제를 순서대로 해결했다.

#### ① PVC Pending — ReadWriteMany 미지원

```
captcha-bank-model-store-pvc  Pending
captcha-bank-data-pvc         Pending
```

`local-path` provisioner는 `ReadWriteOnce`만 지원한다. `pvc-captcha-bank.yaml`의 `accessModes`를 `ReadWriteMany` → `ReadWriteOnce`로 수정 후 PVC 재생성.

#### ② CreateContainerConfigError — subPath 디렉토리 없음

```
Error: failed to create subPath directory for volumeMount "model-store"
```

`subPath: captcha_bank/current`로 마운트하려면 PVC 위에 해당 디렉토리가 미리 존재해야 한다. init container `init-dirs`를 추가해 `mkdir -p`로 선생성:

```yaml
initContainers:
- name: init-dirs
  image: busybox:1.36
  command: [sh, -c, "mkdir -p /model-store/captcha_bank/current"]
  volumeMounts:
  - name: model-store
    mountPath: /model-store
```

#### ③ init container 충돌 — Read-only file system

```
mkdir: can't create directory: Read-only file system
```

`volumes.persistentVolumeClaim.readOnly: true`는 init container를 포함한 **모든** 컨테이너의 쓰기를 막는다. `volumes` 레벨의 `readOnly` 제거, `readOnly: true`는 메인 컨테이너의 `volumeMounts`에만 유지.

#### ④ 메인 컨테이너 실패 — readOnly 경로 안에 중첩 마운트

```
cannot create mountpoint /data/context_emotion/attempt_logs: Read-only filesystem
```

`attempt-logs` PVC를 `/data/context_emotion/attempt_logs`에 마운트하려 했으나, 상위 경로 `/data/context_emotion`이 이미 readOnly로 마운트되어 있어 컨테이너 런타임이 mountpoint 디렉토리를 생성하지 못한다.

`attempt-logs` 마운트 경로를 `/attempt-logs`(최상위)로 이동, `ATTEMPT_LOG_DIR=/attempt-logs` 환경변수 업데이트.

---

### 2-3. 데이터 PVC 적재

4.1G 분량의 `context_emotion` 이미지 데이터를 `captcha-bank-data-pvc`에 올렸다.

- workspace 서버(`data-prep-workspace`, IP `10.42.3.212`)에서 Python HTTP 서버 기동
- 파이프라인 pod에서 `wget`으로 tar 다운로드 후 압축 해제

중간에 `tar.gz`(압축) 전송이 중단돼 1.9G 불완전 파일이 생겼다. 비압축 `tar`(4.1G)로 재생성해 재전송.

---

### 2-4. 파이프라인 디버깅 — 2개 스크립트 버그 수정

`captcha-bank-pipeline` 실행 후 두 단계가 순서대로 실패했다.

#### STEP 5 / package-model — LabelEncoder JSON 직렬화 오류

```
TypeError: Object of type LabelEncoder is not JSON serializable
  package_model.py, line 112: json.dump(metadata, f, ...)
```

`bundle["emotion_attacker"]` dict 안에 sklearn `LabelEncoder` 객체가 포함돼 있었다. `_json_safe()` 헬퍼를 추가해 numpy/sklearn 타입을 JSON 직렬화 가능한 값으로 변환:

```python
def _json_safe(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    try:
        import numpy as np
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
    except ImportError:
        pass
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(type(v).__name__)
```

#### STEP 7 / promote-model — global 선언 위치 SyntaxError

```
SyntaxError: name '_STORE' is used prior to global declaration
  promote_model.py, line 133
```

Python은 함수 내에서 변수를 사용한 뒤 `global` 선언할 수 없다. `main()`에서 `ap.add_argument("--store", default=_STORE)`가 `global _STORE` 선언보다 앞에 있었다. `global` 선언을 `main()` 첫 줄로 이동해 해결.

---

## 3. 현재 상태 (세션 종료 시점)

| 항목 | 상태 |
|---|---|
| ArgoCD Application `context-emotion-api` | Synced, Running |
| CI/CD (manifest 커밋 방식) | 정상 동작 |
| `captcha-bank-data-pvc` | 4.1G 데이터 적재 완료 |
| `captcha-bank-model-store-pvc` | 비어 있음 (파이프라인 미완료) |
| API health | `pool_loaded: false` (모델 미적재) |
| `package_model.py` 버그 | 수정 완료, 푸시 완료 |
| `promote_model.py` 버그 | 수정 완료, 푸시 완료 |
| **GitHub Actions 빌드** | **진행 중 또는 완료 대기** |

파이프라인 재실행은 GitHub Actions 빌드가 완료된 뒤 새 이미지가 배포되면 가능하다.

---

## 4. 좋았던 점

- **Image Updater 포기 결정이 빨랐다**: 설치 시도 → 실패 → flashlight 패턴 확인 → 동일 방식 채택까지 불필요한 시간을 낭비하지 않았다. 이미 검증된 패턴이 있으면 재발명하지 않는 것이 맞다.
- **Deployment 4연속 오류를 체계적으로 추적했다**: 각 오류의 원인이 앞선 수정의 부작용이 아니라 독립적인 레이어 문제였다. readOnly의 두 가지 적용 레벨(volumes vs volumeMounts), subPath 선생성 요건, 중첩 마운트 제약을 이번에 명확히 정리했다.

## 5. 아쉬운 점 / 리스크

- **터미널 80자 라인-랩 문제**: 긴 URL이나 명령어가 잘려서 복붙 오류가 반복됐다. 변수 할당, 스크립트 파일, Python heredoc 등 우회책을 매번 써야 했다. 터미널 width 설정을 세션 초반에 확인하는 것이 좋다.
- **promote_model.py SyntaxError가 런타임 전에 발견되지 않았다**: 파이프라인 7단계까지 다 성공하고 마지막에 SyntaxError가 터지는 건 CI에 Python lint(`python -m py_compile` 또는 `ruff`)가 없기 때문이다. 빌드 시 스크립트 전체에 `py_compile` 한 번 돌리는 것을 검토할 만하다.
- **CronWorkflow 미완료**: `spec.schedule` → `spec.schedules` 수정이 `captcha-bank-pipeline`용 CronWorkflow에는 아직 적용되지 않았다. 다음 세션에서 처리 필요.

## 6. 다음 액션

1. **GitHub Actions 빌드 완료 확인 후 파이프라인 재실행** (아래 HANDOVER.md 참조)
2. API `pool_loaded: true` 확인
3. `captcha-bank-pipeline` CronWorkflow `spec.schedule` → `spec.schedules` 수정
