# Playwright 매크로 봇 테스트 및 known-target bot 검증 회고록

**일자**: 2026-05-21  
**담당**: agami MLOps

---

## 매크로 봇 테스트 환경 구축

Playwright 기반 매크로 봇 테스트 환경을 본격적으로 구축했다.

- 브라우저에서 CAPTCHA 페이지 자동 접근 → 시작 버튼 클릭 → CAPTCHA 진입 흐름 구현
- 마우스 이동/클릭 자동화: 단순 직선 이동이 아니라 사람처럼 보이도록 랜덤 흔들림, 속도 변화, 이동 경로 변형 적용

**구현한 매크로 봇**
- `grid_search_bot`
- `random_scan_bot`
- 단순 랜덤 탐색 봇

이를 통해 실제 서비스 화면에서 봇이 CAPTCHA를 어떻게 시도하는지 검증할 수 있었다.

---

## 현재 CAPTCHA 구조 분석

테스트 과정에서 손전등 CAPTCHA의 프론트 구조를 분석했다.

- CAPTCHA는 `<canvas>` 기반이 아님 — React `div + img + overlay` 구조
- 클릭 좌표는 0~1 범위로 정규화됨
- 백엔드 verifier에서 bbox 기준으로 정답 여부 판정

즉, 실제 화면 픽셀 좌표와 백엔드가 판단하는 정규화 좌표 사이의 변환이 매우 중요했다.

---

## 데이터셋 및 Challenge 생성 구조 분석

`captcha_labels/*.json` 파일에서 bbox를 읽어 픽셀 좌표 → 정규화 좌표로 변환하는 구조를 확인했다. 데이터셋 로더에서 `center_x_norm`, `center_y_norm`을 생성하는 흐름도 확인했다.

Challenge 생성 과정에서 `FlashlightChallengeSpec` 구조를 분석하며 중요한 점을 발견했다.

> `sub_challenges`에는 bbox 정보가 포함되지 않고, `sub_answers`에만 bbox와 정답 좌표가 저장되고 있었다.

즉, 실제 사용자에게는 bbox가 노출되지 않는 구조였지만, 테스트용 매크로 검증을 위해서는 bbox를 확인할 수 있는 장치가 필요했다.

---

## 테스트용 bbox 노출 기능 추가

known-target bot 검증을 위해 테스트용으로 bbox 노출 기능을 추가했다.

- `FlashlightSubChallenge` 모델에 bbox 필드 추가
- generator에서 bbox를 `sub_challenges`에 포함하도록 수정
- 프론트에서 `window.__CURRENT_SUB__ = currentSub` 추가

이를 통해 Playwright에서 현재 문제의 bbox 정보를 읽을 수 있게 했다.

---

## known-target 매크로 봇 구현

구현 흐름:
1. Playwright에서 `window.__CURRENT_SUB__` 읽기
2. bbox 중심 좌표 계산
3. 실제 브라우저 렌더 크기 기준으로 좌표 변환
4. bbox 내부로 마우스 이동
5. 사람처럼 흔들리며 클릭

---

## 트러블슈팅 — 좌표 변환 버그

**원인**  
bbox가 이미 정규화 좌표인데, 다시 800×600 기준으로 나누고 있었음 — 정규화된 값을 다시 정규화하면서 클릭 좌표가 틀어지는 문제 발생.

**해결**  
실제 브라우저 렌더 크기 기준으로 클릭 좌표를 계산하도록 변경.

**결과**
- index 0 → 1 → 2 정상 진행
- bbox 내부 클릭 성공
- verifier 통과

정답 bbox를 알고 있는 known-target bot의 동작 검증을 완료했다.
