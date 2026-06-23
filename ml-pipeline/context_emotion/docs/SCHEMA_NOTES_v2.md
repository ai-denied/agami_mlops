# context_emotion_train_dataset_v2 - schema notes

이 문서는 `build_train_dataset_v2.py`를 재실행할 때마다 덮어써지는
`context_emotion_dataset_build_report_v2.md` / `context_emotion_label_distribution_v2.md`와
달리, 수동으로 유지되는 스키마 설계 노트다. v2 검수 중 자주 나오는 질문을 정리했다.

## image_path 기준 경로

`image_path` 컬럼은 `/workspace/data/context_emotion` (`image_root`) 기준
상대경로다. 예: `emotic_dataset/emotic/emodb_small/images/xxx.jpg` ->
`/workspace/data/context_emotion/emotic_dataset/emotic/emodb_small/images/xxx.jpg`.

## review_status='unreviewed'가 전부인 이유

의도된 상태다. 이 데이터셋은 EMOTIC 원본 라벨 복원 + manual 이미지 폴더명
기반 추론으로 만든 1차 산출물이고, 사람이 1건씩 보고 답을 단 적이 없다.
`reviewer_answer/reviewer_confidence/reviewer_note/reviewed_at`이 전부 빈
값인 것도 같은 이유. 실제 1인 검수는 `review_app.py`로 별도 진행하는
단계이며, 검수가 끝난 행만 `review_status`가 바뀐다.

## provisional_emotion이 비어 있는 행 (situation_label만 있는 행)

버그 아님. situation_label은 emotion_label과 동등한 라벨 축이라
emotion CAPTCHA용 학습에서 제외할 필요 없음. `balanced_sample()`에서
`no_emotion` 그룹으로 따로 모아서 그대로 최종셋에 포함시킨다.

## manual 소스 행의 target_person_bbox가 비어 있는 이유

의도된 상태. manual 이미지는 실제 사람 탐지 bbox가 없고(직접 찍거나
생성한 이미지라 person-detector를 돌리지 않음), 전체 이미지를 bbox로
가짜로 채우는 대신 빈 값으로 남겨둔다.

## bbox 처리 (2026-06-23 수정)

- emotic 행의 bbox는 원본 Annotations.mat 좌표 기준이라 이미지 경계를
  몇 픽셀 넘어가는 경우가 있었음(반올림/리사이즈 오차로 추정) ->
  `filter_and_inspect()`에서 이미지 width/height 기준으로 clip.
- 다음 2건은 단순 clip으로 설명되지 않는 수준이라 클리핑 대신
  `exclude_reason=bbox_invalid`로 학습셋에서 제외:
  - `emotic-a7caa73392` - bbox y2가 이미지 height를 46px 초과 (다른 clip
    대상들은 1~19px 수준)
  - `emotic-430281d7f3` - bbox(317x480, 세로형)와 이미지(230x143, 가로형)
    의 가로/세로 비율이 전혀 안 맞음, 이미지 매칭 자체가 의심됨
  - 제외 목록은 `build_train_dataset_v2.py`의 `BBOX_INVALID_SAMPLE_IDS`에
    있음
