# context_emotion_dataset_build_report_v2

## v1과의 차이

- emotion_label -> candidate_emotions(멀티라벨 전체) + provisional_emotion(대표 1개)로 분리
- source_image_id, image_width, image_height, target_person_bbox 추가 (emotic만 실제 bbox 보유, manual은 빈 값)
- content_hash(정확 중복), perceptual_hash(근접 중복), split_group_id(둘 중 하나라도 같으면 같은 그룹) 추가
- dedup을 image_path 단위가 아니라 split_group_id 단위로 변경 - 같은 사진 속 다른 인물(다른 bbox)은 더 이상 '중복'으로 취급해 버리지 않고, 대신 같은 split_group_id로 묶어서 같은 split에만 들어가게 함
- reviewer_answer/reviewer_confidence/reviewer_note/reviewed_at 추가, review_status는 전부 'unreviewed' (실제 검수자가 아직 검토하지 않았으므로)

## 필터링 / 샘플링 (작업 전후 건수)

- candidate 26690 -> 필터 통과 22162 / 제외 4528
- 필터 통과 22162 -> 샘플링 후 7752 (class_quota_exceeded 14410건)
- 감정 클래스별 최대 800장 캡, low_resource(<300): ['aversion', 'embarrassment', 'sadness']

## split

- split_group_id 6708개를 감정 라벨 기준 stratified 70/15/15(시드 13)로 분할, 그룹 내 모든 행은 같은 split
- 최종 split별 건수: {'train': 5412, 'val': 1159, 'test': 1181}

## 최종 산출물 건수

- context_emotion_train_dataset_v2.csv: 7752
- context_emotion_excluded_v2.csv: 18938
