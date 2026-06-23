# context_emotion_label_distribution_v2

## 감정 14종별 건수 (최종 학습셋)

- happiness: 800
- calm: 664
- anticipation: 800
- affection: 800
- anger: 800
- fear: 390
- sadness: 253 (low_resource)
- disconnection: 800
- suffering: 382
- aversion: 268 (low_resource)
- embarrassment: 76 (low_resource)
- confidence: 800
- confusion: 329
- yearning: 465

## 상황 7종별 건수 (최종 학습셋)

- conflict: 16
- danger: 43
- loss_absence: 3
- pressure: 2
- safety: 55
- teasing: 8
- vanity: 1

## source별 건수

- manual: 224
- emotic: 7528

## dataset_split별 건수

- train: 5412
- val: 1159
- test: 1181

## split_group_id 개수: 6708 (행 7752건이 이 그룹으로 묶임)


## 제외 사유별 건수 (context_emotion_excluded_v2.csv)

- class_quota_exceeded: 14410
- image_not_found: 4528
