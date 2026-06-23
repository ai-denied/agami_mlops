"""Build context_emotion_train_dataset_v1.csv from the reconstructed label
pools (captcha_bank_human_reviewed.csv, manual_images_labeled.csv).

Does NOT touch the four upstream files (captcha_bank_human_reviewed.csv,
manual_images_labeled.csv, excluded_pool.csv, manual_images_unresolved.csv)
- those stay as the "full reconstructed pool" artifacts. This script reads
them, filters/validates/samples/splits, and writes its own v1-suffixed
outputs.

Usage:
    python build_train_dataset_v1.py \
        --processed-dir /workspace/data/context_emotion/processed \
        --annotations /workspace/data/context_emotion/emotic_dataset/Annotations/Annotations.mat \
        --out-dir /workspace/data/context_emotion/processed
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import scipy.io as sio

from emotion_mapping import EMOTION_CLASSES, SITUATION_CLASSES, \
    EMOTIC_CATEGORY_TO_EMOTION, MANUAL_FOLDER_TO_LABELS, NO_SCHEMA_SLOT_FOLDERS
from image_paths import resolve_emotic_path, resolve_manual_path
from normalize_label import normalize_emotion, normalize_situation
from restore_emotic_labels import annotator_category_lists, iter_people

MAX_PER_EMOTION_CLASS = 800
LOW_RESOURCE_THRESHOLD = 300
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MANUAL_LABEL_CONFIDENCE = 0.6  # flat heuristic - see build report rationale
RANDOM_SEED = 13


def sample_id_for(source, natural_key):
    digest = hashlib.md5(natural_key.encode("utf-8")).hexdigest()[:10]
    return f"{source}-{digest}"


def build_emotic_confidence_lookup(annotations_path):
    """(split, folder, filename, person_index) -> mean fraction of annotators
    who voted for each category in that person's majority-vote set."""
    mat = sio.loadmat(annotations_path, squeeze_me=True, struct_as_record=False)
    lookup = {}
    for split in ("train", "val", "test"):
        for struct in mat[split]:
            filename = str(struct.filename)
            folder = str(struct.folder)
            for p_idx, person in enumerate(iter_people(struct)):
                cat_lists = annotator_category_lists(person.annotations_categories)
                n = len(cat_lists)
                counts = Counter(c for cats in cat_lists for c in cats)
                fractions = [v / n for cat, v in counts.items() if v / n >= 0.5]
                mean_fraction = sum(fractions) / len(fractions) if fractions else 0.0
                lookup[(split, folder, filename, p_idx)] = round(mean_fraction, 2)
    return lookup


def load_emotic_candidates(processed_dir, confidence_lookup):
    rows = []
    path = os.path.join(processed_dir, "captcha_bank_human_reviewed.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["split"], r["folder"], r["filename"], int(r["person_index"]))
            confidence = confidence_lookup.get(key, 0.0)
            raw_emotions = [e for e in r["emotion_classes"].split(";") if e]
            primary_raw = raw_emotions[0] if raw_emotions else ""
            emotion_norm = normalize_emotion(primary_raw)
            image_path = resolve_emotic_path(r["folder"], r["filename"])
            natural_key = f"emotic|{r['split']}|{r['folder']}|{r['filename']}|{r['person_index']}"
            rows.append({
                "sample_id": sample_id_for("emotic", natural_key),
                "image_path": image_path or "",
                "source": "emotic",
                "original_labels": r["raw_categories_majority"],
                "emotion_label_raw": emotion_norm,
                "situation_label_raw": None,  # EMOTIC has no situation axis
                "review_status": r["label_status"],
                "label_confidence": confidence,
                "exclude_reason": "",
            })
    return rows


def load_manual_candidates(processed_dir):
    rows = []
    path = os.path.join(processed_dir, "manual_images_labeled.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            image_path = resolve_manual_path(r["folder"], r["filename"])
            emotion_norm = normalize_emotion(r["emotion_class"]) if r["emotion_class"] else None
            situation_norm = normalize_situation(r["situation_class"]) if r["situation_class"] else None
            natural_key = f"manual|{r['folder']}|{r['filename']}"
            rows.append({
                "sample_id": sample_id_for("manual", natural_key),
                "image_path": image_path or "",
                "source": "manual",
                "original_labels": r["folder"],
                "emotion_label_raw": emotion_norm,
                "situation_label_raw": situation_norm,
                "review_status": r["label_status"],
                "label_confidence": MANUAL_LABEL_CONFIDENCE,
                "exclude_reason": "",
            })
    return rows


def filter_candidates(rows):
    kept, excluded = [], []
    for r in rows:
        reasons = []
        if not r["image_path"]:
            reasons.append("image_not_found")
        emotion_label = r["emotion_label_raw"]
        situation_label = r["situation_label_raw"]
        if emotion_label == "__unknown__":
            reasons.append("emotion_label_not_in_schema")
            emotion_label = None
        if situation_label == "__unknown__":
            reasons.append("situation_label_not_in_schema")
            situation_label = None
        if not emotion_label and not situation_label:
            reasons.append("empty_label")

        r["emotion_label"] = emotion_label or ""
        r["situation_label"] = situation_label or ""
        del r["emotion_label_raw"]
        del r["situation_label_raw"]

        if reasons:
            r["exclude_reason"] = ";".join(reasons)
            excluded.append(r)
        else:
            kept.append(r)
    return kept, excluded


def balanced_sample(rows, rng):
    # manual rows are scarce and hand-curated - they must never lose a random
    # shuffle coin-flip against the much larger emotic pool for the same
    # class. So per class: keep ALL manual rows first, then fill the
    # remaining quota up to MAX_PER_EMOTION_CLASS from emotic only.
    by_emotion = defaultdict(lambda: {"manual": [], "emotic": []})
    no_emotion = []
    for r in rows:
        if r["emotion_label"]:
            by_emotion[r["emotion_label"]][r["source"]].append(r)
        else:
            no_emotion.append(r)  # situation-only manual rows (e.g. safety, danger)

    sampled = []
    dropped = []  # rows excluded during sampling, kept with a reason
    low_resource_classes = set()
    for cls in EMOTION_CLASSES:
        manual_pool = by_emotion[cls]["manual"]
        emotic_pool = by_emotion[cls]["emotic"]
        rng.shuffle(emotic_pool)
        remaining_quota = max(0, MAX_PER_EMOTION_CLASS - len(manual_pool))
        kept_emotic = emotic_pool[:remaining_quota]
        overflow_emotic = emotic_pool[remaining_quota:]

        total = len(manual_pool) + len(kept_emotic)
        if total < LOW_RESOURCE_THRESHOLD:
            low_resource_classes.add(cls)

        sampled.extend(manual_pool)
        sampled.extend(kept_emotic)
        for r in overflow_emotic:
            r["exclude_reason"] = "class_quota_exceeded"
            dropped.append(r)

    # situation-only rows (no emotion_label) aren't capped by the emotion-class
    # rule above; keep them all - they carry the only everyday/safety/danger/etc.
    # signal in the manual pool and are comparatively rare already.
    sampled.extend(no_emotion)

    # dedup by image_path - the same photo never appears twice (and therefore
    # never crosses a split boundary later).
    seen_paths = set()
    deduped = []
    for r in sampled:
        if r["image_path"] in seen_paths:
            r["exclude_reason"] = "duplicate_image"
            dropped.append(r)
            continue
        seen_paths.add(r["image_path"])
        deduped.append(r)

    return deduped, low_resource_classes, dropped


def stratified_split(rows, rng):
    by_emotion = defaultdict(list)
    for r in rows:
        by_emotion[r["emotion_label"] or "__no_emotion__"].append(r)

    assignments = {}
    for cls, group in by_emotion.items():
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * SPLIT_RATIOS["train"])
        n_val = round(n * SPLIT_RATIOS["val"])
        for i, r in enumerate(group):
            if i < n_train:
                assignments[r["sample_id"]] = "train"
            elif i < n_train + n_val:
                assignments[r["sample_id"]] = "val"
            else:
                assignments[r["sample_id"]] = "test"
    for r in rows:
        r["dataset_split"] = assignments[r["sample_id"]]
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    rng = random.Random(RANDOM_SEED)

    print("Parsing Annotations.mat for per-row label_confidence ...")
    confidence_lookup = build_emotic_confidence_lookup(args.annotations)

    print("Loading candidates ...")
    candidates = (
        load_emotic_candidates(args.processed_dir, confidence_lookup)
        + load_manual_candidates(args.processed_dir)
    )
    print(f"  candidates: {len(candidates)}")

    kept, filter_excluded = filter_candidates(candidates)
    print(f"  kept after filtering: {len(kept)}  excluded: {len(filter_excluded)}")

    sampled, low_resource_classes, sampling_dropped = balanced_sample(kept, rng)
    print(f"  after balanced sampling + dedup: {len(sampled)} "
          f"(dropped during sampling: {len(sampling_dropped)})")
    if low_resource_classes:
        print(f"  low_resource emotion classes (<{LOW_RESOURCE_THRESHOLD}): "
              f"{sorted(low_resource_classes)}")

    # every row excluded anywhere in the pipeline (filter stage + sampling
    # stage) ends up in context_emotion_excluded_v1.csv with its reason -
    # nothing is silently dropped.
    excluded = filter_excluded + sampling_dropped

    final_rows = stratified_split(sampled, rng)

    fieldnames = ["sample_id", "image_path", "source", "original_labels",
                  "emotion_label", "situation_label", "review_status",
                  "label_confidence", "exclude_reason", "dataset_split"]
    train_path = os.path.join(args.out_dir, "context_emotion_train_dataset_v1.csv")
    write_csv(train_path, final_rows, fieldnames)

    excluded_fieldnames = ["sample_id", "image_path", "source", "original_labels",
                            "emotion_label", "situation_label", "review_status",
                            "label_confidence", "exclude_reason"]
    excluded_path = os.path.join(args.out_dir, "context_emotion_excluded_v1.csv")
    write_csv(excluded_path, excluded, excluded_fieldnames)

    # ---- distribution report ----
    emotion_counts = Counter(r["emotion_label"] for r in final_rows if r["emotion_label"])
    situation_counts = Counter(r["situation_label"] for r in final_rows if r["situation_label"])
    combo_counts = Counter(
        (r["emotion_label"], r["situation_label"]) for r in final_rows
        if r["emotion_label"] and r["situation_label"]
    )
    source_counts = Counter(r["source"] for r in final_rows)
    split_counts = Counter(r["dataset_split"] for r in final_rows)
    exclude_reason_counts = Counter()
    for r in excluded:
        for reason in r["exclude_reason"].split(";"):
            exclude_reason_counts[reason] += 1
    excluded_by_source = Counter(r["source"] for r in excluded)

    dist_lines = ["# context_emotion_label_distribution_v1", ""]
    dist_lines += [f"## 감정 {len(EMOTION_CLASSES)}종별 건수 (최종 학습셋)", ""]
    for cls in EMOTION_CLASSES:
        flag = " (low_resource)" if cls in low_resource_classes else ""
        dist_lines.append(f"- {cls}: {emotion_counts.get(cls, 0)}{flag}")
    dist_lines += ["", f"## 상황 {len(SITUATION_CLASSES)}종별 건수 (최종 학습셋)", ""]
    for cls in SITUATION_CLASSES:
        dist_lines.append(f"- {cls}: {situation_counts.get(cls, 0)}")
    dist_lines += ["", "## 감정 x 상황 조합별 건수", ""]
    for (e, s), n in sorted(combo_counts.items(), key=lambda kv: -kv[1]):
        dist_lines.append(f"- {e} x {s}: {n}")
    if not combo_counts:
        dist_lines.append("(없음 - emotic은 situation 축이 없고, manual은 폴더당 emotion/situation 중 하나만 갖는 경우가 대부분)")
    dist_lines += ["", "## source별 건수", ""]
    for src, n in source_counts.items():
        dist_lines.append(f"- {src}: {n}")
    dist_lines += ["", "## dataset_split별 건수", ""]
    for split, n in split_counts.items():
        dist_lines.append(f"- {split}: {n}")
    dist_lines += ["", "## 제외 사유별 건수 (context_emotion_excluded_v1.csv)", ""]
    for reason, n in sorted(exclude_reason_counts.items(), key=lambda kv: -kv[1]):
        dist_lines.append(f"- {reason}: {n}")

    dist_path = os.path.join(args.out_dir, "context_emotion_label_distribution_v1.md")
    with open(dist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dist_lines) + "\n")

    # ---- label mapping json ----
    mapping_out = {
        "emotion_classes": EMOTION_CLASSES,
        "situation_classes": SITUATION_CLASSES,
        "emotic_category_to_emotion": EMOTIC_CATEGORY_TO_EMOTION,
        "manual_folder_to_labels": {
            k: {"emotion": v[0], "situation": v[1]} for k, v in MANUAL_FOLDER_TO_LABELS.items()
        },
        "no_schema_slot_folders": sorted(NO_SCHEMA_SLOT_FOLDERS),
        "normalization_notes": (
            "normalize_label.py lowercases/strips separators and maps known "
            "typo variants (doubt_confusion, doubt_confusning, ...) before "
            "validating against emotion_classes/situation_classes. Engagement, "
            "Surprise and the literal string 'test' are explicitly mapped to "
            "null (out-of-schema), not treated as typos."
        ),
    }
    mapping_path = os.path.join(args.out_dir, "context_emotion_label_mapping_v1.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_out, f, ensure_ascii=False, indent=2)

    # ---- build report ----
    report_lines = ["# context_emotion_dataset_build_report_v1", ""]
    report_lines += ["## 입력", ""]
    report_lines.append(f"- captcha_bank_human_reviewed.csv + manual_images_labeled.csv 합산 candidate: {len(candidates)}")
    report_lines.append("- excluded_pool.csv, manual_images_unresolved.csv는 사용하지 않음 (학습 후보에서 원천 배제)")
    report_lines += ["", "## 필터링 (작업 전후 건수)", ""]
    report_lines.append(f"- candidate {len(candidates)} -> 필터 통과 {len(kept)} / 제외(이미지 없음 등) {len(filter_excluded)}")
    report_lines += ["", "## 균형 샘플링 (작업 전후 건수)", ""]
    report_lines.append(f"- 필터 통과 {len(kept)} -> 샘플링+중복제거 후 {len(sampled)} (샘플링 단계에서 추가로 빠진 건수 {len(sampling_dropped)})")
    report_lines.append(f"- 감정 클래스별 최대 {MAX_PER_EMOTION_CLASS}장 캡, {LOW_RESOURCE_THRESHOLD}장 미만은 low_resource 표시 후 가능한 만큼 포함")
    report_lines.append(f"- low_resource 클래스: {sorted(low_resource_classes) if low_resource_classes else '없음'}")
    report_lines.append("- manual 행은 source별로 cap을 먼저 보장(전부 유지)한 뒤, 같은 클래스의 emotic 행으로 나머지 800장 한도를 채움 - manual이 emotic과 같은 풀에서 랜덤 셔플로 경쟁하다 우연히 캡 밖으로 밀려나는 것을 방지")
    report_lines.append("- situation만 있고 emotion이 없는 manual 행(safety/danger/conflict/pressure/teasing/vanity/loss_absence)은 감정 클래스 캡 대상이 아니므로 전부 유지")
    report_lines += ["", "## 전체 제외 사유 분포 (context_emotion_excluded_v1.csv, 필터+샘플링 단계 합산)", ""]
    report_lines.append(f"- 사유별: {dict(exclude_reason_counts)}")
    report_lines.append(f"- source별: {dict(excluded_by_source)}")
    report_lines += ["", "## split", ""]
    report_lines.append(f"- train/val/test = {SPLIT_RATIOS}, 감정 라벨 기준 stratified, 시드={RANDOM_SEED}")
    report_lines.append(f"- 최종 split별 건수: {dict(split_counts)}")
    report_lines.append("- 이미지 단위로 먼저 중복 제거한 뒤 split을 나눠서, 동일 이미지가 다른 split에 들어가는 경우 없음")
    report_lines += ["", "## label_confidence 산정 방식", ""]
    report_lines.append("- emotic: Annotations.mat을 다시 파싱해, 해당 행의 raw_categories_majority에 속한 각 카테고리의 (동의한 annotator 수 / 전체 annotator 수) 평균. train split은 annotator 1명이라 항상 1.0")
    report_lines.append(f"- manual: 폴더명 기반 휴리스틱 재구성이라 사람/모델 검증이 없어 고정값 {MANUAL_LABEL_CONFIDENCE}")
    report_lines += ["", "## 최종 산출물 건수", ""]
    report_lines.append(f"- context_emotion_train_dataset_v1.csv: {len(final_rows)}")
    report_lines.append(f"- context_emotion_excluded_v1.csv: {len(excluded)}")

    report_path = os.path.join(args.out_dir, "context_emotion_dataset_build_report_v1.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nwrote: {train_path} ({len(final_rows)} rows)")
    print(f"wrote: {excluded_path} ({len(excluded)} rows)")
    print(f"wrote: {dist_path}")
    print(f"wrote: {mapping_path}")
    print(f"wrote: {report_path}")


if __name__ == "__main__":
    main()
