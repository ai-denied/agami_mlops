"""Build context_emotion_train_dataset_v2.csv - same source pools as v1
(captcha_bank_human_reviewed.csv, manual_images_labeled.csv) but a wider
feature schema requested for the human-review handoff: per-image
dimensions/bbox, a full candidate_emotions list separate from a single
provisional_emotion, content/perceptual hashes, and a split_group_id so
near-duplicate images never land in different train/val/test splits.
The reviewer_answer/reviewer_confidence/reviewer_note/reviewed_at columns
are left blank with review_status='unreviewed' - no human has reviewed
this reconstruction, so the dataset must not claim otherwise.

Usage:
    python build_train_dataset_v2.py \
        --processed-dir /workspace/data/context_emotion/processed \
        --annotations /workspace/data/context_emotion/emotic_dataset/Annotations/Annotations.mat \
        --out-dir /workspace/data/context_emotion/processed
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

from build_train_dataset_v1 import (
    build_emotic_confidence_lookup, sample_id_for,
    MAX_PER_EMOTION_CLASS, LOW_RESOURCE_THRESHOLD, SPLIT_RATIOS,
    MANUAL_LABEL_CONFIDENCE, RANDOM_SEED,
)
from emotion_mapping import EMOTION_CLASSES, SITUATION_CLASSES
from image_paths import resolve_emotic_path, resolve_manual_path, inspect_image
from normalize_label import normalize_emotion, normalize_situation

DATA_ROOT = "/workspace/data/context_emotion"


def relative_path(absolute_path):
    return os.path.relpath(absolute_path, DATA_ROOT)


def format_bbox(bbox_str):
    if not bbox_str:
        return ""
    values = [float(v) for v in bbox_str.split(";")]
    return "[" + ",".join(str(v) for v in values) + "]"


def load_emotic_candidates(processed_dir, confidence_lookup):
    rows = []
    path = os.path.join(processed_dir, "captcha_bank_human_reviewed.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["split"], r["folder"], r["filename"], int(r["person_index"]))
            confidence = confidence_lookup.get(key, 0.0)
            candidate_emotions = [e for e in r["emotion_classes"].split(";") if e]
            provisional_raw = candidate_emotions[0] if candidate_emotions else ""
            provisional_norm = normalize_emotion(provisional_raw)
            image_path = resolve_emotic_path(r["folder"], r["filename"])
            natural_key = f"emotic|{r['split']}|{r['folder']}|{r['filename']}|{r['person_index']}"
            rows.append({
                "sample_id": sample_id_for("emotic", natural_key),
                "image_path": image_path or "",
                "source": "emotic",
                "source_image_id": f"{r['folder']}/{r['filename']}",
                "target_person_bbox": format_bbox(r["bbox"]),
                "original_labels": r["raw_categories_majority"],
                "candidate_emotions_raw": candidate_emotions,
                "provisional_emotion_raw": provisional_norm,
                "situation_label_raw": None,  # EMOTIC has no situation axis
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
            provisional_norm = normalize_emotion(r["emotion_class"]) if r["emotion_class"] else None
            situation_norm = normalize_situation(r["situation_class"]) if r["situation_class"] else None
            candidate_emotions = [r["emotion_class"]] if r["emotion_class"] else []
            natural_key = f"manual|{r['folder']}|{r['filename']}"
            rows.append({
                "sample_id": sample_id_for("manual", natural_key),
                "image_path": image_path or "",
                "source": "manual",
                "source_image_id": f"{r['folder']}/{r['filename']}",
                # no real person-detection bbox exists for manual images -
                # left blank rather than faked as a full-image box.
                "target_person_bbox": "",
                "original_labels": r["folder"],
                "candidate_emotions_raw": candidate_emotions,
                "provisional_emotion_raw": provisional_norm,
                "situation_label_raw": situation_norm,
                "label_confidence": MANUAL_LABEL_CONFIDENCE,
                "exclude_reason": "",
            })
    return rows


def filter_and_inspect(rows):
    kept, excluded = [], []
    for r in rows:
        reasons = []
        meta = None
        if not r["image_path"]:
            reasons.append("image_not_found")
        else:
            meta = inspect_image(r["image_path"])

        provisional = r["provisional_emotion_raw"]
        situation = r["situation_label_raw"]
        if provisional == "__unknown__":
            reasons.append("emotion_label_not_in_schema")
            provisional = None
        if situation == "__unknown__":
            reasons.append("situation_label_not_in_schema")
            situation = None
        if not provisional and not situation:
            reasons.append("empty_label")

        r["provisional_emotion"] = provisional or ""
        r["candidate_emotions"] = ";".join(r["candidate_emotions_raw"])
        r["situation_label"] = situation or ""
        r["image_width"] = meta["width"] if meta else ""
        r["image_height"] = meta["height"] if meta else ""
        r["content_hash"] = meta["content_hash"] if meta else ""
        r["perceptual_hash"] = meta["perceptual_hash"] if meta else ""
        for k in ("provisional_emotion_raw", "situation_label_raw", "candidate_emotions_raw"):
            del r[k]

        if reasons:
            r["exclude_reason"] = ";".join(reasons)
            excluded.append(r)
        else:
            kept.append(r)
    return kept, excluded


def balanced_sample(rows, rng):
    by_emotion = defaultdict(lambda: {"manual": [], "emotic": []})
    no_emotion = []
    for r in rows:
        if r["provisional_emotion"]:
            by_emotion[r["provisional_emotion"]][r["source"]].append(r)
        else:
            no_emotion.append(r)

    sampled = []
    dropped = []
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

    sampled.extend(no_emotion)
    return sampled, low_resource_classes, dropped


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def assign_split_groups(rows):
    """Two rows land in the same split_group_id if they share a
    source_image_id (same photo, different bbox/person), an exact
    content_hash, or an exact perceptual_hash. This catches the literal
    duplicate copies across the mirrored EMOTIC directories as well as
    multiple person-instances cropped from the same photo - either case
    must never be split across train/val/test."""
    uf = UnionFind()
    for r in rows:
        uf.union(("source_image_id", r["source_image_id"]), ("content_hash", r["content_hash"]))
        uf.union(("content_hash", r["content_hash"]), ("perceptual_hash", r["perceptual_hash"]))

    group_members = defaultdict(list)
    for r in rows:
        root = uf.find(("source_image_id", r["source_image_id"]))
        group_members[root].append(r)

    for root, members in group_members.items():
        group_id = "grp-" + sample_id_for("g", "|".join(sorted(m["sample_id"] for m in members)))[2:]
        for m in members:
            m["split_group_id"] = group_id
    return rows


def stratified_split_by_group(rows, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[r["split_group_id"]].append(r)

    # one stratification label per group: the first member's provisional
    # emotion (falls back to "no_emotion" for situation-only manual rows).
    by_emotion = defaultdict(list)
    for group_id, members in groups.items():
        label = members[0]["provisional_emotion"] or "__no_emotion__"
        by_emotion[label].append(group_id)

    group_split = {}
    for label, group_ids in by_emotion.items():
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_train = round(n * SPLIT_RATIOS["train"])
        n_val = round(n * SPLIT_RATIOS["val"])
        for i, gid in enumerate(group_ids):
            if i < n_train:
                group_split[gid] = "train"
            elif i < n_train + n_val:
                group_split[gid] = "val"
            else:
                group_split[gid] = "test"

    for r in rows:
        r["dataset_split"] = group_split[r["split_group_id"]]
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


REVIEWER_FIELDS = {
    "reviewer_answer": "",
    "reviewer_confidence": "",
    "reviewer_note": "",
    "review_status": "unreviewed",
    "reviewed_at": "",
}


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

    print("Resolving image paths + computing width/height/hashes "
          "(this opens every candidate image once) ...")
    kept, filter_excluded = filter_and_inspect(candidates)
    print(f"  kept after filtering: {len(kept)}  excluded: {len(filter_excluded)}")

    sampled, low_resource_classes, sampling_dropped = balanced_sample(kept, rng)
    print(f"  after balanced sampling: {len(sampled)} "
          f"(dropped during sampling: {len(sampling_dropped)})")
    if low_resource_classes:
        print(f"  low_resource emotion classes (<{LOW_RESOURCE_THRESHOLD}): "
              f"{sorted(low_resource_classes)}")

    excluded = filter_excluded + sampling_dropped

    sampled = assign_split_groups(sampled)
    final_rows = stratified_split_by_group(sampled, rng)
    for r in final_rows:
        r.update(REVIEWER_FIELDS)
        r["image_path"] = relative_path(r["image_path"]) if r["image_path"] else ""

    fieldnames = ["sample_id", "image_path", "source", "source_image_id",
                  "image_width", "image_height", "target_person_bbox",
                  "original_labels", "candidate_emotions", "provisional_emotion",
                  "situation_label", "label_confidence", "reviewer_answer",
                  "reviewer_confidence", "reviewer_note", "review_status",
                  "reviewed_at", "content_hash", "perceptual_hash",
                  "split_group_id", "dataset_split", "exclude_reason"]
    train_path = os.path.join(args.out_dir, "context_emotion_train_dataset_v2.csv")
    write_csv(train_path, final_rows, fieldnames)

    excluded_fieldnames = [c for c in fieldnames if c != "dataset_split"]
    for r in excluded:
        r.setdefault("split_group_id", "")
        r["image_path"] = relative_path(r["image_path"]) if r["image_path"] else ""
    excluded_path = os.path.join(args.out_dir, "context_emotion_excluded_v2.csv")
    write_csv(excluded_path, excluded, excluded_fieldnames)

    # ---- distribution + build report ----
    emotion_counts = Counter(r["provisional_emotion"] for r in final_rows if r["provisional_emotion"])
    situation_counts = Counter(r["situation_label"] for r in final_rows if r["situation_label"])
    source_counts = Counter(r["source"] for r in final_rows)
    split_counts = Counter(r["dataset_split"] for r in final_rows)
    group_count = len(set(r["split_group_id"] for r in final_rows))
    exclude_reason_counts = Counter()
    for r in excluded:
        for reason in r["exclude_reason"].split(";"):
            exclude_reason_counts[reason] += 1

    dist_lines = ["# context_emotion_label_distribution_v2", ""]
    dist_lines += [f"## 감정 {len(EMOTION_CLASSES)}종별 건수 (최종 학습셋)", ""]
    for cls in EMOTION_CLASSES:
        flag = " (low_resource)" if cls in low_resource_classes else ""
        dist_lines.append(f"- {cls}: {emotion_counts.get(cls, 0)}{flag}")
    dist_lines += ["", f"## 상황 {len(SITUATION_CLASSES)}종별 건수 (최종 학습셋)", ""]
    for cls in SITUATION_CLASSES:
        dist_lines.append(f"- {cls}: {situation_counts.get(cls, 0)}")
    dist_lines += ["", "## source별 건수", ""]
    for src, n in source_counts.items():
        dist_lines.append(f"- {src}: {n}")
    dist_lines += ["", "## dataset_split별 건수", ""]
    for split, n in split_counts.items():
        dist_lines.append(f"- {split}: {n}")
    dist_lines += ["", f"## split_group_id 개수: {group_count} (행 {len(final_rows)}건이 이 그룹으로 묶임)", ""]
    dist_lines += ["", "## 제외 사유별 건수 (context_emotion_excluded_v2.csv)", ""]
    for reason, n in sorted(exclude_reason_counts.items(), key=lambda kv: -kv[1]):
        dist_lines.append(f"- {reason}: {n}")

    dist_path = os.path.join(args.out_dir, "context_emotion_label_distribution_v2.md")
    with open(dist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dist_lines) + "\n")

    report_lines = ["# context_emotion_dataset_build_report_v2", ""]
    report_lines += ["## v1과의 차이", ""]
    report_lines.append("- emotion_label -> candidate_emotions(멀티라벨 전체) + provisional_emotion(대표 1개)로 분리")
    report_lines.append("- source_image_id, image_width, image_height, target_person_bbox 추가 (emotic만 실제 bbox 보유, manual은 빈 값)")
    report_lines.append("- content_hash(정확 중복), perceptual_hash(근접 중복), split_group_id(둘 중 하나라도 같으면 같은 그룹) 추가")
    report_lines.append("- dedup을 image_path 단위가 아니라 split_group_id 단위로 변경 - 같은 사진 속 다른 인물(다른 bbox)은 더 이상 '중복'으로 취급해 버리지 않고, 대신 같은 split_group_id로 묶어서 같은 split에만 들어가게 함")
    report_lines.append("- reviewer_answer/reviewer_confidence/reviewer_note/reviewed_at 추가, review_status는 전부 'unreviewed' (실제 검수자가 아직 검토하지 않았으므로)")
    report_lines += ["", "## 필터링 / 샘플링 (작업 전후 건수)", ""]
    report_lines.append(f"- candidate {len(candidates)} -> 필터 통과 {len(kept)} / 제외 {len(filter_excluded)}")
    report_lines.append(f"- 필터 통과 {len(kept)} -> 샘플링 후 {len(sampled)} (class_quota_exceeded {len([r for r in sampling_dropped if 'class_quota_exceeded' in r['exclude_reason']])}건)")
    report_lines.append(f"- 감정 클래스별 최대 {MAX_PER_EMOTION_CLASS}장 캡, low_resource(<{LOW_RESOURCE_THRESHOLD}): {sorted(low_resource_classes) if low_resource_classes else '없음'}")
    report_lines += ["", "## split", ""]
    report_lines.append(f"- split_group_id {group_count}개를 감정 라벨 기준 stratified 70/15/15(시드 {RANDOM_SEED})로 분할, 그룹 내 모든 행은 같은 split")
    report_lines.append(f"- 최종 split별 건수: {dict(split_counts)}")
    report_lines += ["", "## 최종 산출물 건수", ""]
    report_lines.append(f"- context_emotion_train_dataset_v2.csv: {len(final_rows)}")
    report_lines.append(f"- context_emotion_excluded_v2.csv: {len(excluded)}")

    report_path = os.path.join(args.out_dir, "context_emotion_dataset_build_report_v2.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nwrote: {train_path} ({len(final_rows)} rows)")
    print(f"wrote: {excluded_path} ({len(excluded)} rows)")
    print(f"wrote: {dist_path}")
    print(f"wrote: {report_path}")


if __name__ == "__main__":
    main()
