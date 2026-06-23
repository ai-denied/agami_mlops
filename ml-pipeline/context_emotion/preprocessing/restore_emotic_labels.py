"""Rebuild EMOTIC labels from the original Annotations.mat.

Reconstruction of restore_original_emotic_labels.py (lost when the pod
reset - see RECONSTRUCTION_NOTES.md). For every annotated person/bbox in
the EMOTIC dataset, takes a majority vote (>=50% of that person's
annotators) over the raw 26 EMOTIC categories, then maps the surviving
categories onto the final 14-class emotion scheme via common/constants.py.

Usage:
    python -m context_emotion.preprocessing.restore_emotic_labels \
        --annotations /workspace/data/context_emotion/emotic_dataset/Annotations/Annotations.mat \
        --out-dir /workspace/agami_mlops/ml-pipeline/context_emotion/label_pools
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.common.constants import EMOTIC_CATEGORY_TO_EMOTION  # noqa: E402


def to_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x.tolist())
    return [x]


def annotator_category_lists(annotations_categories):
    """annotations_categories is either one mat_struct (train, single
    annotator) or an ndarray of mat_structs (val/test, multiple
    annotators). Returns a list of category-lists, one per annotator."""
    entries = annotations_categories
    if not isinstance(entries, np.ndarray):
        entries = np.array([entries])
    out = []
    for entry in entries:
        cats = entry.categories if hasattr(entry, "categories") else entry
        out.append([str(c) for c in to_list(cats)])
    return out


def majority_vote(category_lists):
    n = len(category_lists)
    counts = Counter(c for cats in category_lists for c in cats)
    return sorted(cat for cat, n_votes in counts.items() if n_votes / n >= 0.5)


def map_to_emotions(raw_categories):
    mapped = []
    dropped = []
    for cat in raw_categories:
        final = EMOTIC_CATEGORY_TO_EMOTION.get(cat, "__unknown__")
        if final is None:
            dropped.append(cat)
        elif final == "__unknown__":
            dropped.append(f"UNKNOWN:{cat}")
        else:
            mapped.append(final)
    # dedup, keep order
    seen = set()
    mapped_unique = [m for m in mapped if not (m in seen or seen.add(m))]
    return mapped_unique, dropped


def iter_people(struct):
    person = struct.person
    if not isinstance(person, np.ndarray):
        person = np.array([person])
    return person


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    mat = sio.loadmat(args.annotations, squeeze_me=True, struct_as_record=False)
    os.makedirs(args.out_dir, exist_ok=True)

    bank_path = os.path.join(args.out_dir, "captcha_bank_human_reviewed.csv")
    excluded_path = os.path.join(args.out_dir, "excluded_pool.csv")

    bank_rows = []
    excluded_rows = []

    for split in ("train", "val", "test"):
        for img_idx, struct in enumerate(mat[split]):
            filename = str(struct.filename)
            folder = str(struct.folder)
            for p_idx, person in enumerate(iter_people(struct)):
                cat_lists = annotator_category_lists(person.annotations_categories)
                raw_majority = majority_vote(cat_lists)
                emotion_classes, dropped = map_to_emotions(raw_majority)
                bbox = to_list(person.body_bbox)
                row = {
                    "source": "emotic",
                    "split": split,
                    "folder": folder,
                    "filename": filename,
                    "person_index": p_idx,
                    "n_annotators": len(cat_lists),
                    "raw_categories_majority": ";".join(raw_majority),
                    "emotion_classes": ";".join(emotion_classes),
                    "dropped_categories": ";".join(dropped),
                    "bbox": ";".join(str(round(float(v), 1)) for v in bbox),
                    "label_status": "reconstructed_approx",
                    "reviewer_note": "restored_from_emotic_original_majority_vote",
                }
                if emotion_classes:
                    bank_rows.append(row)
                else:
                    row["reviewer_note"] = "excluded_no_mapped_emotion_class"
                    excluded_rows.append(row)

    fieldnames = list(bank_rows[0].keys()) if bank_rows else list(excluded_rows[0].keys())

    with open(bank_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(bank_rows)

    with open(excluded_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(excluded_rows)

    print(f"emotic bank rows: {len(bank_rows)} -> {bank_path}")
    print(f"emotic excluded rows: {len(excluded_rows)} -> {excluded_path}")

    emotion_counts = Counter()
    for row in bank_rows:
        for c in row["emotion_classes"].split(";"):
            emotion_counts[c] += 1
    for cls, n in emotion_counts.most_common():
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
