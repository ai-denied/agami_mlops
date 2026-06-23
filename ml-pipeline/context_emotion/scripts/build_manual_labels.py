"""Rebuild manual_images_labeled.csv from the manual_images/<folder>/ tree.

Reconstruction of the manual-image labeling step from the Day 4/5
retrospectives (the original manual_images_labeled.csv was lost when the
pod reset). Folder name -> (emotion_class, situation_class) comes from
emotion_mapping.MANUAL_FOLDER_TO_LABELS, reverse-engineered to match the
documented final distribution (manual 감정 99건 / 상황 128건). See
RECONSTRUCTION_NOTES.md for the despair folder, which stays unresolved -
the original work depended on a Qwen attack run that no longer exists.

Usage:
    python build_manual_labels.py \
        --manual-dir /workspace/data/context_emotion/manual_images \
        --out-dir /workspace/data/context_emotion/processed
"""
import argparse
import csv
import os

from emotion_mapping import MANUAL_FOLDER_TO_LABELS, NO_SCHEMA_SLOT_FOLDERS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labeled_path = os.path.join(args.out_dir, "manual_images_labeled.csv")
    unresolved_path = os.path.join(args.out_dir, "manual_images_unresolved.csv")

    labeled_rows = []
    unresolved_rows = []
    unknown_folders = set()

    for folder in sorted(os.listdir(args.manual_dir)):
        folder_path = os.path.join(args.manual_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        mapping = MANUAL_FOLDER_TO_LABELS.get(folder)
        if mapping is None:
            unknown_folders.add(folder)
            mapping = (None, None)
        emotion_class, situation_class = mapping

        for fname in sorted(os.listdir(folder_path)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            row = {
                "source": "manual",
                "folder": folder,
                "filename": fname,
                "emotion_class": emotion_class or "",
                "situation_class": situation_class or "",
                "label_status": "reconstructed_approx",
            }
            if folder in NO_SCHEMA_SLOT_FOLDERS:
                row["label_status"] = "unresolved"
                row["reviewer_note"] = "no_schema_slot_in_fixed_14_emotion_classes"
                unresolved_rows.append(row)
            else:
                row["reviewer_note"] = "folder_name_based_reconstruction"
                labeled_rows.append(row)

    fieldnames = ["source", "folder", "filename", "emotion_class",
                  "situation_class", "label_status", "reviewer_note"]

    with open(labeled_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(labeled_rows)

    with open(unresolved_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(unresolved_rows)

    print(f"manual labeled rows: {len(labeled_rows)} -> {labeled_path}")
    print(f"manual unresolved rows: {len(unresolved_rows)} -> {unresolved_path}")
    if unknown_folders:
        print(f"WARNING: folders with no mapping (left blank): {sorted(unknown_folders)}")


if __name__ == "__main__":
    main()
