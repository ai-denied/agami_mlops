"""Copy only the images referenced by context_emotion_train_dataset_v1.csv
into a small, self-contained folder for upload (e.g. to Google Drive).

The raw context_emotion data under /workspace/data/context_emotion/ has
~35k files across several duplicated EMOTIC mirrors; the v1 training set
only actually uses 6,706 of them. This copies just those files (renamed to
sample_id to avoid collisions across sources) plus a CSV whose image_path
points at the new, flat layout.

Usage:
    python export_train_dataset_v1.py \
        --train-csv /workspace/data/context_emotion/processed/context_emotion_train_dataset_v1.csv \
        --out-dir /workspace/data/context_emotion/export_v1
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    with open(args.train_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    missing = []
    for r in rows:
        src = r["image_path"]
        ext = os.path.splitext(src)[1].lower() or ".jpg"
        new_name = f"{r['sample_id']}{ext}"
        dst = os.path.join(images_dir, new_name)
        if not os.path.isfile(src):
            missing.append(r["sample_id"])
            r["image_path"] = ""
            continue
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        r["image_path"] = f"images/{new_name}"

    out_csv = os.path.join(args.out_dir, os.path.basename(args.train_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"copied {len(rows) - len(missing)} images -> {images_dir}")
    if missing:
        print(f"WARNING: {len(missing)} rows had no source file at export time: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
