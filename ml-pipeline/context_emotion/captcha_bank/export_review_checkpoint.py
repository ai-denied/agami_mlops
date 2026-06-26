#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

MERGED_LABELS = {
    "happiness": "positive",
    "affection": "positive",
    "confidence": "positive",
    "anticipation": "positive",
    "calm": "calm",
    "sadness": "distress",
    "suffering": "distress",
    "disconnection": "distress",
    "embarrassment": "distress",
    "fear": "threat",
    "anger": "threat",
    "aversion": "threat",
    "confusion": "confusion",
    "yearning": "yearning",
}

MERGED_GUIDE = {
    "positive": "positive social or goal-oriented emotion: happiness, affection, confidence, anticipation",
    "calm": "peaceful, relaxed, neutral, safe, or content",
    "distress": "sadness, suffering, isolation, shame, exhaustion, or emotional pain",
    "threat": "fear, anger, disgust, hostility, danger, or rejection",
    "confusion": "uncertainty, puzzlement, not understanding the situation",
    "yearning": "longing, missing, hoping for someone or something absent",
}

MODEL_KEYS = ("qwen25_vl_3b", "smolvlm2_2b")


def merged(label: str) -> str:
    return MERGED_LABELS.get((label or "").strip(), "")


def parse_json_list(text: str) -> list[str]:
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def read_queue(path: Path) -> tuple[dict[str, dict], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = {row["sample_id"]: row for row in reader}
        return rows, list(reader.fieldnames or [])


def read_reviews(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(value.get("sample_id", ""))
            if not sid:
                continue
            if value.get("human_decision") == "__undo__":
                completed.pop(sid, None)
            else:
                completed[sid] = value
    return completed


def model_features(model_predictions: str) -> dict[str, str]:
    features: dict[str, str] = {}
    try:
        predictions = json.loads(model_predictions or "{}")
    except json.JSONDecodeError:
        predictions = {}
    for key in MODEL_KEYS:
        value = predictions.get(key) or {}
        emotion = str(value.get("emotion", ""))
        features[f"{key}_emotion"] = emotion
        features[f"{key}_merged_emotion"] = merged(emotion)
        features[f"{key}_confidence"] = str(value.get("confidence", ""))
        features[f"{key}_ambiguous"] = str(bool(value.get("ambiguous", False)))
        features[f"{key}_evidence"] = str(value.get("evidence", value.get("visual_evidence", "")))
    return features


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--reviews", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    queue_rows, queue_fields = read_queue(args.queue)
    completed = read_reviews(args.reviews)

    checkpoint_rows: list[dict] = []
    ai_rows: list[dict] = []
    transition = Counter()
    merged_transition = Counter()
    decisions = Counter()
    human_labels = Counter()
    human_merged = Counter()
    provisional_labels = Counter()
    model_agreement = Counter()

    for sid, review in completed.items():
        queue = queue_rows.get(sid)
        if not queue:
            continue
        decision = str(review.get("human_decision", ""))
        human_emotion = str(review.get("human_emotion", ""))
        provisional_emotion = str(queue.get("provisional_emotion", ""))
        aux_emotions = parse_json_list(str(review.get("human_aux_emotions", "[]")))
        human_merged_label = merged(human_emotion)
        provisional_merged_label = merged(provisional_emotion)
        features = model_features(queue.get("model_predictions", ""))

        row = dict(queue)
        row.update({
            "human_decision": decision,
            "human_emotion": human_emotion,
            "human_aux_emotions": json.dumps(aux_emotions, ensure_ascii=False),
            "human_confidence": str(review.get("human_confidence", "")),
            "human_note": str(review.get("human_note", "")),
            "reviewer_id": str(review.get("reviewer_id", "")),
            "provisional_merged_emotion": provisional_merged_label,
            "human_merged_emotion": human_merged_label,
            "human_aux_merged_emotions": json.dumps(sorted({merged(e) for e in aux_emotions if merged(e)}), ensure_ascii=False),
        })
        row.update(features)
        for key in MODEL_KEYS:
            row[f"{key}_agrees_human"] = str(features.get(f"{key}_emotion") == human_emotion and bool(human_emotion))
            row[f"{key}_agrees_human_merged"] = str(features.get(f"{key}_merged_emotion") == human_merged_label and bool(human_merged_label))
        checkpoint_rows.append(row)

        decisions[decision] += 1
        provisional_labels[provisional_emotion] += 1
        if human_emotion:
            human_labels[human_emotion] += 1
        if human_merged_label:
            human_merged[human_merged_label] += 1
        if human_emotion and provisional_emotion != human_emotion:
            transition[(provisional_emotion, human_emotion)] += 1
        if human_merged_label and provisional_merged_label != human_merged_label:
            merged_transition[(provisional_merged_label, human_merged_label)] += 1
        for key in MODEL_KEYS:
            if human_emotion and features.get(f"{key}_emotion") == human_emotion:
                model_agreement[f"{key}_exact"] += 1
            if human_merged_label and features.get(f"{key}_merged_emotion") == human_merged_label:
                model_agreement[f"{key}_merged"] += 1

        if decision in {"accept", "relabel"} and human_merged_label:
            ai_rows.append({
                "sample_id": sid,
                "image_path": queue.get("image_path", ""),
                "target_person_bbox": queue.get("target_person_bbox", ""),
                "source": queue.get("source", ""),
                "source_image_id": queue.get("source_image_id", ""),
                "provisional_emotion": provisional_emotion,
                "provisional_merged_emotion": provisional_merged_label,
                "human_emotion": human_emotion,
                "human_aux_emotions": aux_emotions,
                "human_merged_emotion": human_merged_label,
                "human_aux_merged_emotions": sorted({merged(e) for e in aux_emotions if merged(e)}),
                "human_confidence": review.get("human_confidence", ""),
                "attack_hardness": queue.get("attack_hardness", ""),
                "model_predictions": json.loads(queue.get("model_predictions") or "{}"),
                "merged_label_guide": MERGED_GUIDE,
                "task": "Re-validate whether the human_merged_emotion is supported by the image context. Return ai_status as ai_agree, ai_disagree, or ai_ambiguous, plus suggested_merged_emotion and brief evidence.",
            })

    out = args.out_dir
    checkpoint_fields = queue_fields + [
        "human_aux_emotions",
        "reviewer_id",
        "provisional_merged_emotion",
        "human_merged_emotion",
        "human_aux_merged_emotions",
    ]
    for key in MODEL_KEYS:
        checkpoint_fields += [
            f"{key}_emotion",
            f"{key}_merged_emotion",
            f"{key}_confidence",
            f"{key}_ambiguous",
            f"{key}_evidence",
            f"{key}_agrees_human",
            f"{key}_agrees_human_merged",
        ]
    checkpoint_fields = list(dict.fromkeys(checkpoint_fields))
    write_csv(out / "review_checkpoint.csv", checkpoint_rows, checkpoint_fields)

    write_csv(
        out / "review_label_transitions.csv",
        [{"provisional_emotion": p, "human_emotion": h, "count": n} for (p, h), n in transition.most_common()],
        ["provisional_emotion", "human_emotion", "count"],
    )
    write_csv(
        out / "review_merged_label_transitions.csv",
        [{"provisional_merged_emotion": p, "human_merged_emotion": h, "count": n} for (p, h), n in merged_transition.most_common()],
        ["provisional_merged_emotion", "human_merged_emotion", "count"],
    )

    with (out / "ai_revalidation_input.jsonl").open("w", encoding="utf-8") as f:
        for row in ai_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = []
    report.append("# Human Review Checkpoint Report")
    report.append("")
    report.append(f"- queue rows: {len(queue_rows)}")
    report.append(f"- completed unique reviews: {len(checkpoint_rows)}")
    report.append(f"- AI revalidation candidates: {len(ai_rows)}")
    report.append("")
    report.append("## Decision Distribution")
    report.extend(f"- {k}: {v}" for k, v in decisions.most_common())
    report.append("")
    report.append("## Human Emotion Distribution")
    report.extend(f"- {k}: {v}" for k, v in human_labels.most_common())
    report.append("")
    report.append("## Human Merged Label Distribution")
    report.extend(f"- {k}: {v}" for k, v in human_merged.most_common())
    report.append("")
    report.append("## Top Provisional -> Human Transitions")
    report.extend(f"- {p} -> {h}: {n}" for (p, h), n in transition.most_common(40))
    report.append("")
    report.append("## Top Merged Transitions")
    report.extend(f"- {p} -> {h}: {n}" for (p, h), n in merged_transition.most_common(40))
    report.append("")
    report.append("## Model Agreement With Human Labels")
    for key in MODEL_KEYS:
        exact = model_agreement[f"{key}_exact"]
        merged_count = model_agreement[f"{key}_merged"]
        denom = max(1, sum(1 for row in checkpoint_rows if row.get("human_emotion")))
        report.append(f"- {key} exact: {exact}/{denom} ({exact / denom:.1%})")
        report.append(f"- {key} merged: {merged_count}/{denom} ({merged_count / denom:.1%})")
    report.append("")
    report.append("## Merged Label Guide")
    for label, guide in MERGED_GUIDE.items():
        members = ", ".join(k for k, v in MERGED_LABELS.items() if v == label)
        report.append(f"- {label}: {members} — {guide}")
    (out / "review_confusion_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"wrote {out / 'review_checkpoint.csv'} rows={len(checkpoint_rows)}")
    print(f"wrote {out / 'review_confusion_report.md'}")
    print(f"wrote {out / 'review_label_transitions.csv'}")
    print(f"wrote {out / 'review_merged_label_transitions.csv'}")
    print(f"wrote {out / 'ai_revalidation_input.jsonl'} rows={len(ai_rows)}")


if __name__ == "__main__":
    main()
