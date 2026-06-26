#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from context_emotion.captcha_bank.common import atomic_write_csv, read_csv

_HERE = Path(__file__).resolve().parent


def load_completed_reviews(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = value.get("sample_id")
            if not sid:
                continue
            sid = str(sid)
            if value.get("human_decision") == "__undo__":
                completed.pop(sid, None)
            else:
                completed[sid] = value
    return completed


def main() -> None:
    p = argparse.ArgumentParser(
        description="인간 검수 완료 결과로부터 최종 CAPTCHA 풀 CSV를 생성한다."
    )
    p.add_argument("--queue", type=Path, default=_HERE / "artifacts" / "human_review_queue.csv")
    p.add_argument("--reviews", type=Path, default=_HERE / "artifacts" / "human_reviews.jsonl")
    p.add_argument("--output", type=Path, default=_HERE / "artifacts" / "captcha_pool.csv")
    a = p.parse_args()

    rows, fields = read_csv(a.queue)
    reviews = load_completed_reviews(a.reviews)
    output = []
    for row in rows:
        review = reviews.get(row["sample_id"])
        if not review or review.get("human_decision") not in {"accept", "relabel"}:
            continue
        emotion = (
            row["provisional_emotion"]
            if review["human_decision"] == "accept"
            else review.get("human_emotion", "")
        )
        if emotion:
            output.append({
                **row,
                **review,
                "final_emotion": emotion,
                "approved_at": datetime.now(timezone.utc).isoformat(),
            })

    out_fields = fields + ["human_aux_emotions", "final_emotion", "approved_at"]
    atomic_write_csv(a.output, output, out_fields)
    print(f"captcha_pool={len(output)} output={a.output}")


if __name__ == "__main__":
    main()
