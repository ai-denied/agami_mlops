#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from context_emotion.captcha_bank.choice_generation import (
    choice_credit,
    generate_choices,
    load_rows,
    parse_aux,
)

ATTACKERS = {
    "qwen": "qwen_emotion",
    "smolvlm": "smolvlm_emotion",
    "self_attack": "self_attack_emotion",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="4지선다 CAPTCHA 선택지 정책 리포트를 생성한다."
    )
    p.add_argument("--input-csv", type=Path, required=True, help="scored dual CSV (final_emotion 열 포함)")
    p.add_argument("--output-csv", type=Path, required=True, help="선택지 포함 결과 CSV 저장 경로")
    p.add_argument("--output-md", type=Path, required=True, help="정책 리포트 Markdown 저장 경로")
    p.add_argument("--threshold", type=float, default=2.5, help="3문제 통과 기준 점수 (기본: 2.5)")
    return p.parse_args()


def attacker_points(row: dict, attacker_col: str) -> float:
    pred = row.get(attacker_col, "")
    choices = json.loads(row["choices"])
    if pred not in choices:
        return 0.0
    return choice_credit(pred, row)


def estimate_pass(rows: list[dict], attacker_col: str, threshold: float):
    counts = Counter(attacker_points(row, attacker_col) for row in rows)
    total = len(rows)
    dist = {0.0: 1.0}
    for _ in range(3):
        nxt: Counter = Counter()
        for score, prob in dist.items():
            for points, count in counts.items():
                nxt[score + points] += prob * (count / total)
        dist = dict(nxt)
    return sum(prob for score, prob in dist.items() if score >= threshold), counts


def main():
    args = parse_args()
    rows = load_rows(args.input_csv)
    output_rows = []
    for row in rows:
        choices = generate_choices(row)
        row = dict(row)
        row["choices"] = json.dumps(choices, ensure_ascii=False)
        row["choice_count"] = str(len(choices))
        row["has_aux_choice"] = str(any(c in parse_aux(row.get("aux_emotions", "[]")) for c in choices))
        for name, col in ATTACKERS.items():
            pred = row.get(col, "")
            row[f"{name}_choice_points"] = str(attacker_points(row, col))
            row[f"{name}_choice_in_options"] = str(pred in choices)
        output_rows.append(row)

    if not output_rows:
        print("입력 CSV가 비어 있습니다.")
        return

    fields = list(output_rows[0].keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    lines = ["# 4-Choice CAPTCHA Policy Report", ""]
    lines.append("## Policy")
    lines.append("- Each question shows 4 emotion choices.")
    lines.append("- `final_emotion` is always included.")
    lines.append("- Human auxiliary emotions and attacker wrong labels are preferred as distractors.")
    lines.append("- Primary answer gives 1.0 point, auxiliary answer gives 0.5 point, other choices give 0.0 point.")
    lines.append(f"- A 3-question challenge passes at total score >= {args.threshold}.")
    lines.append("")
    lines.append("## Pool")
    lines.append(f"- input: `{args.input_csv}`")
    lines.append(f"- output: `{args.output_csv}`")
    lines.append(f"- rows: {len(output_rows)}")
    lines.append(f"- security_grade: {dict(Counter(r.get('security_grade', '') for r in output_rows))}")
    lines.append(f"- answer_strength: {dict(Counter(r.get('answer_strength', '') for r in output_rows))}")
    lines.append(f"- rows with auxiliary choice included: {sum(r['has_aux_choice'] == 'True' for r in output_rows)}")
    lines.append("")
    lines.append("## Attacker Estimate Under 4 Choices")
    lines.append("If an attacker prediction is not one of the 4 choices, it receives 0 point in this estimate.")
    for name, col in ATTACKERS.items():
        if col not in output_rows[0]:
            continue
        pass_rate, counts = estimate_pass(output_rows, col, args.threshold)
        primary = sum(float(r[f"{name}_choice_points"]) == 1.0 for r in output_rows) / max(1, len(output_rows))
        partial = sum(float(r[f"{name}_choice_points"]) == 0.5 for r in output_rows) / max(1, len(output_rows))
        in_options = sum(r[f"{name}_choice_in_options"] == "True" for r in output_rows) / max(1, len(output_rows))
        lines.append(f"\n### {name}")
        lines.append(f"- prediction appears in choices: {in_options:.4%}")
        lines.append(f"- single-question primary rate: {primary:.4%}")
        lines.append(f"- single-question partial rate: {partial:.4%}")
        lines.append(f"- estimated 3-question pass rate: {pass_rate:.4%}")
        lines.append(f"- point counts: {dict(counts)}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is a conservative estimate for attackers that output one label directly.")
    lines.append("- A future VLM attacker could read all 4 choices and choose among them.")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_csv)
    print(args.output_md)


if __name__ == "__main__":
    main()
