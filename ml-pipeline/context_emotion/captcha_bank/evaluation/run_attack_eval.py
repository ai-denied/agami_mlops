#!/usr/bin/env python3
"""
어태커 프록시 모델 평가 스크립트.

학습된 model.joblib을 captcha_pool.csv에 적용해 보안 지표를 계산하고
evaluation_result.json을 저장한다.

사용법:
    python -m context_emotion.captcha_bank.evaluation.run_attack_eval \\
        --pool-csv  /path/to/captcha_pool.csv \\
        --model     /path/to/model.joblib \\
        --output    evaluation_result.json \\
        --version   v1_20260701
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from context_emotion.captcha_bank.choice_generation import EMOTIONS, load_rows
from context_emotion.captcha_bank.evaluation.metrics import compute_eval_metrics
from context_emotion.captcha_bank.training.features import (
    build_attacker_matrix,
)

_W = 60


def _predict(bundle: dict, rows: list[dict]) -> list[str]:
    attacker = bundle["emotion_attacker"]
    clf: object = attacker["model"]
    le = attacker["label_encoder"]
    X = build_attacker_matrix(rows, seed=42)
    preds_idx = clf.predict(X)
    return [le.inverse_transform([i])[0] for i in preds_idx]


def evaluate(pool_csv: Path, model_path: Path, output: Path, version: str) -> dict:
    print("═" * _W)
    print(f"  captcha_bank 보안 평가  [{version}]")
    print("═" * _W)

    rows = [r for r in load_rows(pool_csv) if r.get("final_emotion", "") in EMOTIONS]
    print(f"  유효 문항 수: {len(rows)}")
    if not rows:
        raise ValueError(f"유효한 문항이 없습니다: {pool_csv}")

    print(f"  모델 로드: {model_path}")
    bundle = joblib.load(model_path)

    print("  어태커 예측 중 ...")
    preds = _predict(bundle, rows)

    print("  보안 지표 계산 중 ...")
    result = compute_eval_metrics(rows, preds, version, pool_csv=str(pool_csv))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    _print_summary(result)
    print(f"\n  → {output}")
    return result


def _print_summary(r: dict) -> None:
    print()
    print(f"  {'지표':<32} {'값':>10}")
    print(f"  {'-'*44}")
    print(f"  {'pool_size':<32} {r['pool_size']:>10}")
    print(f"  {'attacker_pass_rate':<32} {r['attacker_pass_rate']:>10.4f}")
    print(f"  {'robust_rate':<32} {r['robust_rate']:>10.4f}")
    print(f"  {'ambiguous_rate':<32} {r['ambiguous_rate']:>10.4f}")
    print(f"  {'choice_policy_pass_rate (3q)':<32} {r['choice_policy_pass_rate']:>10.4f}")
    print(f"  {'macro_f1_attacker':<32} {r['macro_f1_attacker']:>10.4f}")
    eligible = r.get("promotion_eligible", False)
    print(f"\n  promotion_eligible: {'✓ YES' if eligible else '✗ NO'}")
    if r.get("vlm_attacker_stats"):
        print("\n  VLM 어태커 3문제 통과율:")
        for name, st in r["vlm_attacker_stats"].items():
            print(f"    {name:<16} {st['three_q_pass_rate']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPTCHA 풀 보안 평가")
    ap.add_argument("--pool-csv", type=Path, required=True)
    ap.add_argument("--model",    type=Path, required=True, help="train_attack_model.py 출력 model.joblib")
    ap.add_argument("--output",   type=Path, default=Path("evaluation_result.json"))
    ap.add_argument("--version",  required=True)
    args = ap.parse_args()
    evaluate(args.pool_csv, args.model, args.output, args.version)


if __name__ == "__main__":
    main()
