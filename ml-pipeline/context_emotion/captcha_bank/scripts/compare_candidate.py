#!/usr/bin/env python3
"""
후보(candidate) vs. 현재(current) 보안 지표 비교 및 승격 게이트 판단.

비교 대상:
    model-store/captcha_bank/candidates/{version}/evaluation_result.json
    model-store/captcha_bank/current/evaluation_result.json  (없으면 절대값 게이트만 적용)

종료 코드:
    0 — 승격 허용 (모든 게이트 통과)
    1 — 게이트 실패 (승격 거부)

사용법:
    python -m context_emotion.captcha_bank.scripts.compare_candidate \\
        --version v1_20260701 [--policy config/promotion_policy.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml  # PyYAML

_SCRIPT_DIR       = Path(__file__).resolve().parent
_ML_PIPELINE_ROOT = _SCRIPT_DIR.parents[3]
_STORE            = _ML_PIPELINE_ROOT / "model-store" / "captcha_bank"
_CANDIDATES_DIR   = _STORE / "candidates"
_CURRENT_DIR      = _STORE / "current"
_DEFAULT_POLICY   = _SCRIPT_DIR.parent / "config" / "promotion_policy.yaml"

_W = 60


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_policy(policy_path: Path) -> dict:
    with policy_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def compare(
    version: str,
    policy_path: Path,
    candidates_dir: Path = _CANDIDATES_DIR,
    current_dir: Path = _CURRENT_DIR,
) -> bool:
    print("═" * _W)
    print(f"  captcha_bank 후보 비교  [{version}]")
    print("═" * _W)

    candidate_eval = _load_json(candidates_dir / version / "evaluation_result.json")
    current_eval   = _load_json(current_dir / "evaluation_result.json")
    policy         = _load_policy(policy_path)
    gates          = policy.get("gates", {})

    if candidate_eval is None:
        print(f"  ✗ 후보 evaluation_result.json 없음: {candidates_dir / version}")
        return False

    print(f"  후보 버전: {candidate_eval.get('version', '?')}")
    if current_eval:
        print(f"  현재 버전: {current_eval.get('version', '?')}")
    else:
        print("  현재 모델 없음 — 절대값 게이트만 적용")

    results: list[tuple[str, bool, str]] = []

    # ── 게이트 1: 풀 규모 ─────────────────────────────────────────────────
    min_size = gates.get("min_pool_size", 200)
    pool_size = candidate_eval.get("pool_size", 0) or 0
    results.append((
        f"pool_size >= {min_size}",
        pool_size >= min_size,
        f"{pool_size}",
    ))

    # ── 게이트 2: attacker_pass_rate ──────────────────────────────────────
    max_apr = gates.get("max_attacker_pass_rate", 0.35)
    apr = candidate_eval.get("attacker_pass_rate") or 1.0
    results.append((
        f"attacker_pass_rate <= {max_apr}",
        apr <= max_apr,
        f"{apr:.4f}",
    ))

    # ── 게이트 3: robust_rate ─────────────────────────────────────────────
    min_rr = gates.get("min_robust_rate", 0.65)
    rr = candidate_eval.get("robust_rate") or 0.0
    results.append((
        f"robust_rate >= {min_rr}",
        rr >= min_rr,
        f"{rr:.4f}",
    ))

    # ── 게이트 4: choice_policy_pass_rate ────────────────────────────────
    max_cppr = gates.get("max_choice_policy_pass_rate", 0.10)
    cppr = candidate_eval.get("choice_policy_pass_rate") or 1.0
    results.append((
        f"choice_policy_pass_rate <= {max_cppr}",
        cppr <= max_cppr,
        f"{cppr:.4f}",
    ))

    # ── 게이트 5: ambiguous_rate ──────────────────────────────────────────
    max_amb = gates.get("max_ambiguous_rate", 0.20)
    amb = candidate_eval.get("ambiguous_rate") or 0.0
    results.append((
        f"ambiguous_rate <= {max_amb}",
        amb <= max_amb,
        f"{amb:.4f}",
    ))

    # ── 게이트 6: 현재 대비 pass_rate 악화 폭 ────────────────────────────
    if current_eval and current_eval.get("attacker_pass_rate") is not None:
        max_increase = gates.get("max_pass_rate_increase_vs_current", 0.05)
        current_apr = float(current_eval["attacker_pass_rate"])
        increase = apr - current_apr
        results.append((
            f"pass_rate increase <= {max_increase} vs current ({current_apr:.4f})",
            increase <= max_increase,
            f"Δ={increase:+.4f}",
        ))

    # ── 게이트 7: 감정 클래스 다양성 ─────────────────────────────────────
    from context_emotion.captcha_bank.choice_generation import EMOTIONS
    max_zero = gates.get("max_zero_class_emotions", 2)
    dist = candidate_eval.get("label_distribution") or {}
    zero_classes = [e for e in EMOTIONS if (dist.get(e) or 0) == 0]
    results.append((
        f"zero-class emotions <= {max_zero}",
        len(zero_classes) <= max_zero,
        f"{len(zero_classes)} zero: {zero_classes}" if zero_classes else "0",
    ))

    # ── 결과 출력 ────────────────────────────────────────────────────────
    print()
    all_pass = True
    for label, passed, value in results:
        icon = "✓" if passed else "✗"
        print(f"  {icon} {label:<50} {value}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  └─ ✓ 모든 게이트 통과 — 승격 허용")
    else:
        failed = [label for label, passed, _ in results if not passed]
        print(f"  └─ ✗ 게이트 실패 — 승격 거부")
        print(f"     실패 게이트: {failed}")

    return all_pass


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPTCHA bank 후보 vs. 현재 비교")
    ap.add_argument("--version",  required=True)
    ap.add_argument("--policy",   type=Path, default=_DEFAULT_POLICY)
    ap.add_argument("--candidates-dir", type=Path, default=_CANDIDATES_DIR)
    ap.add_argument("--current-dir",    type=Path, default=_CURRENT_DIR)
    args = ap.parse_args()

    ok = compare(args.version, args.policy, args.candidates_dir, args.current_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
