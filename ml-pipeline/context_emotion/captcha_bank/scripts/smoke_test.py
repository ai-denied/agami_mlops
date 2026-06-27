#!/usr/bin/env python3
"""
current/ 모델 스모크 테스트.

model-store/captcha_bank/current/ 의 model.joblib을 로드하고
captcha_pool.csv에서 5개 샘플을 뽑아 추론이 정상 동작하는지 확인한다.

사용법:
    python -m context_emotion.captcha_bank.scripts.smoke_test
    python -m context_emotion.captcha_bank.scripts.smoke_test --current-dir /path/to/current
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR       = Path(__file__).resolve().parent
_ML_PIPELINE_ROOT = _SCRIPT_DIR.parents[2]   # ml-pipeline/
_CURRENT_DIR      = _ML_PIPELINE_ROOT / "model-store" / "captcha_bank" / "current"

_W = 60
_SAMPLE_N = 5


def smoke_test(current_dir: Path) -> bool:
    print("═" * _W)
    print("  captcha_bank 스모크 테스트")
    print("═" * _W)

    checks: list[tuple[str, bool, str]] = []

    # ── 1. 필수 파일 존재 ────────────────────────────────────────────────
    required = ["model.joblib", "metadata.json", "evaluation_result.json", "captcha_pool.csv"]
    for fname in required:
        exists = (current_dir / fname).exists()
        checks.append((f"파일 존재: {fname}", exists, "" if exists else "없음"))

    all_exist = all(p for _, p, _ in checks)
    if not all_exist:
        _print_checks(checks)
        return False

    # ── 2. metadata.json 파싱 ────────────────────────────────────────────
    try:
        with (current_dir / "metadata.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        version = meta.get("version", "?")
        checks.append(("metadata.json 파싱", True, f"version={version}"))
    except Exception as e:
        checks.append(("metadata.json 파싱", False, str(e)))
        _print_checks(checks)
        return False

    # ── 3. evaluation_result.json 핵심 필드 ─────────────────────────────
    try:
        with (current_dir / "evaluation_result.json").open(encoding="utf-8") as f:
            er = json.load(f)
        apr = er.get("attacker_pass_rate")
        checks.append((
            "evaluation_result.json",
            apr is not None,
            f"attacker_pass_rate={apr}",
        ))
    except Exception as e:
        checks.append(("evaluation_result.json 파싱", False, str(e)))
        _print_checks(checks)
        return False

    # ── 4. model.joblib 로드 + 추론 ─────────────────────────────────────
    try:
        import joblib
        bundle = joblib.load(current_dir / "model.joblib")
        checks.append(("model.joblib 로드", True, f"keys={list(bundle.keys())}"))
    except Exception as e:
        checks.append(("model.joblib 로드", False, str(e)))
        _print_checks(checks)
        return False

    try:
        from context_emotion.captcha_bank.choice_generation import EMOTIONS, load_rows
        from context_emotion.captcha_bank.training.features import build_attacker_matrix

        rows = load_rows(current_dir / "captcha_pool.csv")
        sample = [r for r in rows if r.get("final_emotion", "") in EMOTIONS][:_SAMPLE_N]

        if not sample:
            checks.append(("샘플 추론", False, "유효 샘플 없음"))
            _print_checks(checks)
            return False

        clf = bundle["emotion_attacker"]["model"]
        le  = bundle["emotion_attacker"]["label_encoder"]
        X   = build_attacker_matrix(sample, seed=42)
        preds_idx = clf.predict(X)
        preds = [le.inverse_transform([i])[0] for i in preds_idx]
        all_valid = all(p in EMOTIONS for p in preds)
        checks.append((
            f"{_SAMPLE_N}개 샘플 추론",
            all_valid,
            f"preds={preds[:3]}..." if len(preds) > 3 else f"preds={preds}",
        ))
    except Exception as e:
        checks.append(("샘플 추론", False, str(e)))
        _print_checks(checks)
        return False

    _print_checks(checks)
    all_pass = all(p for _, p, _ in checks)
    print()
    if all_pass:
        print("  └─ ✓ 스모크 테스트 통과")
    else:
        print("  └─ ✗ 스모크 테스트 실패")
    return all_pass


def _print_checks(checks: list[tuple[str, bool, str]]) -> None:
    print()
    for label, passed, note in checks:
        icon = "✓" if passed else "✗"
        suffix = f"  ({note})" if note else ""
        print(f"  {icon} {label}{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="captcha_bank current 스모크 테스트")
    ap.add_argument("--current-dir", type=Path, default=_CURRENT_DIR)
    args = ap.parse_args()

    ok = smoke_test(args.current_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
