#!/usr/bin/env python3
"""
학습·평가 결과물을 모델 스토어 후보 디렉토리에 패키징한다.

model-store/captcha_bank/candidates/{version}/
├── model.joblib              sklearn 듀얼 어태커 프록시 모델
├── metadata.json             버전/풀 정보 + 학습 지표 요약
├── evaluation_result.json    보안 평가 지표
├── captcha_pool.csv          이 버전을 생성하는 데 쓰인 풀
├── choice_policy_report.md   4지선다 정책 리포트 (선택)
└── manifest.json             파일 sha256 목록

사용법:
    python -m context_emotion.captcha_bank.scripts.package_model \\
        --version v1_20260701 \\
        --model-joblib /path/to/model.joblib \\
        --eval-json    /path/to/evaluation_result.json \\
        --pool-csv     /path/to/captcha_pool.csv \\
        [--policy-md   /path/to/choice_policy_report.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

_SCRIPT_DIR       = Path(__file__).resolve().parent
_ML_PIPELINE_ROOT = _SCRIPT_DIR.parents[3]   # ml-pipeline/
_STORE            = _ML_PIPELINE_ROOT / "model-store" / "captcha_bank"
_CANDIDATES_DIR   = _STORE / "candidates"

_W = 60


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def package(
    version: str,
    model_joblib: Path,
    eval_json: Path,
    pool_csv: Path,
    policy_md: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    print("═" * _W)
    print(f"  captcha_bank 패키징  [{version}]")
    print("═" * _W)

    out = (output_dir or _CANDIDATES_DIR) / version
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # ── 모델 번들에서 metadata 추출 ─────────────────────────────────────────
    bundle = joblib.load(model_joblib)
    with eval_json.open(encoding="utf-8") as f:
        eval_result = json.load(f)

    attacker_meta = {k: v for k, v in bundle.get("emotion_attacker", {}).items()
                     if k != "model"}
    ranker_meta   = {k: v for k, v in (bundle.get("security_ranker") or {}).items()
                     if k != "model"}

    metadata = {
        "version":             version,
        "packaged_at":         datetime.now(timezone.utc).isoformat(),
        "pool_size":           bundle.get("pool_size"),
        "pool_sha256":         bundle.get("pool_sha256"),
        "trained_at":          bundle.get("trained_at"),
        "emotion_attacker":    attacker_meta,
        "security_ranker":     ranker_meta or None,
        "summary": {
            "attacker_pass_rate":      eval_result.get("attacker_pass_rate"),
            "robust_rate":             eval_result.get("robust_rate"),
            "ambiguous_rate":          eval_result.get("ambiguous_rate"),
            "choice_policy_pass_rate": eval_result.get("choice_policy_pass_rate"),
            "macro_f1_attacker":       eval_result.get("macro_f1_attacker"),
            "promotion_eligible":      eval_result.get("promotion_eligible"),
        },
    }

    # ── 파일 복사 ────────────────────────────────────────────────────────────
    files_to_copy = {
        "model.joblib":           model_joblib,
        "evaluation_result.json": eval_json,
        "captcha_pool.csv":       pool_csv,
    }
    if policy_md and policy_md.exists():
        files_to_copy["choice_policy_report.md"] = policy_md

    manifest = {}
    for dst_name, src in files_to_copy.items():
        dst = out / dst_name
        shutil.copy2(src, dst)
        manifest[dst_name] = _sha256(dst)
        print(f"  ✓ {dst_name}")

    # ── metadata.json ────────────────────────────────────────────────────────
    meta_path = out / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    manifest["metadata.json"] = _sha256(meta_path)
    print("  ✓ metadata.json")

    # ── manifest.json ────────────────────────────────────────────────────────
    manifest_obj = {"version": version, "files": manifest}
    manifest_path = out / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, ensure_ascii=False, indent=2)
    print("  ✓ manifest.json")

    print(f"\n  → {out}/")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPTCHA bank 후보 패키징")
    ap.add_argument("--version",      required=True)
    ap.add_argument("--model-joblib", type=Path, required=True)
    ap.add_argument("--eval-json",    type=Path, required=True)
    ap.add_argument("--pool-csv",     type=Path, required=True)
    ap.add_argument("--policy-md",    type=Path, default=None)
    ap.add_argument("--output-dir",   type=Path, default=None,
                    help=f"기본값: {_CANDIDATES_DIR}")
    args = ap.parse_args()

    package(
        args.version, args.model_joblib, args.eval_json,
        args.pool_csv, args.policy_md, args.output_dir,
    )


if __name__ == "__main__":
    main()
