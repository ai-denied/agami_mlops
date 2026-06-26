#!/usr/bin/env python3
"""
captcha_bank 후보 → current 승격 스크립트.

승격 절차:
  1. candidates/{version}/ 파일 존재 확인
  2. current/ → archive/{timestamp}_{old_version}/ 백업
  3. candidates/{version}/ → current_staging_{ts}/ 복사
  4. metadata.json 에 promoted_at / promoted_from_candidate 기록
  5. current_staging_{ts}/ → current/ 원자적 교체 (rename)
  6. 실패 시 current/ 자동 복원

사용법:
    python -m context_emotion.captcha_bank.scripts.promote_model \\
        --version v1_20260701
    python -m context_emotion.captcha_bank.scripts.promote_model \\
        --version v1_20260701 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR       = Path(__file__).resolve().parent
_ML_PIPELINE_ROOT = _SCRIPT_DIR.parents[3]
_STORE            = _ML_PIPELINE_ROOT / "model-store" / "captcha_bank"
_CANDIDATES_DIR   = _STORE / "candidates"
_CURRENT_DIR      = _STORE / "current"
_ARCHIVE_DIR      = _STORE / "archive"

REQUIRED_FILES = ["model.joblib", "metadata.json", "evaluation_result.json", "captcha_pool.csv"]

_W = 60


def _log(msg: str, dry: bool = False) -> None:
    prefix = "[DRY-RUN] " if dry else ""
    print(f"  {prefix}{msg}")


def _validate_candidate(candidate_dir: Path) -> None:
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"후보 폴더 없음: {candidate_dir}")
    missing = [f for f in REQUIRED_FILES if not (candidate_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"필수 파일 없음: {missing}  (경로: {candidate_dir})")


def promote(version: str, dry: bool = False) -> bool:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("═" * _W)
    print(f"  captcha_bank 모델 승격  [{version}]")
    print("═" * _W)

    candidate_dir = _CANDIDATES_DIR / version

    # ── 1. 후보 검증 ────────────────────────────────────────────────────────
    _log("후보 파일 확인 중 ...", dry)
    _validate_candidate(candidate_dir)
    _log(f"✓ 후보 검증 완료: {candidate_dir}", dry)

    # ── 2. 현재 모델 아카이브 ─────────────────────────────────────────────
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    old_version = "unknown"
    if _CURRENT_DIR.is_dir():
        meta_path = _CURRENT_DIR / "metadata.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                old_version = json.load(f).get("version", "unknown")
        archive_dst = _ARCHIVE_DIR / f"{ts}_{old_version}"
        _log(f"현재 모델 아카이브: {archive_dst}", dry)
        if not dry:
            shutil.copytree(_CURRENT_DIR, archive_dst)
        _log(f"✓ 아카이브 완료", dry)
    else:
        _log("현재 모델 없음 — 아카이브 건너뜀", dry)

    # ── 3. staging 복사 ───────────────────────────────────────────────────
    staging = _STORE / f"current_staging_{ts}"
    _log(f"staging 복사: {candidate_dir} → {staging}", dry)
    if not dry:
        shutil.copytree(candidate_dir, staging)

    # ── 4. metadata 업데이트 ─────────────────────────────────────────────
    _log("metadata.json 업데이트", dry)
    if not dry:
        meta_path = staging / "metadata.json"
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        meta["promoted_at"]               = ts
        meta["promoted_from_candidate"]   = str(candidate_dir)
        meta["previous_version"]          = old_version
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ── 5. 원자적 교체 ───────────────────────────────────────────────────
    _log("current/ 원자적 교체 ...", dry)
    old_temp = _STORE / f"_current_old_{ts}"
    if not dry:
        try:
            if _CURRENT_DIR.is_dir():
                _CURRENT_DIR.rename(old_temp)
            staging.rename(_CURRENT_DIR)
            if old_temp.is_dir():
                shutil.rmtree(old_temp)
        except Exception as e:
            # 롤백
            if staging.is_dir() and not _CURRENT_DIR.is_dir():
                staging.rename(_CURRENT_DIR)
            if old_temp.is_dir():
                old_temp.rename(_CURRENT_DIR)
            raise RuntimeError(f"교체 실패 — 롤백 완료: {e}") from e

    _log(f"✓ 승격 완료: {version} → current/", dry)
    print()
    print(f"  └─ ✓ 승격 성공  [{version}]")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPTCHA bank 모델 승격")
    ap.add_argument("--version",  required=True)
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--store",    type=Path, default=_STORE,
                    help=f"모델 스토어 루트 (기본: {_STORE})")
    args = ap.parse_args()

    global _STORE, _CANDIDATES_DIR, _CURRENT_DIR, _ARCHIVE_DIR
    _STORE          = args.store
    _CANDIDATES_DIR = _STORE / "candidates"
    _CURRENT_DIR    = _STORE / "current"
    _ARCHIVE_DIR    = _STORE / "archive"

    try:
        ok = promote(args.version, dry=args.dry_run)
    except Exception as e:
        print(f"\n  └─ ✗ 승격 실패: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
