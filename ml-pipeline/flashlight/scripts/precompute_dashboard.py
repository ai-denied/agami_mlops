#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raw/mouse_logs/dynamic/ 의 모든 세션 JSON을 ONNX 모델로 일괄 추론하여
dashboard_cache.json 을 생성한다.

Usage:
    python -m flashlight.scripts.precompute_dashboard
    python -m flashlight.scripts.precompute_dashboard \\
        --data-dir /workspace/data/flashlight \\
        --out /workspace/data/flashlight/dashboard_cache.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ML_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ML_PIPELINE_ROOT))

from flashlight.api import loader

BOT_TYPE_DISPLAY: Dict[str, str] = {
    "grid_search": "Grid Search",
    "known_target": "Known Target",
    "other_search": "AI Vision (GPT-Vision)",
    "random_search": "Random Search",
}

_DEFAULT_DATA_DIR = Path("/workspace/data/flashlight")
_DEFAULT_OUT = _DEFAULT_DATA_DIR / "dashboard_cache.json"


def _iter_sessions(dynamic_dir: Path):
    """dynamic/ 하위 human/* 과 bot/*/* 파일을 순회하며 dict 를 yield."""
    human_dir = dynamic_dir / "human"
    if human_dir.exists():
        for f in sorted(human_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            yield {
                "file": f.name,
                "source_type": "human",
                "bot_type": None,
                "label": int(data.get("label", 0)),
                "image_id": data.get("image_id"),
                "static_features": data.get("static_features", {}),
                "dynamic_features": data.get("dynamic_features", []),
            }

    bot_dir = dynamic_dir / "bot"
    if bot_dir.exists():
        for bt_dir in sorted(d for d in bot_dir.iterdir() if d.is_dir()):
            bot_type = bt_dir.name
            for f in sorted(bt_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                yield {
                    "file": f.name,
                    "source_type": "bot",
                    "bot_type": bot_type,
                    "label": int(data.get("label", 1)),
                    "image_id": data.get("image_id"),
                    "static_features": data.get("static_features", {}),
                    "dynamic_features": data.get("dynamic_features", []),
                }


def _count_files(dynamic_dir: Path) -> int:
    total = 0
    for subdir in (dynamic_dir / "human",):
        if subdir.exists():
            total += sum(1 for _ in subdir.glob("*.json"))
    bot_dir = dynamic_dir / "bot"
    if bot_dir.exists():
        for bt_dir in (d for d in bot_dir.iterdir() if d.is_dir()):
            total += sum(1 for _ in bt_dir.glob("*.json"))
    return total


def run(data_dir: Path, out_path: Path, model_dir: Optional[str]) -> None:
    if model_dir:
        os.environ["FLASHLIGHT_MODEL_DIR"] = model_dir

    print("모델 로드 중...", flush=True)
    loader.load_detector()
    detector = loader.get_detector()
    meta = loader.get_metadata()
    print(f"모델 버전: {meta.get('version', 'unknown')}", flush=True)

    dynamic_dir = data_dir / "raw" / "mouse_logs" / "dynamic"
    if not dynamic_dir.exists():
        print(f"ERROR: {dynamic_dir} 디렉토리 없음", file=sys.stderr)
        sys.exit(1)

    total_files = _count_files(dynamic_dir)
    print(f"총 {total_files:,}개 세션 처리 시작...", flush=True)

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, sess in enumerate(_iter_sessions(dynamic_dir), start=1):
        sample = {
            "static_features": sess["static_features"],
            "dynamic_features": sess["dynamic_features"],
        }
        try:
            pred = detector.predict_features(sample)
            risk_score = round(float(pred["bot_risk_score"]), 6)
            risk_band = pred["risk_band"]
        except Exception:
            risk_score = 0.0
            risk_band = "low_risk"

        results.append({
            "file": sess["file"],
            "source_type": sess["source_type"],
            "bot_type": sess["bot_type"],
            "label": sess["label"],
            "image_id": sess["image_id"],
            "bot_risk_score": risk_score,
            "risk_band": risk_band,
            "is_blocked": risk_band == "high_risk",
        })

        if i % 1000 == 0 or i == total_files:
            elapsed = time.time() - t0
            rps = i / elapsed
            eta = (total_files - i) / rps if rps > 0 else 0
            print(
                f"  [{i:>6}/{total_files}]  {elapsed:.1f}s 경과  ETA {eta:.0f}s",
                flush=True,
            )

    # ── 집계 ──────────────────────────────────────────────────────────────────
    human = [r for r in results if r["source_type"] == "human"]
    bot = [r for r in results if r["source_type"] == "bot"]

    def _rate(items, cond):
        return round(sum(1 for r in items if cond(r)) / max(len(items), 1), 6)

    attack_type_counts: Dict[str, Dict[str, int]] = {}
    for r in bot:
        bt = r["bot_type"] or "unknown"
        if bt not in attack_type_counts:
            attack_type_counts[bt] = {"count": 0, "blocked": 0}
        attack_type_counts[bt]["count"] += 1
        if r["is_blocked"]:
            attack_type_counts[bt]["blocked"] += 1

    risk_band_counts = {
        "low_risk": sum(1 for r in results if r["risk_band"] == "low_risk"),
        "suspicious": sum(1 for r in results if r["risk_band"] == "suspicious"),
        "high_risk": sum(1 for r in results if r["risk_band"] == "high_risk"),
    }

    summary = {
        "total_sessions": len(results),
        "human_total": len(human),
        "bot_total": len(bot),
        "human_pass_rate": _rate(human, lambda r: r["risk_band"] == "low_risk"),
        "human_suspicious_rate": _rate(human, lambda r: r["risk_band"] == "suspicious"),
        "human_block_rate": _rate(human, lambda r: r["is_blocked"]),
        "bot_detect_rate": _rate(bot, lambda r: r["is_blocked"]),
        "bot_miss_rate": _rate(bot, lambda r: not r["is_blocked"]),
        "attack_type_counts": attack_type_counts,
        "risk_band_counts": risk_band_counts,
    }

    cache = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": meta.get("version", "unknown"),
        "summary": summary,
        "sessions": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n완료! ({elapsed:.1f}s)")
    print(f"  저장 위치: {out_path}")
    print(f"  human {len(human):,}건 / bot {len(bot):,}건")
    print(f"  human_pass_rate:  {summary['human_pass_rate']:.3f}")
    print(f"  bot_detect_rate:  {summary['bot_detect_rate']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Flashlight dashboard cache 생성")
    ap.add_argument(
        "--data-dir",
        default=str(_DEFAULT_DATA_DIR),
        type=Path,
        help=f"flashlight 데이터 루트 (기본: {_DEFAULT_DATA_DIR})",
    )
    ap.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        type=Path,
        help=f"출력 cache JSON 경로 (기본: {_DEFAULT_OUT})",
    )
    ap.add_argument(
        "--model-dir",
        default=None,
        help="모델 디렉토리 오버라이드 (FLASHLIGHT_MODEL_DIR)",
    )
    args = ap.parse_args()
    run(args.data_dir, args.out, args.model_dir)


if __name__ == "__main__":
    main()
