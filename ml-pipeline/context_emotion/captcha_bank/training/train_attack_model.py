#!/usr/bin/env python3
"""
sklearn 기반 듀얼 어태커 프록시 모델 학습.

Model 1 — emotion_attacker (RandomForestClassifier)
    선택지 세트 20-dim 피처만으로 final_emotion을 예측.
    이미지를 전혀 보지 못하는 메타데이터 전용 공격자의 baseline 정답률을 측정한다.
    attacker_solve_rate가 낮을수록 CAPTCHA 풀이 더 안전하다.

Model 2 — security_ranker (GradientBoostingRegressor, optional)
    문항 메타데이터 4-dim 피처로 attack_hardness(VLM 공격 난이도)를 회귀 예측.
    풀 난이도 분포 분석에 사용한다. attack_hardness 열이 없으면 skip한다.

출력: model.joblib
    {
        "version": str,
        "emotion_attacker": {"model": ..., "label_names": [...], ...},
        "security_ranker":  {"model": ..., ...} | None,
        "pool_size": int,
        "trained_at": str,
        "pool_sha256": str,
    }

사용법:
    python -m context_emotion.captcha_bank.training.train_attack_model \\
        --pool-csv  /path/to/captcha_pool.csv \\
        --output    model.joblib \\
        --version   v1_20260701
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from context_emotion.captcha_bank.choice_generation import EMOTIONS, load_rows
from context_emotion.captcha_bank.training.features import (
    ATTACKER_FEATURE_NAMES,
    DIFFICULTY_FEATURE_NAMES,
    attack_hardness_targets,
    build_attacker_matrix,
    build_difficulty_matrix,
    emotion_labels,
)

_W = 60


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _filter_valid(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("final_emotion", "") in EMOTIONS]


def train_emotion_attacker(rows: list[dict], test_size: float = 0.2, random_state: int = 42):
    """RandomForest that predicts final_emotion from choice-set features."""
    X = build_attacker_matrix(rows, seed=random_state)
    le = LabelEncoder().fit(EMOTIONS)
    y = le.transform([r["final_emotion"] for r in rows])

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    val_preds = clf.predict(X_val)
    val_acc   = float(accuracy_score(y_val, val_preds))
    val_f1    = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    train_acc = float(accuracy_score(y_tr, clf.predict(X_tr)))

    return {
        "model":         clf,
        "label_encoder": le,
        "label_names":   EMOTIONS,
        "feature_names": ATTACKER_FEATURE_NAMES,
        "n_train":       len(X_tr),
        "n_val":         len(X_val),
        "train_accuracy": round(train_acc, 4),
        "val_accuracy":   round(val_acc, 4),
        "val_macro_f1":   round(val_f1, 4),
    }


def train_security_ranker(rows: list[dict], test_size: float = 0.2, random_state: int = 42):
    """GradientBoosting that predicts attack_hardness from metadata features."""
    hardness = attack_hardness_targets(rows)
    if np.all(hardness == 0.5):
        print("  [SKIP] attack_hardness 열이 없거나 전부 결측 — security_ranker 학습 건너뜀")
        return None

    X = build_difficulty_matrix(rows)
    X_tr, X_val, y_tr, y_val = train_test_split(X, hardness, test_size=test_size, random_state=random_state)

    reg = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=random_state,
    )
    reg.fit(X_tr, y_tr)

    val_mae   = float(mean_absolute_error(y_val, reg.predict(X_val)))
    train_mae = float(mean_absolute_error(y_tr, reg.predict(X_tr)))

    return {
        "model":          reg,
        "feature_names":  DIFFICULTY_FEATURE_NAMES,
        "n_train":        len(X_tr),
        "n_val":          len(X_val),
        "train_mae":      round(train_mae, 4),
        "val_mae":        round(val_mae, 4),
    }


def train(pool_csv: Path, output: Path, version: str) -> dict:
    print("═" * _W)
    print(f"  captcha_bank 어태커 프록시 모델 학습  [{version}]")
    print("═" * _W)

    rows = _filter_valid(load_rows(pool_csv))
    print(f"  유효 문항 수: {len(rows)}")
    if len(rows) < 50:
        raise ValueError(f"학습 데이터 부족: {len(rows)}개 (최소 50개 필요)")

    print("\n  [1/2] emotion_attacker 학습 중 ...")
    attacker = train_emotion_attacker(rows)
    print(f"        val_accuracy={attacker['val_accuracy']:.4f}  val_macro_f1={attacker['val_macro_f1']:.4f}")

    print("  [2/2] security_ranker 학습 중 ...")
    ranker = train_security_ranker(rows)
    if ranker:
        print(f"        val_mae={ranker['val_mae']:.4f}")

    bundle = {
        "version":          version,
        "pool_size":        len(rows),
        "pool_sha256":      _sha256(pool_csv),
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "emotion_attacker": attacker,
        "security_ranker":  ranker,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"\n  → {output}  ({output.stat().st_size // 1024} KB)")

    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPTCHA 풀 어태커 프록시 모델 학습")
    ap.add_argument("--pool-csv",  type=Path, required=True, help="export_captcha_pool.py 출력 CSV")
    ap.add_argument("--output",    type=Path, default=Path("model.joblib"))
    ap.add_argument("--version",   required=True, help="모델 버전 (예: v1_20260701)")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    train(args.pool_csv, args.output, args.version)


if __name__ == "__main__":
    main()
