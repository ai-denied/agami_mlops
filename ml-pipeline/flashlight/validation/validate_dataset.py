#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flashlight Dataset Validation Layer
마우스 봇 감지 학습 데이터 전처리 전/후 품질 검증 도구

실행 예시
python ml-pipeline/flashlight/validation/validate_dataset.py \
  --data /path/to/merged_dynamic_features_sampled.json \
  --out-dir ./runs/validation

CI/CD 연동 (경고도 실패 처리):
python ... --strict && echo "OK" || echo "FAIL"
"""

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from flashlight.common.constants import SEQ_FEATURES, STATIC_FEATURES  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 검증 기준 상수
# ─────────────────────────────────────────────────────────────────────────────

VALIDATOR_VERSION = "1.1.0"

# 전처리 필터 기준 (extract_all_features.py 와 동일하게 유지)
DURATION_MIN_MS: float = 100.0
DURATION_MAX_MS: float = 20_000.0
MIN_LOG_COUNT: int = 5
MIN_DYNAMIC_FRAMES: int = 1

# 물리적 상한선 (px/ms 단위 — mouse_feature_adapter.py 기준)
# velocity = distance(px) / dt(ms)  →  일반 마우스 0.1~3 px/ms, 최대 20 px/ms
# acceleration = Δvelocity / dt      →  급격한 방향 변환 기준 10 px/ms²
# angle_change 범위: [0, π] (라디안)
VELOCITY_PHYSICS_MAX: float = 20.0
ACCELERATION_PHYSICS_MAX: float = 10.0
ANGLE_CHANGE_MAX: float = math.pi

# IQR 아웃라이어 배수 (3.0 → 극단적 이상값만 탐지)
OUTLIER_IQR_FACTOR: float = 3.0

# 클래스 불균형 경고 기준
IMBALANCE_WARN_RATIO_LOW: float = 0.05    # bot/human < 5%
IMBALANCE_WARN_RATIO_HIGH: float = 20.0   # bot/human > 20x

# 보고서 이슈 목록 최대 출력 수 (JSON 크기 제한)
MAX_ISSUES_PER_CHECK: int = 200
MAX_DUPLICATE_GROUPS: int = 50
MAX_OUTLIER_SAMPLES: int = 100

REQUIRED_TOP_LEVEL: Set[str] = {"label", "static_features", "dynamic_features"}
REQUIRED_STATIC: Set[str] = set(STATIC_FEATURES)
REQUIRED_SEQ: Set[str] = set(SEQ_FEATURES)
VALID_LABELS: Set[int] = {0, 1}

# static feature별 물리적 허용 범위 (하한, 상한) — NaN/IQR과 별개로 1차 필터
STATIC_PHYSICS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "duration":          (DURATION_MIN_MS, DURATION_MAX_MS),
    "log_count":         (MIN_LOG_COUNT, 100_000),
    "total_distance":    (0.0, float("inf")),
    "straight_distance": (0.0, float("inf")),
    "distance_ratio":    (0.0, float("inf")),
    "avg_speed":         (0.0, VELOCITY_PHYSICS_MAX * 2),
    "max_speed":         (0.0, VELOCITY_PHYSICS_MAX * 5),
    "speed_std":         (0.0, float("inf")),
    "direction_changes": (0.0, float("inf")),
    "pauses":            (0.0, float("inf")),
}


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _is_bad(v: Any) -> bool:
    """NaN 또는 Inf 여부를 안전하게 확인."""
    try:
        f = float(v)
        return math.isnan(f) or math.isinf(f)
    except (TypeError, ValueError):
        return False


def _fingerprint(sample: Dict) -> str:
    """세션 중복 탐지용 fingerprint 생성 (content-based)."""
    sf = sample.get("static_features") or {}
    parts = [
        str(sample.get("label", "")),
        str(sample.get("source_file", "") or sample.get("original_file", "")),
        str(sample.get("image_id", "")),
        f"{float(sf.get('duration', 0)):.1f}",
        str(int(sf.get("log_count", 0))),
        f"{float(sf.get('total_distance', 0)):.4f}",
        f"{float(sf.get('avg_speed', 0)):.6f}",
    ]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def _collect_static_values(samples: List[Dict]) -> Dict[str, List[float]]:
    """각 static feature의 유효 수치를 수집 (NaN/Inf 제외)."""
    result: Dict[str, List[float]] = defaultdict(list)
    for s in samples:
        sf = s.get("static_features")
        if not isinstance(sf, dict):
            continue
        for feat in STATIC_FEATURES:
            v = sf.get(feat)
            if v is not None and not _is_bad(v):
                try:
                    result[feat].append(float(v))
                except (TypeError, ValueError):
                    pass
    return result


def _iqr_fence(values: List[float], factor: float) -> Tuple[float, float]:
    """IQR 기반 이상값 경계 계산."""
    arr = np.array(values, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def _feature_stats(values: List[float]) -> Dict[str, float]:
    """간단한 기술 통계 (보고서 포함용)."""
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "count":  len(values),
        "mean":   round(float(arr.mean()), 6),
        "std":    round(float(arr.std()), 6),
        "min":    round(float(arr.min()), 6),
        "p25":    round(float(np.percentile(arr, 25)), 6),
        "median": round(float(np.median(arr)), 6),
        "p75":    round(float(np.percentile(arr, 75)), 6),
        "max":    round(float(arr.max()), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — JSON 스키마 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_schema(samples: List[Dict]) -> Dict[str, Any]:
    """
    최상위 구조 및 label 값 유효성 검증.

    실패 조건:
    - label 필드 없음 또는 {0, 1} 외 값
    - static_features가 dict가 아님
    - dynamic_features가 list가 아님
    """
    issues = []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            issues.append({"index": i, "reason": "sample is not a dict"})
            continue

        missing = REQUIRED_TOP_LEVEL - set(s.keys())
        if missing:
            issues.append({"index": i, "reason": f"missing top-level fields: {sorted(missing)}"})
            continue

        label = s.get("label")
        if label not in VALID_LABELS:
            issues.append({"index": i, "reason": f"invalid label: {label!r} (expected 0 or 1)"})

        if not isinstance(s.get("static_features"), dict):
            issues.append({"index": i, "reason": "static_features must be a dict"})

        if not isinstance(s.get("dynamic_features"), list):
            issues.append({"index": i, "reason": "dynamic_features must be a list"})

        if len(issues) >= MAX_ISSUES_PER_CHECK:
            break

    return {
        "status": "failed" if issues else "passed",
        "total_invalid": len(issues),
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — 필수 피처 키 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_required_fields(samples: List[Dict]) -> Dict[str, Any]:
    """
    static_features / dynamic_features 내부 필수 키 존재 여부 검증.
    constants.py의 SEQ_FEATURES, STATIC_FEATURES 기준.
    """
    issues = []
    missing_static_counter: Dict[str, int] = defaultdict(int)
    missing_seq_counter: Dict[str, int] = defaultdict(int)

    for i, s in enumerate(samples):
        sf = s.get("static_features")
        if isinstance(sf, dict):
            missing = REQUIRED_STATIC - set(sf.keys())
            for k in missing:
                missing_static_counter[k] += 1
            if missing:
                issues.append({
                    "index": i,
                    "missing_static": sorted(missing),
                })

        df = s.get("dynamic_features")
        if isinstance(df, list) and len(df) > 0 and isinstance(df[0], dict):
            missing = REQUIRED_SEQ - set(df[0].keys())
            for k in missing:
                missing_seq_counter[k] += 1
            if missing:
                issues.append({
                    "index": i,
                    "missing_seq_fields": sorted(missing),
                })

        if len(issues) >= MAX_ISSUES_PER_CHECK:
            break

    return {
        "status": "failed" if issues else "passed",
        "total_invalid": len(issues),
        "missing_static_key_counts": dict(missing_static_counter),
        "missing_seq_key_counts": dict(missing_seq_counter),
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — NaN / Inf 탐지
# ─────────────────────────────────────────────────────────────────────────────

def check_nan_inf(samples: List[Dict]) -> Dict[str, Any]:
    """
    static_features 및 dynamic_features 내 모든 수치에서 NaN / Inf 탐지.
    학습 시 loss=nan 또는 gradient explode의 주원인.
    """
    issues = []
    static_bad_counts: Dict[str, int] = defaultdict(int)
    seq_bad_counts: Dict[str, int] = defaultdict(int)

    for i, s in enumerate(samples):
        bad_fields: List[str] = []

        sf = s.get("static_features")
        if isinstance(sf, dict):
            for k, v in sf.items():
                if _is_bad(v):
                    bad_fields.append(f"static_features.{k}={v}")
                    static_bad_counts[k] += 1

        df = s.get("dynamic_features")
        if isinstance(df, list):
            for fi, frame in enumerate(df):
                if not isinstance(frame, dict):
                    continue
                for k, v in frame.items():
                    if _is_bad(v):
                        bad_fields.append(f"dynamic_features[{fi}].{k}={v}")
                        seq_bad_counts[k] += 1
                if len(bad_fields) >= 20:
                    bad_fields.append("... (truncated)")
                    break

        if bad_fields:
            issues.append({"index": i, "bad_fields": bad_fields[:20]})

        if len(issues) >= MAX_ISSUES_PER_CHECK:
            break

    return {
        "status": "failed" if issues else "passed",
        "total_invalid": len(issues),
        "static_nan_inf_counts": dict(static_bad_counts),
        "seq_nan_inf_counts": dict(seq_bad_counts),
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — duration 범위 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_duration(samples: List[Dict]) -> Dict[str, Any]:
    """
    duration이 전처리 필터 기준(100~20,000ms)을 벗어난 샘플 탐지.
    머지된 데이터에 잘못된 값이 포함된 경우를 잡아냄.
    """
    issues = []
    durations: List[float] = []

    for i, s in enumerate(samples):
        sf = s.get("static_features")
        if not isinstance(sf, dict):
            continue
        dur = sf.get("duration")
        if dur is None:
            continue
        try:
            dur_f = float(dur)
        except (TypeError, ValueError):
            issues.append({"index": i, "duration": dur, "reason": "non-numeric"})
            continue

        durations.append(dur_f)
        if not (DURATION_MIN_MS <= dur_f <= DURATION_MAX_MS):
            issues.append({
                "index": i,
                "duration_ms": round(dur_f, 2),
                "reason": f"out of range [{DURATION_MIN_MS:.0f}, {DURATION_MAX_MS:.0f}]ms",
            })

        if len(issues) >= MAX_ISSUES_PER_CHECK:
            break

    return {
        "status": "warning" if issues else "passed",
        "valid_range_ms": [DURATION_MIN_MS, DURATION_MAX_MS],
        "total_invalid": len(issues),
        "stats": _feature_stats(durations),
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — 마우스 로그 길이 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_log_length(samples: List[Dict]) -> Dict[str, Any]:
    """
    log_count 최소값 및 dynamic_features 프레임 수 검증.
    너무 짧은 세션은 피처 추출이 불안정해 학습 품질을 저하시킴.
    """
    issues = []
    log_counts: List[float] = []
    dynamic_lens: List[int] = []

    for i, s in enumerate(samples):
        sf = s.get("static_features") or {}
        df = s.get("dynamic_features") or []

        log_count = sf.get("log_count")
        dynamic_len = len(df) if isinstance(df, list) else 0
        dynamic_lens.append(dynamic_len)

        if log_count is not None:
            try:
                lc = int(float(log_count))
                log_counts.append(float(lc))
                if lc < MIN_LOG_COUNT:
                    issues.append({
                        "index": i,
                        "log_count": lc,
                        "reason": f"log_count < {MIN_LOG_COUNT}",
                    })
            except (TypeError, ValueError):
                issues.append({"index": i, "log_count": log_count, "reason": "non-numeric"})

        if dynamic_len < MIN_DYNAMIC_FRAMES:
            issues.append({
                "index": i,
                "dynamic_frames": dynamic_len,
                "reason": f"dynamic_features length < {MIN_DYNAMIC_FRAMES}",
            })

        if len(issues) >= MAX_ISSUES_PER_CHECK:
            break

    return {
        "status": "warning" if issues else "passed",
        "min_log_count": MIN_LOG_COUNT,
        "min_dynamic_frames": MIN_DYNAMIC_FRAMES,
        "total_invalid": len(issues),
        "log_count_stats": _feature_stats(log_counts),
        "dynamic_len_stats": _feature_stats([float(x) for x in dynamic_lens]),
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 6 — 중복 세션 탐지
# ─────────────────────────────────────────────────────────────────────────────

def check_duplicates(samples: List[Dict]) -> Dict[str, Any]:
    """
    동일 세션이 중복 포함된 케이스를 fingerprint 기반으로 탐지.
    (source_file + image_id + duration + log_count + total_distance + avg_speed 조합)

    중복 샘플이 있으면 group-based split이 데이터 누수로 이어질 수 있음.
    """
    seen: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        fp = _fingerprint(s)
        seen[fp].append(i)

    duplicate_groups = [indices for indices in seen.values() if len(indices) > 1]
    total_dup_samples = sum(len(g) - 1 for g in duplicate_groups)

    dup_issues = [
        {
            "indices": g,
            "count": len(g),
            "label": samples[g[0]].get("label"),
            "source_type": samples[g[0]].get("source_type"),
        }
        for g in duplicate_groups[:MAX_DUPLICATE_GROUPS]
    ]

    return {
        "status": "warning" if duplicate_groups else "passed",
        "total_duplicate_samples": total_dup_samples,
        "total_duplicate_groups": len(duplicate_groups),
        "issues": dup_issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 7 — 이상값(Outlier) 탐지
# ─────────────────────────────────────────────────────────────────────────────

def check_outliers(samples: List[Dict]) -> Dict[str, Any]:
    """
    두 가지 방식으로 이상값을 탐지한다.

    [방법 1] 물리적 상한선 위반 (dynamic features)
      - velocity > 20 px/ms (인간 마우스의 물리적 한계 초과)
      - |acceleration| > 10 px/ms²
      - angle_change > π (라디안 계산 오류 지표)

    [방법 2] IQR 기반 이상값 (static features)
      - Q1 - 3×IQR 미만 또는 Q3 + 3×IQR 초과
      - 피처별 상하한 및 영향받은 샘플 수 보고
    """
    physics_violations: List[Dict] = []
    iqr_outlier_samples: Dict[int, List[str]] = defaultdict(list)
    iqr_counts: Dict[str, int] = defaultdict(int)

    # ── 방법 1: 물리적 상한선 ──
    for i, s in enumerate(samples):
        df = s.get("dynamic_features")
        if not isinstance(df, list):
            continue
        violations: List[str] = []
        for fi, frame in enumerate(df):
            if not isinstance(frame, dict):
                continue
            v_vel = frame.get("velocity", 0)
            v_acc = frame.get("acceleration", 0)
            v_ang = frame.get("angle_change", 0)

            try:
                if abs(float(v_vel)) > VELOCITY_PHYSICS_MAX:
                    violations.append(
                        f"frame[{fi}].velocity={float(v_vel):.4f} > {VELOCITY_PHYSICS_MAX}"
                    )
                if abs(float(v_acc)) > ACCELERATION_PHYSICS_MAX:
                    violations.append(
                        f"frame[{fi}].acceleration={float(v_acc):.4f} > {ACCELERATION_PHYSICS_MAX}"
                    )
                if abs(float(v_ang)) > ANGLE_CHANGE_MAX:
                    violations.append(
                        f"frame[{fi}].angle_change={float(v_ang):.4f} > {ANGLE_CHANGE_MAX:.4f}"
                    )
            except (TypeError, ValueError):
                pass

            if len(violations) >= 5:
                violations.append("... (truncated)")
                break

        if violations:
            physics_violations.append({"index": i, "violations": violations})
            if len(physics_violations) >= MAX_ISSUES_PER_CHECK:
                break

    # ── 방법 2: IQR 기반 ──
    static_values = _collect_static_values(samples)
    iqr_bounds: Dict[str, Dict[str, float]] = {}

    for feat, vals in static_values.items():
        if len(vals) < 4:
            continue
        lo, hi = _iqr_fence(vals, OUTLIER_IQR_FACTOR)
        stats = _feature_stats(vals)
        iqr_bounds[feat] = {
            "lower_fence": round(lo, 6),
            "upper_fence": round(hi, 6),
            **stats,
        }

    for i, s in enumerate(samples):
        sf = s.get("static_features")
        if not isinstance(sf, dict):
            continue
        for feat, bounds in iqr_bounds.items():
            v = sf.get(feat)
            if v is None or _is_bad(v):
                continue
            fv = float(v)
            lo = bounds["lower_fence"]
            hi = bounds["upper_fence"]
            if fv < lo or fv > hi:
                label = f"{feat}={fv:.4f} (fence=[{lo:.4f}, {hi:.4f}])"
                iqr_outlier_samples[i].append(label)
                iqr_counts[feat] += 1

    iqr_issues = [
        {"index": idx, "outlier_fields": fields[:5]}
        for idx, fields in list(iqr_outlier_samples.items())[:MAX_OUTLIER_SAMPLES]
    ]

    has_issues = bool(physics_violations) or bool(iqr_outlier_samples)

    return {
        "status": "warning" if has_issues else "passed",
        "physics_bounds": {
            "velocity_max_px_per_ms":      VELOCITY_PHYSICS_MAX,
            "acceleration_max_px_per_ms2": ACCELERATION_PHYSICS_MAX,
            "angle_change_max_rad":        round(ANGLE_CHANGE_MAX, 6),
        },
        "physics_violations": {
            "total_affected_samples": len(physics_violations),
            "samples": physics_violations[:50],
        },
        "iqr_outliers": {
            "iqr_factor":            OUTLIER_IQR_FACTOR,
            "per_feature_bounds":    iqr_bounds,
            "per_feature_counts":    dict(iqr_counts),
            "total_affected_samples": len(iqr_outlier_samples),
            "samples":               iqr_issues,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 8 — 클래스 분포 보고
# ─────────────────────────────────────────────────────────────────────────────

def check_class_distribution(samples: List[Dict]) -> Dict[str, Any]:
    """
    label(0=human / 1=bot) 분포, bot_type별 분포, 클래스 불균형 비율 보고.
    심각한 불균형은 pos_weight 설정에 영향을 줌.
    """
    n_total = len(samples)
    n_human = 0
    n_bot = 0
    n_unknown = 0
    bot_type_counts: Dict[str, int] = defaultdict(int)
    source_type_counts: Dict[str, int] = defaultdict(int)
    user_id_counts: Dict[str, int] = defaultdict(int)

    for s in samples:
        label = s.get("label")
        if label == 0:
            n_human += 1
        elif label == 1:
            n_bot += 1
        else:
            n_unknown += 1

        bt = str(s.get("bot_type") or "none")
        st = str(s.get("source_type") or "unknown")
        uid = str(s.get("user_id") or s.get("participant_id") or "unknown")

        bot_type_counts[bt] += 1
        source_type_counts[st] += 1
        user_id_counts[uid] += 1

    imbalance_ratio = round(n_bot / max(n_human, 1), 4)
    n_unique_users = len(user_id_counts)

    distribution_warnings: List[str] = []
    if n_human == 0:
        distribution_warnings.append("human 샘플(label=0) 없음")
    if n_bot == 0:
        distribution_warnings.append("bot 샘플(label=1) 없음")
    if n_unknown > 0:
        distribution_warnings.append(f"유효하지 않은 label 샘플 {n_unknown}건")
    if n_human > 0 and imbalance_ratio < IMBALANCE_WARN_RATIO_LOW:
        distribution_warnings.append(
            f"심각한 불균형: bot/human = {imbalance_ratio} < {IMBALANCE_WARN_RATIO_LOW}"
        )
    if imbalance_ratio > IMBALANCE_WARN_RATIO_HIGH:
        distribution_warnings.append(
            f"심각한 불균형: bot/human = {imbalance_ratio} > {IMBALANCE_WARN_RATIO_HIGH}"
        )

    return {
        "status": "warning" if distribution_warnings else "passed",
        "total_samples":       n_total,
        "human_count":         n_human,
        "bot_count":           n_bot,
        "unknown_label_count": n_unknown,
        "human_ratio":         round(n_human / max(n_total, 1), 4),
        "bot_ratio":           round(n_bot / max(n_total, 1), 4),
        "bot_to_human_ratio":  imbalance_ratio,
        "unique_user_ids":     n_unique_users,
        "bot_type_distribution":    dict(sorted(bot_type_counts.items())),
        "source_type_distribution": dict(source_type_counts),
        "distribution_warnings":    distribution_warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 메인 검증 함수
# ─────────────────────────────────────────────────────────────────────────────

_CHECKS = [
    ("schema_validation",  check_schema),
    ("required_fields",    check_required_fields),
    ("nan_inf",            check_nan_inf),
    ("duration",           check_duration),
    ("log_length",         check_log_length),
    ("duplicates",         check_duplicates),
    ("outliers",           check_outliers),
    ("class_distribution", check_class_distribution),
]

# 이 상태가 fail이면 전체 검증 실패 (warning은 기본적으로 통과)
_CRITICAL_CHECKS = {"schema_validation", "required_fields", "nan_inf"}


def validate_dataset(
    data_path: str,
    out_dir: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    데이터셋 전체 검증을 실행하고 결과 dict를 반환한다.
    out_dir이 주어지면 validation_report.json으로 저장한다.

    Parameters
    ----------
    data_path : 검증할 JSON 파일 경로
    out_dir   : 보고서 저장 폴더 (None이면 저장 안 함)
    strict    : True이면 warning도 실패로 처리 (CI/CD용)

    Returns
    -------
    report dict (summary.passed == True/False)

    Raises
    ------
    FileNotFoundError : 데이터 파일 없음
    ValueError        : JSON 최상위가 list가 아님
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("최상위 JSON이 list 형태여야 합니다.")

    print(f"\n[Flashlight Validation] 파일 로드 완료: {len(samples):,}건  →  {data_path}")

    checks: Dict[str, Any] = {}
    for name, fn in _CHECKS:
        print(f"  ▸ {name:<30}", end=" ", flush=True)
        result = fn(samples)
        checks[name] = result
        status_label = result["status"].upper()
        n = result.get("total_invalid", result.get("total_duplicate_samples", "–"))
        print(f"[{status_label:>7}]  issues={n}")

    # ── 집계 ──
    failed_checks  = [k for k, v in checks.items() if v["status"] == "failed"]
    warning_checks = [k for k, v in checks.items() if v["status"] == "warning"]

    critical_issues = sum(checks[k].get("total_invalid", 0) for k in failed_checks)
    warning_issues  = sum(
        checks[k].get("total_invalid",
            checks[k].get("total_duplicate_samples", 0))
        for k in warning_checks
    )

    # strict=False : critical check 실패만 전체 실패로 처리
    # strict=True  : warning도 전체 실패로 처리
    has_critical_failure = any(k in _CRITICAL_CHECKS for k in failed_checks) or bool(failed_checks)
    overall_passed = not has_critical_failure and (not strict or not warning_checks)

    # ── 무효 샘플 인덱스 수집 (schema/field/nan/duration/log_length 기준) ──
    invalid_indices: Set[int] = set()
    for check_name in ("schema_validation", "required_fields", "nan_inf", "duration", "log_length"):
        for issue in checks.get(check_name, {}).get("issues", []):
            if isinstance(issue.get("index"), int):
                invalid_indices.add(issue["index"])
    # 중복: 첫 번째 인덱스를 원본으로 보고 나머지를 무효로 처리
    for group_info in checks.get("duplicates", {}).get("issues", []):
        for dup_idx in group_info.get("indices", [])[1:]:
            invalid_indices.add(dup_idx)

    report: Dict[str, Any] = {
        "metadata": {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "data_path":         os.path.abspath(data_path),
            "total_samples":     len(samples),
            "validator_version": VALIDATOR_VERSION,
        },
        "summary": {
            "passed":                overall_passed,
            "failed_checks":         failed_checks,
            "warning_checks":        warning_checks,
            "total_critical_issues": critical_issues,
            "total_warning_issues":  warning_issues,
            "total_invalid_samples": len(invalid_indices),
            "strict_mode":           strict,
            "checks_run":            [name for name, _ in _CHECKS],
        },
        "checks": checks,
        "invalid_sample_indices": sorted(invalid_indices),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "validation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[Report] 저장 완료: {report_path}")

    _print_summary(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 터미널 출력 요약
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    checks  = report["checks"]
    meta    = report["metadata"]

    print("\n" + "=" * 65)
    print("  FLASHLIGHT DATASET VALIDATION REPORT")
    print("=" * 65)
    print(f"  파일           : {meta['data_path']}")
    print(f"  총 샘플 수     : {meta['total_samples']:,}")
    print(f"  검증 시각      : {meta['timestamp']}")
    print(f"  Validator ver  : {meta['validator_version']}")
    print()
    print(f"  최종 결과      : {'PASSED' if summary['passed'] else 'FAILED'}")
    print(f"  Strict mode    : {summary['strict_mode']}")
    print(f"  Critical issues: {summary['total_critical_issues']}")
    print(f"  Warning issues : {summary['total_warning_issues']}")
    print(f"  무효 샘플 수   : {summary['total_invalid_samples']}")
    print()
    print(f"  {'Check':<32}{'Status':>9}   Issues")
    print("  " + "-" * 55)

    status_icon = {"passed": "OK", "failed": "FAIL", "warning": "WARN"}
    for name, _ in _CHECKS:
        result = checks[name]
        st     = result["status"]
        icon   = status_icon.get(st, "?")
        n      = result.get("total_invalid",
                    result.get("total_duplicate_samples", "–"))
        print(f"  {name:<32}[{icon:>4}]   {n}")

    # 클래스 분포 요약
    dist = checks.get("class_distribution", {})
    if dist:
        print()
        print(f"  클래스 분포 :")
        print(f"    human={dist.get('human_count', 0):,}  "
              f"bot={dist.get('bot_count', 0):,}  "
              f"(bot/human={dist.get('bot_to_human_ratio', '?')})")
        for bt, cnt in dist.get("bot_type_distribution", {}).items():
            print(f"    {bt}: {cnt:,}")
        for w in dist.get("distribution_warnings", []):
            print(f"    [!] {w}")

    # 실패/경고 항목 요약
    if summary["failed_checks"]:
        print(f"\n  [FAIL] {summary['failed_checks']}")
    if summary["warning_checks"]:
        print(f"  [WARN] {summary['warning_checks']}")

    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flashlight 마우스 봇 감지 데이터셋 검증 도구"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="검증할 JSON 파일 경로 (merged_dynamic_features_sampled.json)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./runs/validation",
        help="validation_report.json 저장 폴더 (기본: ./runs/validation)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="경고(warning)도 실패로 처리하여 exit code 1 반환 (CI/CD용)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="보고서 파일 저장 생략 (터미널 출력만)",
    )
    args = parser.parse_args()

    report = validate_dataset(
        data_path=args.data,
        out_dir=None if args.no_save else args.out_dir,
        strict=args.strict,
    )

    sys.exit(0 if report["summary"]["passed"] else 1)


if __name__ == "__main__":
    main()
