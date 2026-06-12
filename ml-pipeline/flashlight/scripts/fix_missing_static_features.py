#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
missing static_features 복구 스크립트
- dynamic_features를 기반으로 static_features를 재계산
- 원본 파일은 수정하지 않음
- mouse_feature_adapter.py와 동일한 계산 로직 사용

사용법:
  python flashlight/scripts/fix_missing_static_features.py
"""

import json
import math
import os
import sys
from collections import Counter

INPUT_PATH  = "/home/ubuntu/agami-mlops/data/flashlight/processed/merged_dynamic_features_sampled.json"
OUTPUT_PATH = "/home/ubuntu/agami-mlops/data/flashlight/processed/merged_dynamic_features_sampled_fixed.json"

# mouse_feature_adapter.py 와 동일한 임계값
DIRECTION_CHANGE_THRESHOLD_RAD: float = 0.5
PAUSE_VELOCITY_THRESHOLD: float = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 계산 로직 (mouse_feature_adapter.py build_features_from_trajectory 기반)
# ─────────────────────────────────────────────────────────────────────────────

def _pop_std(values):
    """Population standard deviation (adapter와 동일)."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def compute_static_from_dynamic(sample):
    """
    dynamic_features 리스트에서 static_features 10개를 재계산한다.

    log_count  = len(original_points) = len(dynamic_frames) + 1
    duration   = top-level 'duration' 필드 우선, 없으면 sum(dt)
    direction_changes = angle_change >= 0.5 rad 인 프레임 수
    pauses            = velocity <= 0.01 px/ms 인 프레임 수
    """
    dyn = sample.get("dynamic_features", [])
    if not dyn:
        return None

    velocities    = [float(f.get("velocity",     0.0)) for f in dyn]
    angle_changes = [float(f.get("angle_change", 0.0)) for f in dyn]
    distances     = [float(f.get("distance",     0.0)) for f in dyn]
    dts           = [float(f.get("dt",           0.0)) for f in dyn]
    dxs           = [float(f.get("dx",           0.0)) for f in dyn]
    dys           = [float(f.get("dy",           0.0)) for f in dyn]

    # duration: 상위 필드 우선, 없으면 합산
    raw_duration = sample.get("duration")
    if raw_duration is not None:
        duration = float(raw_duration)
    else:
        duration = max(sum(dts), 1.0)

    log_count        = float(len(dyn) + 1)
    total_distance   = float(sum(distances))

    net_dx = sum(dxs)
    net_dy = sum(dys)
    straight_distance = math.sqrt(net_dx ** 2 + net_dy ** 2)
    distance_ratio    = total_distance / max(straight_distance, 1e-6)

    avg_speed  = sum(velocities) / max(len(velocities), 1)
    max_speed  = max(velocities) if velocities else 0.0
    speed_std  = _pop_std(velocities)

    direction_changes = float(sum(1 for a in angle_changes if a >= DIRECTION_CHANGE_THRESHOLD_RAD))
    pauses            = float(sum(1 for v in velocities    if v <= PAUSE_VELOCITY_THRESHOLD))

    return {
        "duration":          duration,
        "log_count":         log_count,
        "total_distance":    total_distance,
        "straight_distance": straight_distance,
        "distance_ratio":    distance_ratio,
        "avg_speed":         avg_speed,
        "max_speed":         max_speed,
        "speed_std":         speed_std,
        "direction_changes": direction_changes,
        "pauses":            pauses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 분포 출력 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _dist_str(items):
    return dict(sorted(Counter(str(v) for v in items).items()))


def print_stats(label, data):
    labels      = [d.get("label")       for d in data]
    src_types   = [d.get("source_type") for d in data]
    missing_cnt = sum(1 for d in data if not isinstance(d.get("static_features"), dict))

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  전체 샘플 수          : {len(data):,}")
    print(f"  static_features 누락  : {missing_cnt}")
    print(f"  label 분포            : {_dist_str(labels)}")
    print(f"  source_type 분포      : {_dist_str(src_types)}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"입력 파일: {INPUT_PATH}")
    print(f"출력 파일: {OUTPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print_stats("수정 전", data)

    # ── 복구 ──────────────────────────────────────────────────────────────────
    recovered = 0
    failed    = 0

    fixed_data = []
    for i, sample in enumerate(data):
        s = dict(sample)  # shallow copy (원본 수정 방지)

        if not isinstance(s.get("static_features"), dict):
            sf = compute_static_from_dynamic(s)
            if sf is None:
                print(f"  [WARN] index={i} dynamic_features 비어있어 복구 불가 — 제외하지 않고 유지")
                failed += 1
            else:
                # 기존 static_features 필드에 없는 extra 키(label 등) 보존을 위해
                # 기존 샘플이 가진 비-STATIC 키는 그대로 두고 sf만 삽입
                s["static_features"] = sf
                recovered += 1

        fixed_data.append(s)

    print(f"\n  복구된 샘플 수  : {recovered}")
    print(f"  복구 실패 수    : {failed}  (dynamic_features 없음)")

    print_stats("수정 후", fixed_data)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, ensure_ascii=False)

    print(f"\n  저장 완료: {OUTPUT_PATH}")

    # ── 검증 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  validate_dataset 실행")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from flashlight.validation.validate_dataset import validate_dataset

    report = validate_dataset(
        data_path=OUTPUT_PATH,
        out_dir=None,
        strict=False,
    )

    schema_status = report["checks"]["schema_validation"]["status"]
    req_status    = report["checks"]["required_fields"]["status"]
    nan_status    = report["checks"]["nan_inf"]["status"]

    print(f"\n  schema_validation  : {schema_status.upper()}")
    print(f"  required_fields    : {req_status.upper()}")
    print(f"  nan_inf            : {nan_status.upper()}")
    print(f"  전체 통과 여부     : {'PASSED' if report['summary']['passed'] else 'FAILED'}")

    if not report["summary"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
