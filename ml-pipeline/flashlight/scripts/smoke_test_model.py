#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flashlight 추론 API 운영 스모크 테스트.

promote_model.py 승격 + deployment rollout restart 직후 실행한다. 모델 파일이
로컬 디스크에 잘 놓였는지가 아니라, 실제로 떠 있는 API pod가 "방금 승격한
그 버전"으로 응답하는지를 확인한다 - readinessProbe(/health)는 model_loaded만
보기 때문에, rollout이 끝나도 (예: 여러 replica 중 일부가 재시작에 실패해)
옛 모델을 계속 서빙 중인 pod가 섞여 있는 경우를 못 잡는다.

검사 항목:
  1. GET  /health         — model_loaded=true, model_version == --expected-version
  2. POST /api/v1/predict — 샘플 trajectory로 200 응답, bot_risk_score/risk_band 정상 반환

실패하면 비0 종료코드를 반환한다 - Argo Workflow가 이 단계 실패로 인해
파이프라인 전체를 실패로 표시하게 하기 위함.

사용법:
  python -m flashlight.scripts.smoke_test_model \\
    --base-url http://flashlight-inference-api-svc.agami.svc.cluster.local \\
    --expected-version v4_20260610
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple


def _http_get_json(url: str, timeout: float) -> Tuple[int, Dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict, timeout: float) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        body_json = json.loads(raw.decode("utf-8")) if raw else {}
        return e.code, body_json


def _sample_trajectory() -> List[Dict[str, float]]:
    """사람 마우스 움직임을 단순 흉내낸 더미 trajectory. 점수 자체의 정확도는
    보지 않고 "추론이 에러 없이 끝나는지"만 본다 (스모크 테스트 범위)."""
    points = []
    t = 0.0
    x, y = 0.1, 0.1
    for i in range(20):
        x += 0.02
        y += 0.015 * ((-1) ** i)
        t += 16.0
        points.append({"x": round(x, 4), "y": round(y, 4), "t": t})
    return points


def check_health(base_url: str, expected_version: str, timeout: float) -> List[str]:
    problems = []
    try:
        status, body = _http_get_json(f"{base_url}/health", timeout)
    except Exception as e:
        return [f"/health 요청 실패: {e}"]

    if status != 200:
        problems.append(f"/health HTTP status {status} (200 기대)")
    if not body.get("model_loaded"):
        problems.append(f"/health model_loaded={body.get('model_loaded')!r} (true 기대)")

    actual_version = body.get("model_version")
    if actual_version != expected_version:
        problems.append(
            f"/health model_version={actual_version!r} != 방금 승격한 candidate 버전={expected_version!r} "
            f"- rollout이 끝났지만 옛 모델을 서빙 중인 pod가 있을 수 있음"
        )
    return problems


def check_predict(base_url: str, timeout: float) -> List[str]:
    problems = []
    payload = {
        "trajectory": _sample_trajectory(),
        "coordinate_mode": "normalized",
        "canvas_width": 800,
        "canvas_height": 600,
    }
    try:
        status, body = _http_post_json(f"{base_url}/api/v1/predict", payload, timeout)
    except Exception as e:
        return [f"/api/v1/predict 요청 실패: {e}"]

    if status != 200:
        return [f"/api/v1/predict HTTP status {status} (200 기대) — body={body}"]

    for key in ("bot_risk_score", "risk_band"):
        if key not in body:
            problems.append(f"/api/v1/predict 응답에 '{key}' 없음 — body={body}")

    score = body.get("bot_risk_score")
    if isinstance(score, (int, float)) and not (0.0 <= score <= 1.0):
        problems.append(f"bot_risk_score 범위 오류: {score}")

    risk_band = body.get("risk_band")
    if risk_band not in ("low_risk", "suspicious", "high_risk"):
        problems.append(f"risk_band 값 이상: {risk_band!r}")

    return problems


def run_smoke_test(base_url: str, expected_version: str, timeout: float) -> bool:
    base_url = base_url.rstrip("/")

    print(f"[1/2] GET  /health  (expected_version={expected_version})")
    health_problems = check_health(base_url, expected_version, timeout)
    if health_problems:
        for p in health_problems:
            print(f"  [FAIL] {p}")
    else:
        print("  [OK] model_loaded=true, model_version 일치")

    print("[2/2] POST /api/v1/predict")
    predict_problems = check_predict(base_url, timeout)
    if predict_problems:
        for p in predict_problems:
            print(f"  [FAIL] {p}")
    else:
        print("  [OK] bot_risk_score/risk_band 정상 반환")

    problems = health_problems + predict_problems
    print()
    if problems:
        print(f"[FAILED] 스모크 테스트 실패 — {len(problems)}개 문제")
        return False
    print("[OK] 스모크 테스트 통과")
    return True


def _parse_args():
    parser = argparse.ArgumentParser(description="flashlight 추론 API 운영 스모크 테스트")
    parser.add_argument(
        "--base-url", required=True,
        help="추론 API base URL (예: http://flashlight-inference-api-svc.agami.svc.cluster.local)",
    )
    parser.add_argument(
        "--expected-version", required=True,
        help="방금 승격한 candidate 버전 — /health의 model_version과 비교",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="요청 타임아웃(초)")
    parser.add_argument(
        "--retries", type=int, default=6,
        help="rollout 직후 일시적 비가용 구간을 흡수하기 위한 재시도 횟수",
    )
    parser.add_argument("--retry-interval", type=float, default=5.0, help="재시도 간격(초)")
    return parser.parse_args()


def main():
    args = _parse_args()

    for attempt in range(1, args.retries + 1):
        print(f"=== 스모크 테스트 시도 {attempt}/{args.retries} ===")
        if run_smoke_test(args.base_url, args.expected_version, args.timeout):
            sys.exit(0)
        if attempt < args.retries:
            print(f"  {args.retry_interval:.0f}초 후 재시도...\n")
            time.sleep(args.retry_interval)

    print(f"\n[FAILED] {args.retries}회 재시도 후에도 스모크 테스트 실패")
    sys.exit(1)


if __name__ == "__main__":
    main()
