#!/usr/bin/env python3
"""current vs candidate 비교 + promotion_gate 판정.

evaluate_candidate.py가 만든 evaluation_result.json 두 개(current/
candidate)를 읽어서 evaluation/promotion_gate.py에 넘기고, 그 결과를
promotion_decision.json으로 candidates/{version}/ 아래에 쓴다. 이 파일이
유일하게 deployment/promote_model.py 호출 여부를 사람/오케스트레이터가
판단하는 근거가 된다 - compare_candidate.py 자신은 어떤 파일도 옮기거나
지우지 않는다 (flashlight의 compare_candidate.py와 동일한 원칙).

종료 코드: 0=promote, 1=reject, 2=manual_review - run_model_update_pipeline.py가
이 코드로 다음 단계(promote 실행 여부)를 결정한다.

사용법:
    python -m context_emotion.evaluation.compare_candidate --version v1_20260701
    python -m context_emotion.evaluation.compare_candidate --version v1_20260701 --json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.deployment import model_store  # noqa: E402
from context_emotion.evaluation import promotion_gate  # noqa: E402

_EXIT_CODES = {"promote": 0, "reject": 1, "manual_review": 2}


def _load_evaluation_result(artifact_dir: str, label: str) -> dict:
    path = os.path.join(artifact_dir, "evaluation_result.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} evaluation_result.json을 찾을 수 없습니다: {path}")
    return model_store.load_json(path)


def run_compare(version: str) -> dict:
    candidate_dir = model_store.candidate_dir(version)
    current_dir = model_store.CURRENT_DIR

    candidate_eval = _load_evaluation_result(candidate_dir, f"candidate {version}")
    try:
        current_eval = _load_evaluation_result(current_dir, "current")
        current_version = model_store.load_json(os.path.join(current_dir, "metadata.json")).get("version", "unknown")
    except FileNotFoundError:
        # 첫 번째 모델이라 current가 아예 없는 경우 - 정상 상황, current_eval=None으로 진행
        current_eval = None
        current_version = "none"

    # 패키지 전체 contract(필수 파일/metadata/label_schema/preprocessing_config) 재검증.
    # evaluate_candidate.py의 artifact_integrity는 onnx/label만 보므로 더 넓은 검증을 여기서 보강한다.
    package_problems = model_store.validate_artifact_dir(candidate_dir)
    flat_package_problems = [p for plist in package_problems.values() for p in plist]
    if flat_package_problems:
        candidate_eval = dict(candidate_eval)
        candidate_eval["artifact_integrity"] = {
            "onnx_loadable": candidate_eval["artifact_integrity"].get("onnx_loadable", False),
            "input_output_match": candidate_eval["artifact_integrity"].get("input_output_match", False),
            "label_schema_match": False,
        }

    policy = model_store.load_promotion_policy()
    decision = promotion_gate.decide(
        current_version=current_version,
        candidate_version=version,
        current_eval=current_eval,
        candidate_eval=candidate_eval,
        policy=policy,
    )
    if flat_package_problems:
        decision["reasons"].extend([f"package_contract: {p}" for p in flat_package_problems])

    decision_path = os.path.join(candidate_dir, "promotion_decision.json")
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    decision["_decision_path"] = decision_path
    return decision


def print_decision(decision: dict) -> None:
    print("=" * 60)
    print(f"  context_emotion 승격 판정: current({decision['current_version']}) vs candidate({decision['candidate_version']})")
    print("=" * 60)
    for gate, status in decision["gates"].items():
        print(f"  [{status.upper():^14}] {gate}")
    print()
    print("  사유:")
    for reason in decision["reasons"]:
        print(f"    - {reason}")
    print()
    print(f"  최종 판정: {decision['final_decision'].upper()}")
    print(f"  기록 위치: {decision.get('_decision_path')}")
    print("=" * 60)


def _parse_args():
    ap = argparse.ArgumentParser(description="current vs candidate 비교 및 승격 게이트 판정")
    ap.add_argument("--version", required=True)
    ap.add_argument("--json", action="store_true", dest="as_json")
    return ap.parse_args()


def main():
    args = _parse_args()
    try:
        decision = run_compare(args.version)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(json.dumps(decision, indent=2, ensure_ascii=False))
    else:
        print_decision(decision)

    sys.exit(_EXIT_CODES[decision["final_decision"]])


if __name__ == "__main__":
    main()
