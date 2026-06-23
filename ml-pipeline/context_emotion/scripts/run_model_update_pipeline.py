#!/usr/bin/env python3
"""context_emotion 전체 모델 업데이트 파이프라인.

evaluate -> package -> compare -> [gate] -> promote -> smoke_test -> record

flashlight/scripts/run_model_update_pipeline.py(package -> compare ->
promote)보다 단계가 많은 이유: emotion CAPTCHA는 단순 성능 비교가
아니라 클래스별 회귀/사람 애매함/공격 프록시까지 보는 게이트가 있고
(MLOPS_OPERATION_DESIGN.md), 승격 후 실제로 떠 있는지 보는 smoke test와
운영 메타데이터 기록까지가 "한 번의 업데이트"로 취급된다.

compare 단계의 final_decision이 'promote'가 아니면(reject/manual_review)
그 다음 단계(promote/smoke_test/record)는 전부 건너뛴다 - flashlight와
동일한 "FAIL이면 중단" 원칙.

사용법:
    python -m context_emotion.scripts.run_model_update_pipeline \\
        --version v1_20260701 \\
        --onnx runs/emotion_classifier_v1/model.onnx \\
        --metadata runs/emotion_classifier_v1/metadata.json \\
        --label-schema runs/emotion_classifier_v1/label_schema.json \\
        --preprocessing-config runs/emotion_classifier_v1/preprocessing_config.json \\
        --eval-csv /workspace/data/context_emotion/processed/context_emotion_train_dataset_v2.csv \\
        --image-root /workspace/data/context_emotion

    # promote까지는 가지 말고 evaluate/package/compare만 보고 싶을 때
    ... --dry-run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.deployment import model_store, promote_model  # noqa: E402
from context_emotion.evaluation import compare_candidate, evaluate_candidate  # noqa: E402
from context_emotion.ops_metrics.recorder import DailyMetrics, record_daily_metrics  # noqa: E402
from context_emotion.scripts import package_emotion_model, smoke_test_model  # noqa: E402


def _step(n: int, total: int, name: str) -> None:
    print(f"\n── STEP {n}/{total}  {name} {'─' * max(0, 40 - len(name))}")


def run_pipeline(args) -> int:
    """반환값: compare_candidate.py와 동일한 의미의 종료 코드
    (0=promote 완료, 1=reject, 2=manual_review)."""
    total_steps = 6
    eval_out_path = os.path.join(os.path.dirname(args.onnx), "evaluation_result.json")

    _step(1, total_steps, "evaluate_candidate")
    eval_result = evaluate_candidate.evaluate(
        onnx_path=args.onnx,
        label_schema_path=args.label_schema,
        preprocessing_config_path=args.preprocessing_config,
        eval_csv=args.eval_csv,
        eval_split=args.eval_split,
        image_root=args.image_root,
        version=args.version,
        attacker_proxy_model=args.attacker_proxy_model,
        attacker_proxy_eval_pool=args.attacker_proxy_eval_pool,
    )
    import json
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {eval_out_path}  overall={eval_result['overall']}")

    _step(2, total_steps, "package_emotion_model")
    candidate_dir = package_emotion_model.package(
        onnx=args.onnx, metadata=args.metadata, label_schema=args.label_schema,
        preprocessing_config=args.preprocessing_config, evaluation_result=eval_out_path,
        version=args.version, checkpoint=args.checkpoint,
    )
    print(f"  [OK] {candidate_dir}")

    _step(3, total_steps, "compare_candidate (promotion gate)")
    decision = compare_candidate.run_compare(args.version)
    compare_candidate.print_decision(decision)

    if decision["final_decision"] != "promote":
        print(f"\n[STOP] final_decision={decision['final_decision']} - promote/smoke_test/record를 건너뜁니다.")
        return _EXIT_CODES[decision["final_decision"]]

    _step(4, total_steps, "promote_model" + ("  [DRY-RUN]" if args.dry_run else ""))
    promote_model.promote(version=args.version, dry=args.dry_run)

    if args.dry_run:
        print("\n[DRY-RUN 완료] smoke_test/record는 dry-run에서 건너뜁니다 (current가 실제로 안 바뀌었음).")
        return 0

    _step(5, total_steps, "smoke_test_model")
    smoke_test_model.smoke_test(model_store.CURRENT_DIR)

    _step(6, total_steps, "record ops_metrics (deployment marker)")
    from datetime import date
    path = record_daily_metrics(DailyMetrics(
        date=date.today().isoformat(),
        model_version=args.version,
        exposures=0,
    ))
    print(f"  [OK] {path}  (exposures=0 - 방금 배포됐고 아직 운영 노출 없음, 실제 집계는 서빙 측 호출 필요)")

    print("\n[SUCCESS] 전체 파이프라인 완료")
    return 0


_EXIT_CODES = {"promote": 0, "reject": 1, "manual_review": 2}


def _parse_args():
    ap = argparse.ArgumentParser(description="evaluate -> package -> compare -> promote -> smoke_test -> record")
    ap.add_argument("--version", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--label-schema", required=True)
    ap.add_argument("--preprocessing-config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--attacker-proxy-model", default=None)
    ap.add_argument("--attacker-proxy-eval-pool", default=None)
    ap.add_argument("--dry-run", action="store_true", help="promote를 dry-run으로만 수행, smoke_test/record는 생략")
    return ap.parse_args()


def main():
    args = _parse_args()
    try:
        code = run_pipeline(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[FAILED] {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
