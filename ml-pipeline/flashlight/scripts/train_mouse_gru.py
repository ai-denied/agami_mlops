#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentient-CAPTCHA Mouse Behavior GRU Trainer v2 — 공식 학습 진입점

실행 예시
python ml-pipeline/flashlight/scripts/train_mouse_gru.py \
  --data /path/to/merged_dynamic_features_sampled.json \
  --out-dir ./runs/mouse_gru_final_v3_policy_tuned
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ml-pipeline/ をパスに追加して flashlight パッケージとして認識させる
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from flashlight.common.constants import SEQ_FEATURES, STATIC_FEATURES  # noqa: E402
from flashlight.common.device import get_device, set_seed  # noqa: E402
from flashlight.data.dataset import collate_fn  # noqa: E402
from flashlight.data.normalizer import MouseFeatureNormalizer  # noqa: E402
from flashlight.evaluation.metrics import (  # noqa: E402
    choose_threshold_by_human_block_rate,
    evaluate_thresholds,
    get_tpr_at_fpr,
    print_eval_report,
    safe_pr_auc,
    safe_roc_auc,
)
from flashlight.evaluation.plot import (  # noqa: E402
    plot_loss,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
    plot_threshold_metrics,
)
from flashlight.evaluation.threshold_policy import evaluate_three_attempt_policy  # noqa: E402
from flashlight.model.mouse_gru import MouseGRUModelV2  # noqa: E402
from flashlight.training.split import make_train_val_test_split  # noqa: E402
from flashlight.training.train_gru import (  # noqa: E402
    check_data_path,
    diagnose_dataset,
    evaluate_loss,
    get_pos_weight,
    make_loader,
    predict_risk_scores,
    train_one_epoch,
)
from flashlight.validation.validate_dataset import validate_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Sentient-CAPTCHA mouse behavior GRU trainer with 3-attempt policy"
    )

    parser.add_argument("--data", type=str, required=True, help="학습 데이터 JSON 경로")
    parser.add_argument("--out-dir", type=str, default="./runs/mouse_gru_final_v2", help="결과 저장 폴더")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_auc",
        choices=["val_auc", "val_pr_auc", "val_loss"],
        help="best model 선택 기준",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-key",
        type=str,
        default="auto",
        choices=[
            "auto",
            "user_id",
            "participant_id",
            "real_user_id",
            "bot_type",
            "image_id",
            "original_file",
            "source_file",
        ],
    )

    parser.add_argument(
        "--max-human-block-rate",
        type=float,
        default=0.20,
        help="low risk threshold 선택 시 허용할 최대 사람 오탐률",
    )

    parser.add_argument(
        "--high-risk-human-block-rate",
        type=float,
        default=0.02,
        help="high risk threshold 선택 시 허용할 최대 사람 오탐률. 기본 0.02",
    )

    parser.add_argument(
        "--min-high-risk-threshold",
        type=float,
        default=0.60,
        help="high risk threshold가 너무 낮게 잡히지 않도록 하는 최소 기준. 기본 0.60",
    )

    parser.add_argument("--seq-noise-std", type=float, default=0.01)
    parser.add_argument("--static-noise-std", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="데이터 검증 단계를 건너뜀 (개발/디버그용)",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="경고(warning)도 학습 중단으로 처리",
    )

    parser.add_argument("--use-mlflow", action="store_true", help="MLflow Tracking 활성화")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow Tracking URI (예: http://localhost:5000)")
    parser.add_argument("--mlflow-experiment-name", type=str, default="mouse_gru", help="MLflow Experiment 이름")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="MLflow Run 이름")

    parser.add_argument("--three-attempts", type=int, default=3)
    parser.add_argument("--block-suspicious-count", type=int, default=2)
    parser.add_argument("--block-high-risk-count", type=int, default=1)
    parser.add_argument(
        "--block-total-score",
        type=float,
        default=0.25,
        help="3회 누적 block 기준. 기본 0.25",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── MLflow 초기화 ─────────────────────────────────────────────────────────
    if args.use_mlflow:
        if not MLFLOW_AVAILABLE:
            print(
                "\n[MLflow] mlflow 패키지를 찾을 수 없습니다.\n"
                "         pip install mlflow 를 실행한 뒤 다시 시도해 주세요.\n"
                "         지금은 --use-mlflow 옵션을 무시하고 기존 방식으로 학습합니다.\n"
            )
            args.use_mlflow = False
        else:
            if args.mlflow_tracking_uri:
                mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            mlflow.set_experiment(args.mlflow_experiment_name)
            mlflow.start_run(run_name=args.mlflow_run_name)
            mlflow.log_params({
                "data": args.data,
                "out_dir": args.out_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden": args.hidden,
                "layers": args.layers,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "device": args.device,
                "max_human_block_rate": args.max_human_block_rate,
                "high_risk_human_block_rate": args.high_risk_human_block_rate,
            })
            print(f"[MLflow] Run 시작: experiment={args.mlflow_experiment_name}, run_name={args.mlflow_run_name}")
    # ─────────────────────────────────────────────────────────────────────────

    device = get_device(args.device)

    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("GPU 필수 설정인데 CUDA를 사용할 수 없습니다.")

    print(f"\n사용 디바이스: {device}")

    if device.type == "cuda":
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전(torch): {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")

    check_data_path(args.data)

    with open(args.data, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"\n전체 데이터 수: {len(all_data)}")
    all_labels = [int(d.get("label", 0)) for d in all_data]
    print("전체 label count:", pd.Series(all_labels).value_counts().to_dict())

    if len(set(all_labels)) < 2:
        raise ValueError("label이 한 종류만 있습니다. 0=human, 1=bot 데이터가 모두 필요합니다.")

    # ── 데이터 검증 ──────────────────────────────────────────────────────────
    if args.skip_validation:
        print("\n[Validation] --skip-validation 설정으로 검증 단계 생략")
    else:
        val_report = validate_dataset(
            data_path=args.data,
            out_dir=args.out_dir,
            strict=args.strict_validation,
        )
        if not val_report["summary"]["passed"]:
            failed = val_report["summary"]["failed_checks"]
            report_path = os.path.join(args.out_dir, "validation_report.json")
            raise RuntimeError(
                f"데이터 검증 실패 — failed checks: {failed}\n"
                f"상세 내용: {report_path}"
            )
    # ─────────────────────────────────────────────────────────────────────────

    diagnose_dataset(all_data, args.out_dir)

    train_raw, val_raw, test_raw = make_train_val_test_split(
        all_data,
        group_key=args.group_key,
        seed=args.seed,
    )

    normalizer = MouseFeatureNormalizer()
    normalizer.fit(train_raw)

    train_loader = make_loader(
        train_raw,
        normalizer,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seq_noise_std=args.seq_noise_std,
        static_noise_std=args.static_noise_std,
        training=True,
    )

    val_loader = make_loader(
        val_raw,
        normalizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seq_noise_std=0.0,
        static_noise_std=0.0,
        training=False,
    )

    test_loader = make_loader(
        test_raw,
        normalizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seq_noise_std=0.0,
        static_noise_std=0.0,
        training=False,
    )

    model = MouseGRUModelV2(
        seq_size=len(SEQ_FEATURES),
        static_size=len(STATIC_FEATURES),
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    pos_weight = get_pos_weight(train_raw, device)
    print(f"pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    train_losses = []
    val_losses = []
    history_rows = []

    best_score = None
    best_epoch = 0
    best_model_state = None
    patience_count = 0

    print("\n학습 시작")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=args.grad_clip,
        )

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_scores, val_labels = predict_risk_scores(model, val_loader, device)

        val_auc = safe_roc_auc(val_labels, val_scores)
        val_pr_auc = safe_pr_auc(val_labels, val_scores)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if args.monitor == "val_loss":
            current_score = -val_loss
        elif args.monitor == "val_pr_auc":
            current_score = val_pr_auc if val_pr_auc is not None else -999
        else:
            current_score = val_auc if val_auc is not None else -999

        improved = (
            best_score is None
            or current_score > best_score + args.min_delta
        )

        if improved:
            best_score = current_score
            best_epoch = epoch
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_pr_auc": val_pr_auc,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
        })

        if args.use_mlflow:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc if val_auc is not None else 0.0,
                    "val_pr_auc": val_pr_auc if val_pr_auc is not None else 0.0,
                },
                step=epoch,
            )

        if epoch % args.print_every == 0 or epoch == 1:
            auc_text = f"{val_auc:.4f}" if val_auc is not None else "N/A"
            pr_text = f"{val_pr_auc:.4f}" if val_pr_auc is not None else "N/A"

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val ROC-AUC: {auc_text} "
                f"Val PR-AUC: {pr_text} "
                f"Best Epoch: {best_epoch} "
                f"Patience: {patience_count}/{args.patience}"
            )

        if patience_count >= args.patience:
            print(f"\nEarly stopping 발생. epoch={epoch}, best_epoch={best_epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model 로드 완료: epoch={best_epoch}, monitor={args.monitor}, best_score={best_score:.4f}")

    history_path = os.path.join(args.out_dir, "train_history.csv")
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    print(f"학습 히스토리 저장: {history_path}")

    # Validation threshold 선택
    val_scores, val_labels = predict_risk_scores(model, val_loader, device)
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    val_threshold_df = evaluate_thresholds(val_labels, val_scores, thresholds)

    low_risk_threshold, low_reason = choose_threshold_by_human_block_rate(
        val_threshold_df,
        max_human_block_rate=args.max_human_block_rate,
        mode="best_bot_recall",
    )

    high_risk_threshold, high_reason = choose_threshold_by_human_block_rate(
        val_threshold_df,
        max_human_block_rate=args.high_risk_human_block_rate,
        mode="high_risk",
        min_threshold=args.min_high_risk_threshold,
    )

    if high_risk_threshold < low_risk_threshold:
        high_risk_threshold = low_risk_threshold

    print("\n========== Validation Threshold 결과 ==========")
    print(val_threshold_df.to_string(index=False))
    print(f"\nLow risk threshold : {low_risk_threshold} ({low_reason})")
    print(f"High risk threshold: {high_risk_threshold} ({high_reason})")

    # Test 1회 기준 평가
    test_scores, test_labels = predict_risk_scores(model, test_loader, device)

    print_eval_report(
        name="Test - single attempt",
        y_true=test_labels,
        scores=test_scores,
        threshold=low_risk_threshold,
    )

    tpr_at_fpr = get_tpr_at_fpr(test_labels, test_scores)

    print("\nTPR@FPR")
    for k, v in tpr_at_fpr.items():
        print(f"{k}: TPR={v['tpr']}, FPR={v['fpr']}, threshold={v['threshold']}")

    test_threshold_df = evaluate_thresholds(test_labels, test_scores, thresholds)

    print("\n========== Test Threshold 결과 ==========")
    print(test_threshold_df.to_string(index=False))

    # 3회 누적 정책 평가
    three_val_df, three_val_summary = evaluate_three_attempt_policy(
        y_true=val_labels,
        scores=val_scores,
        low_risk_threshold=low_risk_threshold,
        high_risk_threshold=high_risk_threshold,
        attempts=args.three_attempts,
        seed=args.seed,
        block_suspicious_count=args.block_suspicious_count,
        block_high_risk_count=args.block_high_risk_count,
        block_total_score=args.block_total_score,
    )

    three_test_df, three_test_summary = evaluate_three_attempt_policy(
        y_true=test_labels,
        scores=test_scores,
        low_risk_threshold=low_risk_threshold,
        high_risk_threshold=high_risk_threshold,
        attempts=args.three_attempts,
        seed=args.seed + 99,
        block_suspicious_count=args.block_suspicious_count,
        block_high_risk_count=args.block_high_risk_count,
        block_total_score=args.block_total_score,
    )

    print("\n========== 3회 누적 정책 Validation 요약 ==========")
    print(json.dumps(three_val_summary, ensure_ascii=False, indent=2))

    print("\n========== 3회 누적 정책 Test 요약 ==========")
    print(json.dumps(three_test_summary, ensure_ascii=False, indent=2))

    # 저장
    model_path = os.path.join(args.out_dir, "mouse_gru_server_final_v2.pth")
    normalizer_path = os.path.join(args.out_dir, "mouse_normalizer_server_final_v2.joblib")
    metadata_path = os.path.join(args.out_dir, "mouse_metadata_server_final_v2.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump(normalizer, normalizer_path)

    single_preds = (test_scores >= low_risk_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, single_preds, labels=[0, 1]).ravel()

    test_roc_auc = safe_roc_auc(test_labels, test_scores)
    test_pr_auc = safe_pr_auc(test_labels, test_scores)

    summary = {
        "best_epoch": best_epoch,
        "monitor": args.monitor,
        "best_monitor_score": best_score,
        "score_name": "bot_risk_score",
        "low_risk_threshold": low_risk_threshold,
        "low_threshold_reason": low_reason,
        "high_risk_threshold": high_risk_threshold,
        "high_threshold_reason": high_reason,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_accuracy": accuracy_score(test_labels, single_preds),
        "test_precision_bot": precision_score(test_labels, single_preds, zero_division=0),
        "test_recall_bot": recall_score(test_labels, single_preds, zero_division=0),
        "test_f1_bot": f1_score(test_labels, single_preds, zero_division=0),
        "test_human_block_rate": fp / max(fp + tn, 1),
        "test_bot_miss_rate": fn / max(fn + tp, 1),
        "confusion_matrix": {
            "tn_human_correct": int(tn),
            "fp_human_blocked": int(fp),
            "fn_bot_missed": int(fn),
            "tp_bot_detected": int(tp),
        },
        "tpr_at_fpr": tpr_at_fpr,
        "three_attempt_validation_summary": three_val_summary,
        "three_attempt_test_summary": three_test_summary,
        "hyperparameters": vars(args),
    }

    metadata = {
        "model_name": "MouseGRUModelV2",
        "seq_features": SEQ_FEATURES,
        "static_features": STATIC_FEATURES,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "score_name": "bot_risk_score",
        "low_risk_threshold": low_risk_threshold,
        "high_risk_threshold": high_risk_threshold,
        "label_rule": "0=human, 1=bot",
        "three_attempt_policy": {
            "attempts": args.three_attempts,
            "block_suspicious_count": args.block_suspicious_count,
            "block_high_risk_count": args.block_high_risk_count,
            "block_total_score": float(
                args.block_total_score
                if args.block_total_score is not None
                else low_risk_threshold * 2.0
            ),
        },
        "summary": summary,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if args.use_mlflow:
        mlflow.log_metrics({
            "test_auc": test_roc_auc if test_roc_auc is not None else 0.0,
            "test_pr_auc": test_pr_auc if test_pr_auc is not None else 0.0,
            "accuracy": summary["test_accuracy"],
            "precision": summary["test_precision_bot"],
            "recall": summary["test_recall_bot"],
            "f1": summary["test_f1_bot"],
            "low_risk_threshold": float(low_risk_threshold),
            "high_risk_threshold": float(high_risk_threshold),
        })

    val_threshold_path = os.path.join(args.out_dir, "threshold_metrics_validation.csv")
    test_threshold_path = os.path.join(args.out_dir, "threshold_metrics_test.csv")
    summary_path = os.path.join(args.out_dir, "final_summary.json")
    three_val_path = os.path.join(args.out_dir, "three_attempt_validation.csv")
    three_test_path = os.path.join(args.out_dir, "three_attempt_test.csv")
    three_policy_path = os.path.join(args.out_dir, "three_attempt_service_policy.json")

    val_threshold_df.to_csv(val_threshold_path, index=False)
    test_threshold_df.to_csv(test_threshold_path, index=False)
    three_val_df.to_csv(three_val_path, index=False)
    three_test_df.to_csv(three_test_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    service_policy = {
        "score_name": "bot_risk_score",
        "single_attempt": {
            "low_risk": f"score < {low_risk_threshold}",
            "suspicious": f"{low_risk_threshold} <= score < {high_risk_threshold}",
            "high_risk": f"score >= {high_risk_threshold}",
        },
        "three_attempt_policy": {
            "attempts": args.three_attempts,
            "decision_order": [
                "block if high_risk_count >= block_high_risk_count",
                "block if suspicious_count >= block_suspicious_count",
                "block if total_score >= block_total_score",
                "challenge_again if suspicious_count >= 1",
                "allow otherwise",
            ],
            "block_suspicious_count": args.block_suspicious_count,
            "block_high_risk_count": args.block_high_risk_count,
            "block_total_score": float(
                args.block_total_score
                if args.block_total_score is not None
                else low_risk_threshold * 2.0
            ),
        },
    }

    with open(three_policy_path, "w", encoding="utf-8") as f:
        json.dump(service_policy, f, ensure_ascii=False, indent=2)

    print("\n========== 최종 요약 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n========== 저장 완료 ==========")
    print(f"Model       : {model_path}")
    print(f"Normalizer  : {normalizer_path}")
    print(f"Metadata    : {metadata_path}")
    print(f"Val Metrics : {val_threshold_path}")
    print(f"Test Metrics: {test_threshold_path}")
    print(f"Summary     : {summary_path}")
    print(f"3회 Val     : {three_val_path}")
    print(f"3회 Test    : {three_test_path}")
    print(f"3회 정책    : {three_policy_path}")

    if args.use_mlflow:
        _mlflow_artifacts = [
            history_path,
            os.path.join(args.out_dir, "validation_report.json"),
            os.path.join(args.out_dir, "split_distribution_report.json"),
            summary_path,
            model_path,
            normalizer_path,
            metadata_path,
            val_threshold_path,
            test_threshold_path,
            three_val_path,
            three_test_path,
            three_policy_path,
        ]
        for _artifact in _mlflow_artifacts:
            if os.path.exists(_artifact):
                mlflow.log_artifact(_artifact)
        mlflow.end_run()
        print(f"\n[MLflow] Run 종료: experiment={args.mlflow_experiment_name}")

    plot_loss(train_losses, val_losses, args.out_dir)
    plot_score_distribution(test_labels, test_scores, args.out_dir)
    plot_threshold_metrics(test_threshold_df, args.out_dir)
    plot_roc_curve(test_labels, test_scores, args.out_dir)
    plot_pr_curve(test_labels, test_scores, args.out_dir)

    print("\n완료. 결과 저장 폴더:")
    print(args.out_dir)


if __name__ == "__main__":
    main()