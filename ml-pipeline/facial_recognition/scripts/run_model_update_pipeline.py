"""
facial_recognition 모델 업데이트 파이프라인.

train → export → promote 순서로 실행한다.
각 단계가 실패하면 이후 단계를 중단한다.

사용법:
  # 전체 파이프라인 (재학습 → ONNX 변환 → 승격)
  python -m facial_recognition.scripts.run_model_update_pipeline \\
    --version v1_20260616 \\
    --data    /workspace/ml-pipeline/data/facial_recognition/face_clip_data.npz \\
    --run-dir /workspace/ml-pipeline/runs/face_liveness_v1

  # 이미 학습된 체크포인트만 변환·승격
  python -m facial_recognition.scripts.run_model_update_pipeline \\
    --version    v1_20260616 \\
    --checkpoint /workspace/ml-pipeline/runs/face_liveness_v1/best_gru.pt \\
    --skip-train

  # dry-run (train/export 실행, promote는 파일 변경 없이 출력만)
  python -m facial_recognition.scripts.run_model_update_pipeline \\
    --version v1_20260616 \\
    --data    ... \\
    --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
_ML_PIPELINE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_STORE            = os.path.join(_ML_PIPELINE_ROOT, "model-store", "facial_recognition")
_CANDIDATES_DIR   = os.path.join(_STORE, "candidates")
_W = 60


# ── 로깅 유틸 ─────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print("═" * _W)
    print(f"  {title}")
    print("═" * _W)


def _step_header(n: int, total: int, name: str, tag: str = "") -> None:
    tag_str = f"  [{tag}]" if tag else ""
    print()
    print(f"┌─ STEP {n}/{total}  {name}{tag_str}")
    print(f"│{'─' * (_W - 2)}")


def _step_ok(n: int, total: int, detail: str = "", elapsed: float = 0.0) -> None:
    time_str = f"  ({elapsed:.2f}s)" if elapsed else ""
    suffix   = f" — {detail}" if detail else ""
    print(f"└─ STEP {n}/{total}  ✓ 완료{suffix}{time_str}")


def _step_fail(n: int, total: int, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"└─ STEP {n}/{total}  ✗ 실패{suffix}")


def _step_skip(n: int, total: int, reason: str) -> None:
    print(f"└─ STEP {n}/{total}  ─ 건너뜀  ({reason})")


# ── STEP 1: 학습 ──────────────────────────────────────────────────────────────

def run_train(args) -> tuple[bool, str | None]:
    """
    Returns (success, checkpoint_path)
    """
    from facial_recognition.training.train_gru import train
    try:
        out = train(
            data_path=args.data,
            out_dir=args.run_dir,
            feature_mode=args.feature_mode,
            min_seq_len=args.min_seq_len,
            min_face_rate=args.min_face_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            threshold_strategy=args.threshold_strategy,
            device=args.device,
            seed=args.seed,
        )
        ckpt = str(Path(out) / "best_gru.pt")
        return True, ckpt
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return False, None


# ── STEP 2: ONNX 변환 + 후보 패키징 ─────────────────────────────────────────

def run_export(checkpoint_path: str, version: str) -> tuple[bool, str | None]:
    """
    체크포인트 → ONNX 변환 후 candidates/{version}/ 에 패키징.

    Returns (success, candidate_dir)
    """
    from facial_recognition.export.export_face_liveness_onnx import export

    candidate_dir = os.path.join(_CANDIDATES_DIR, version)
    os.makedirs(candidate_dir, exist_ok=True)

    onnx_path = os.path.join(candidate_dir, "face_liveness.onnx")
    meta_path = os.path.join(candidate_dir, "metadata.json")

    try:
        # ONNX 변환 (onnx_meta 를 직접 metadata.json 으로 사용)
        export(checkpoint_path=checkpoint_path, output_path=onnx_path, meta_path=meta_path)

        # metadata.json 에 버전 정보 추가
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["version"]    = version
        meta["packaged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # seq_scaler.joblib 복사 (checkpoint 와 같은 디렉토리에 있다고 가정)
        ckpt_dir    = os.path.dirname(checkpoint_path)
        scaler_src  = os.path.join(ckpt_dir, "seq_scaler.joblib")
        scaler_dst  = os.path.join(candidate_dir, "seq_scaler.joblib")
        if os.path.isfile(scaler_src):
            shutil.copy2(scaler_src, scaler_dst)
            print(f"  seq_scaler.joblib 복사: {scaler_dst}")
        else:
            print(f"  [WARN] seq_scaler.joblib 없음 — 건너뜀: {scaler_src}")

        print(f"  후보 패키지 저장: {candidate_dir}")
        return True, candidate_dir

    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return False, None


# ── STEP 3: 승격 ─────────────────────────────────────────────────────────────

def run_promote(version: str, dry_run: bool) -> bool:
    from facial_recognition.scripts.promote_model import promote
    try:
        promote(version=version, dry=dry_run, skip_validate=True)
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return False


# ── 파이프라인 오케스트레이터 ─────────────────────────────────────────────────

def run_pipeline(args) -> bool:
    total_steps = 2 if args.skip_train else 3
    dry_tag     = "DRY-RUN" if args.dry_run else ""

    _banner(
        f"facial_recognition 모델 업데이트 파이프라인\n"
        f"  버전: {args.version}"
        + (f"  [{dry_tag}]" if dry_tag else "")
    )

    results: dict[str, bool | None] = {
        "train":   None,
        "export":  None,
        "promote": None,
    }
    checkpoint_path = getattr(args, "checkpoint", None)
    pipeline_start  = time.monotonic()

    step = 0

    # ── STEP 1: 학습 ─────────────────────────────────────────────────────────
    if args.skip_train:
        step += 1
        _step_header(step, total_steps, "학습 (건너뜀)")
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            _step_fail(step, total_steps, f"--checkpoint 파일이 없습니다: {checkpoint_path}")
            _print_final(results, time.monotonic() - pipeline_start)
            return False
        results["train"] = True
        _step_skip(step, total_steps, f"--skip-train  체크포인트: {checkpoint_path}")
    else:
        step += 1
        _step_header(step, total_steps, "학습", "train_gru")
        t0 = time.monotonic()
        ok, checkpoint_path = run_train(args)
        elapsed = time.monotonic() - t0
        results["train"] = ok
        if ok:
            _step_ok(step, total_steps, f"checkpoint: {checkpoint_path}", elapsed)
        else:
            _step_fail(step, total_steps, "학습 실패")
            _print_final(results, time.monotonic() - pipeline_start)
            return False

    # ── STEP 2: ONNX 변환 + 패키징 ───────────────────────────────────────────
    step += 1
    _step_header(step, total_steps, "ONNX 변환 + 후보 패키징", "export_face_liveness_onnx")
    t0 = time.monotonic()
    ok, candidate_dir = run_export(checkpoint_path, args.version)
    elapsed = time.monotonic() - t0
    results["export"] = ok
    if ok:
        _step_ok(step, total_steps, f"candidate: {candidate_dir}", elapsed)
    else:
        _step_fail(step, total_steps, "변환 실패")
        _print_final(results, time.monotonic() - pipeline_start)
        return False

    # ── STEP 3: 승격 ─────────────────────────────────────────────────────────
    step += 1
    promote_tag = "DRY-RUN" if args.dry_run else ""
    _step_header(step, total_steps, "승격", f"promote_model{('  ['+promote_tag+']') if promote_tag else ''}")
    t0 = time.monotonic()
    ok = run_promote(args.version, args.dry_run)
    elapsed = time.monotonic() - t0
    results["promote"] = ok
    if ok:
        detail = "DRY-RUN 완료" if args.dry_run else f"current 버전 → {args.version}"
        _step_ok(step, total_steps, detail, elapsed)
    else:
        _step_fail(step, total_steps, "승격 실패")

    _print_final(results, time.monotonic() - pipeline_start)
    return all(v is True for v in results.values() if v is not None)


def _print_final(results: dict, total_elapsed: float) -> None:
    all_done = all(v is True for v in results.values() if v is not None)
    verdict  = "SUCCESS" if all_done else "FAILED"

    print()
    print("═" * _W)
    print("  단계별 결과:")

    icons = {True: "✓", False: "✗", None: "─"}
    names = {
        "train":   "학습     (train_gru)",
        "export":  "변환     (export_face_liveness_onnx)",
        "promote": "승격     (promote_model)",
    }
    for key, name in names.items():
        val     = results.get(key)
        icon    = icons[val]
        skipped = "  [건너뜀]" if val is None else ""
        print(f"    {icon}  {name}{skipped}")

    print()
    print(f"  총 소요시간: {total_elapsed:.2f}s")
    print(f"  최종 결과:   {verdict}")
    print("═" * _W)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="train → export → promote 파이프라인"
    )

    # 필수
    parser.add_argument("--version", required=True,
                        help="모델 버전명 (예: v1_20260616)")

    # 학습
    train_g = parser.add_argument_group("학습 옵션 (--skip-train 미지정 시 사용)")
    train_g.add_argument("--data",               default=None,
                         help="face_clip_data.npz 경로")
    train_g.add_argument("--run-dir",            default=None,
                         help="학습 출력 디렉토리 (기본: runs/{version})")
    train_g.add_argument("--feature-mode",       default="all",
                         choices=["all", "no_abs", "motion_only", "eye_mouth", "head"])
    train_g.add_argument("--min-seq-len",        type=int,   default=1)
    train_g.add_argument("--min-face-rate",      type=float, default=0.0)
    train_g.add_argument("--epochs",             type=int,   default=80)
    train_g.add_argument("--batch-size",         type=int,   default=64)
    train_g.add_argument("--hidden-size",        type=int,   default=32)
    train_g.add_argument("--num-layers",         type=int,   default=1)
    train_g.add_argument("--dropout",            type=float, default=0.3)
    train_g.add_argument("--lr",                 type=float, default=5e-4)
    train_g.add_argument("--weight-decay",       type=float, default=1e-4)
    train_g.add_argument("--patience",           type=int,   default=15)
    train_g.add_argument("--threshold-strategy", default="best_f1",
                         choices=["best_f1", "eer_like", "low_far", "default"])
    train_g.add_argument("--device",             default="auto")
    train_g.add_argument("--seed",               type=int,   default=42)

    # 학습 건너뜀
    skip_g = parser.add_argument_group("학습 건너뜀 옵션")
    skip_g.add_argument("--skip-train",  action="store_true",
                        help="학습 생략. --checkpoint 필수")
    skip_g.add_argument("--checkpoint",  default=None,
                        help="기존 best_gru.pt 경로 (--skip-train 사용 시)")

    # 공통
    parser.add_argument("--dry-run", action="store_true",
                        help="train/export 실행, promote는 dry-run")

    return parser.parse_args()


def main():
    args = _parse_args()

    # run_dir 기본값
    if args.run_dir is None:
        args.run_dir = os.path.join(_ML_PIPELINE_ROOT, "runs", args.version)

    # 검증
    if not args.skip_train and not args.data:
        print("[ERROR] --data 가 필요합니다 (--skip-train 미지정)", file=sys.stderr)
        sys.exit(1)
    if args.skip_train and not args.checkpoint:
        print("[ERROR] --skip-train 시 --checkpoint 가 필요합니다", file=sys.stderr)
        sys.exit(1)

    passed = run_pipeline(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
