#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from context_emotion.captcha_bank.common import atomic_write_csv, read_csv, read_latest_jsonl

_HERE = Path(__file__).resolve().parent
_DATA_ROOT = Path("/workspace/data/context_emotion")


def main():
    p = argparse.ArgumentParser(
        description="VLM 공격 결과를 바탕으로 인간 검수 큐를 생성한다."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=_DATA_ROOT / "processed" / "context_emotion_train_dataset_v2.csv",
        help="학습 데이터셋 CSV (provisional_emotion 열 포함)",
    )
    p.add_argument(
        "--models-config",
        type=Path,
        default=_HERE / "models.json",
        help="VLM 모델 키 목록 (captcha_bank/models.json)",
    )
    p.add_argument(
        "--attacks-dir",
        type=Path,
        default=_HERE / "artifacts" / "attacks",
        help="VLM 공격 결과 JSONL 디렉터리 ({model_key}.jsonl)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_HERE / "artifacts" / "human_review_queue.csv",
    )
    p.add_argument(
        "--min-successful-models",
        type=int,
        default=2,
        help="검수 큐 포함 최소 성공 모델 수",
    )
    p.add_argument(
        "--min-wrong-models",
        type=int,
        default=2,
        help="검수 큐 포함 최소 오답 모델 수 (공격 난이도 기준)",
    )
    p.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="ambiguous 판정 문항도 큐에 포함",
    )
    a = p.parse_args()

    rows, fields = read_csv(a.input_csv)
    keys = list(json.loads(a.models_config.read_text()))
    outputs = {k: read_latest_jsonl(a.attacks_dir / f"{k}.jsonl") for k in keys}
    queue = []

    for row in rows:
        if not row["provisional_emotion"].strip():
            continue
        results = [(k, outputs[k].get(row["sample_id"])) for k in keys]
        results = [(k, v) for k, v in results if v and v.get("status") == "ok"]
        successful = len(results)
        wrong = sum(v.get("emotion") != row["provisional_emotion"] for _, v in results)
        ambiguous = sum(bool(v.get("ambiguous")) for _, v in results)
        hard = successful >= a.min_successful_models and wrong >= a.min_wrong_models
        if not hard and not (a.include_ambiguous and ambiguous and successful >= a.min_successful_models):
            continue
        predictions = {
            k: {
                "emotion": v.get("emotion"),
                "confidence": v.get("confidence"),
                "ambiguous": v.get("ambiguous"),
                "evidence": v.get("visual_evidence"),
            }
            for k, v in results
        }
        queue.append({
            **row,
            "successful_models": successful,
            "wrong_models": wrong,
            "ambiguous_models": ambiguous,
            "attack_hardness": round(wrong / successful, 4),
            "model_predictions": json.dumps(predictions, ensure_ascii=False),
            "human_decision": "",
            "human_emotion": "",
            "human_confidence": "",
            "human_note": "",
        })

    queue.sort(key=lambda x: (-float(x["attack_hardness"]), -int(x["wrong_models"]), x["sample_id"]))
    extra = [
        "successful_models", "wrong_models", "ambiguous_models",
        "attack_hardness", "model_predictions",
        "human_decision", "human_emotion", "human_confidence", "human_note",
    ]
    atomic_write_csv(a.output, queue, fields + extra)
    print(f"review_queue={len(queue)} output={a.output}")


if __name__ == "__main__":
    main()
