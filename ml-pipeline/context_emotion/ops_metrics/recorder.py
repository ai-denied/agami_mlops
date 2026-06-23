"""Append-only JSONL writer/reader for ops_metrics/schema.md.

No actual production caller exists yet (the CAPTCHA serving side that
would call record_daily_metrics() is out of this scaffold's scope) - this
just defines the schema as code so evaluation/promotion_gate.py and
evaluate_candidate.py have a single typed way to read it once it exists.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from context_emotion.deployment.model_store import load_runtime_contract


def events_path() -> str:
    return load_runtime_contract()["ops_metrics"]["events_path"]


@dataclass
class DailyMetrics:
    date: str
    model_version: str
    exposures: int
    human_correct_rate: Optional[float] = None
    human_ambiguous_rate: Optional[float] = None
    class_selection_distribution: Dict[str, int] = field(default_factory=dict)
    attacker_proxy_solve_rate: Optional[float] = None
    attacker_proxy_error_types: Optional[Dict[str, int]] = None
    excluded_question_rate: Optional[float] = None
    pending_review_rate: Optional[float] = None
    recorded_at: str = field(default_factory=lambda: datetime.now().isoformat())


def record_daily_metrics(metrics: DailyMetrics, path: Optional[str] = None) -> str:
    """TODO: 실제 운영 코드(서빙 측)에서 하루 끝에 호출해야 함 - 지금은
    호출자가 없다. path를 None으로 두면 runtime_contract.yaml 기본 경로.

    학습 데이터 csv 경로와 같은 파일에 쓰지 않는다는 걸 코드로도 강제한다 -
    MLOPS_OPERATION_DESIGN.md 5장의 "절대 같은 파일/정리 정책을 공유하지
    않는다" 원칙이 설정 실수로 깨지는 걸 막는 마지막 방어선
    (예: runtime_contract.yaml을 잘못 고쳐서 두 경로가 같아지는 경우)."""
    path = path or events_path()
    training_csv = load_runtime_contract()["training_dataset"]["train_csv"]
    if os.path.realpath(path) == os.path.realpath(training_csv):
        raise ValueError(
            f"ops_metrics.events_path가 학습 데이터 csv와 같은 경로입니다 ({path}). "
            "runtime_contract.yaml 설정 오류로 보입니다 - 절대 같은 파일에 섞어 쓰면 안 됩니다."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")
    return path


def read_recent_metrics(n: int = 30, path: Optional[str] = None) -> List[dict]:
    path = path or events_path()
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[-n:]


def latest_for_version(model_version: str, path: Optional[str] = None) -> List[dict]:
    return [r for r in read_recent_metrics(n=10_000, path=path) if r["model_version"] == model_version]
