"""Aggregates the 5 gates into a single promotion_decision.

Gate outcome is always one of "pass" | "fail" | "not_configured".
"not_configured" is never silently treated as a pass - see
contracts/promotion_decision.schema.json. Each gate has a `required` flag
in config/promotion_policy.yaml: a required gate that is "not_configured"
forces manual_review; a non-required gate is informational only and never
blocks (used for human_ambiguity / attacker_proxy until those data sources
exist - see promotion_policy.yaml TODOs).

final_decision:
    any required gate == "fail"            -> reject
    elif any required gate == "not_configured" -> manual_review
    else                                    -> promote
"""
from datetime import datetime
from typing import Dict, Optional

from context_emotion.evaluation.class_regression_check import check_class_regression


def _overall_quality_gate(current_eval: dict, candidate_eval: dict, cfg: dict) -> dict:
    min_macro_f1 = cfg.get("min_macro_f1")
    min_accuracy = cfg.get("min_accuracy")
    max_drop = cfg.get("max_macro_f1_drop_vs_current")

    if min_macro_f1 is None and min_accuracy is None and max_drop is None:
        return {"status": "not_configured", "pass": None, "reason": "min_macro_f1/min_accuracy/max_macro_f1_drop_vs_current 미설정"}

    cand_overall = candidate_eval["overall"]
    cur_overall = current_eval["overall"] if current_eval else {}
    checks = []

    if min_macro_f1 is not None:
        checks.append((
            f"macro_f1 {cand_overall['macro_f1']:.4f} >= min_macro_f1 {min_macro_f1:.4f}",
            cand_overall["macro_f1"] >= min_macro_f1,
        ))
    if min_accuracy is not None:
        checks.append((
            f"accuracy {cand_overall['accuracy']:.4f} >= min_accuracy {min_accuracy:.4f}",
            cand_overall["accuracy"] >= min_accuracy,
        ))
    if max_drop is not None and cur_overall.get("macro_f1") is not None:
        drop = cur_overall["macro_f1"] - cand_overall["macro_f1"]
        checks.append((
            f"macro_f1 drop {drop:.4f} (current {cur_overall['macro_f1']:.4f} -> candidate {cand_overall['macro_f1']:.4f}) "
            f"<= max_macro_f1_drop_vs_current {max_drop:.4f}",
            drop <= max_drop,
        ))

    passed = all(ok for _, ok in checks)
    return {"status": "evaluated", "pass": passed, "checks": checks}


def _human_ambiguity_gate(current_eval: dict, candidate_eval: dict, cfg: dict) -> dict:
    amb = candidate_eval["human_ambiguity"]
    if amb["status"] == "not_configured":
        return {"status": "not_configured", "pass": None, "reason": "human_ambiguity 데이터 없음 (애매함 선택지 UX 미출시)"}

    min_exposures = cfg.get("min_exposures_required")
    if min_exposures is not None and amb["exposures"] < min_exposures:
        return {"status": "not_configured", "pass": None, "reason": f"노출 수 {amb['exposures']} < min_exposures_required {min_exposures}"}

    max_rate = cfg.get("max_ambiguous_rate")
    max_increase = cfg.get("max_ambiguous_rate_increase")
    if max_rate is None and max_increase is None:
        return {"status": "not_configured", "pass": None, "reason": "max_ambiguous_rate/max_ambiguous_rate_increase 미설정"}

    checks = []
    if max_rate is not None:
        checks.append((
            f"ambiguous_rate {amb['ambiguous_rate']:.4f} <= max_ambiguous_rate {max_rate:.4f}",
            amb["ambiguous_rate"] <= max_rate,
        ))
    if max_increase is not None and current_eval and current_eval["human_ambiguity"]["status"] == "available":
        increase = amb["ambiguous_rate"] - current_eval["human_ambiguity"]["ambiguous_rate"]
        checks.append((
            f"ambiguous_rate increase {increase:.4f} <= max_ambiguous_rate_increase {max_increase:.4f}",
            increase <= max_increase,
        ))

    return {"status": "evaluated", "pass": all(ok for _, ok in checks), "checks": checks}


def _attacker_proxy_gate(current_eval: dict, candidate_eval: dict, cfg: dict) -> dict:
    proxy = candidate_eval["attacker_proxy"]
    if proxy["status"] == "not_configured":
        return {"status": "not_configured", "pass": None, "reason": "attacker proxy 모델/평가 풀 미확정"}

    max_rate = cfg.get("max_attacker_solve_rate")
    max_increase = cfg.get("max_attacker_solve_rate_increase")
    if max_rate is None and max_increase is None:
        return {"status": "not_configured", "pass": None, "reason": "max_attacker_solve_rate/max_attacker_solve_rate_increase 미설정"}

    checks = []
    if max_rate is not None:
        checks.append((
            f"attacker_solve_rate {proxy['attacker_solve_rate']:.4f} <= max_attacker_solve_rate {max_rate:.4f}",
            proxy["attacker_solve_rate"] <= max_rate,
        ))
    if max_increase is not None and current_eval and current_eval["attacker_proxy"]["status"] == "available":
        increase = proxy["attacker_solve_rate"] - current_eval["attacker_proxy"]["attacker_solve_rate"]
        checks.append((
            f"attacker_solve_rate increase {increase:.4f} <= max_attacker_solve_rate_increase {max_increase:.4f}",
            increase <= max_increase,
        ))

    return {"status": "evaluated", "pass": all(ok for _, ok in checks), "checks": checks}


def _artifact_integrity_gate(candidate_eval: dict) -> dict:
    integrity = candidate_eval.get("artifact_integrity")
    if not integrity:
        return {"status": "evaluated", "pass": False, "reason": "evaluation_result.json에 artifact_integrity 없음"}
    checks = [(f"{k}=True", integrity.get(k) is True) for k in ("onnx_loadable", "input_output_match", "label_schema_match")]
    return {"status": "evaluated", "pass": all(ok for _, ok in checks), "checks": checks}


def decide(
    current_version: str,
    candidate_version: str,
    current_eval: Optional[dict],
    candidate_eval: dict,
    policy: dict,
) -> dict:
    overall_cfg = policy["overall_quality_gate"]
    regression_cfg = policy["class_regression_gate"]
    ambiguity_cfg = policy["human_ambiguity_gate"]
    attacker_cfg = policy["attacker_proxy_gate"]

    overall_result = _overall_quality_gate(current_eval, candidate_eval, overall_cfg)
    regression_result = check_class_regression(
        current_per_class=(current_eval or {}).get("per_class", {}),
        candidate_per_class=candidate_eval["per_class"],
        critical_classes=regression_cfg["critical_classes"],
        max_f1_drop_absolute=regression_cfg.get("max_f1_drop_absolute"),
        max_f1_drop_relative_pct=regression_cfg.get("max_f1_drop_relative_pct"),
    )
    ambiguity_result = _human_ambiguity_gate(current_eval, candidate_eval, ambiguity_cfg)
    attacker_result = _attacker_proxy_gate(current_eval, candidate_eval, attacker_cfg)
    artifact_result = _artifact_integrity_gate(candidate_eval)

    gate_results: Dict[str, dict] = {
        "overall_quality": overall_result,
        "class_regression": regression_result,
        "human_ambiguity": ambiguity_result,
        "attacker_proxy_resistance": attacker_result,
        "artifact_integrity": artifact_result,
    }
    required_flags = {
        "overall_quality": overall_cfg["required"],
        "class_regression": regression_cfg["required"],
        "human_ambiguity": ambiguity_cfg["required"],
        "attacker_proxy_resistance": attacker_cfg["required"],
        "artifact_integrity": policy["artifact_integrity_gate"]["required"],
    }

    def _detail_text(result: dict) -> str:
        """checks(list of (desc, bool)) -> 'desc1 [PASS]; desc2 [FAIL]'.
        class_regression_check.py's 'regressions' (the violating subset) takes
        priority over its full 'details' (every class checked) so the reason
        stays short - the full per-class breakdown is still in the raw dict
        for anyone who reads promotion_decision.json directly, just not
        repeated in the one-line reasons string."""
        if "reason" in result:
            return result["reason"]
        if result.get("checks"):
            return "; ".join(f"{desc} [{'PASS' if ok else 'FAIL'}]" for desc, ok in result["checks"])
        if result.get("regressions"):
            return "; ".join(
                f"{r['class']}: f1 {r['current_f1']:.4f}->{r['candidate_f1']:.4f} (drop {r['drop_absolute']:.4f})"
                for r in result["regressions"]
            )
        if result["status"] == "evaluated":
            return "all checks passed"
        return ""

    gates = {}
    reasons = []
    warnings = []
    for name, result in gate_results.items():
        status = result["status"] if result["status"] == "not_configured" else ("pass" if result["pass"] else "fail")
        gates[name] = status
        required = required_flags[name]
        suffix = "" if required else " (informational only - required=false)"
        reasons.append(f"{name}: {status}{suffix} - {_detail_text(result)}")

        # required=false 게이트가 pass가 아니면, 최종 판정이 promote라도
        # "이 게이트는 사실 검증 안 됐다/실패했다"가 눈에 안 보이고 넘어갈 수
        # 있음 - 특히 attacker_proxy_resistance가 not_configured인 채로
        # 계속 promote되면 방어력 검증 없이 모델이 계속 바뀌는 셈이라 별도
        # warnings로 항상 눈에 띄게 남긴다.
        if not required and status != "pass":
            warnings.append(f"{name} is {status} and NOT blocking promotion (required=false in promotion_policy.yaml) - {_detail_text(result)}")

    blocking = [name for name, status in gates.items() if required_flags[name] and status == "fail"]
    pending = [name for name, status in gates.items() if required_flags[name] and status == "not_configured"]

    if blocking:
        final_decision = "reject"
    elif pending:
        final_decision = "manual_review"
    else:
        final_decision = "promote"

    return {
        "current_version": current_version,
        "candidate_version": candidate_version,
        "decided_at": datetime.now().isoformat(),
        "gates": gates,
        "final_decision": final_decision,
        "reasons": reasons,
        "warnings": warnings,
    }
