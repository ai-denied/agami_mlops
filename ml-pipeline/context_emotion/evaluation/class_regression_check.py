"""Per-class F1 regression check for the critical/low-resource classes.

config/promotion_policy.yaml class_regression_gate.critical_classes is the
list of emotion classes this gate watches (currently the 3 low_resource
classes from docs/context_emotion_label_distribution_v2.md plus
'confusion' - see the comment in that yaml for why). Thresholds are TODO
(null) until a real candidate's evaluation_result.json gives the policy
owner something concrete to set them against.
"""
from typing import Dict, List, Optional


def check_class_regression(
    current_per_class: Dict[str, dict],
    candidate_per_class: Dict[str, dict],
    critical_classes: List[str],
    max_f1_drop_absolute: Optional[float],
    max_f1_drop_relative_pct: Optional[float],
) -> dict:
    """Returns {"status": "not_configured"|"evaluated", "pass": bool|None,
    "regressions": [...], "details": [...]}."""
    if max_f1_drop_absolute is None and max_f1_drop_relative_pct is None:
        return {
            "status": "not_configured",
            "pass": None,
            "regressions": [],
            "details": [],
            "todo": "promotion_policy.yaml class_regression_gate에 "
                    "max_f1_drop_absolute 또는 max_f1_drop_relative_pct를 설정할 것",
        }

    details = []
    regressions = []
    for cls in critical_classes:
        cur = current_per_class.get(cls)
        cand = candidate_per_class.get(cls)
        if cur is None or cand is None:
            details.append({"class": cls, "skipped": True, "reason": "missing in current or candidate per_class"})
            continue

        cur_f1, cand_f1 = cur["f1"], cand["f1"]
        drop_abs = cur_f1 - cand_f1
        drop_rel_pct = (drop_abs / cur_f1 * 100) if cur_f1 else None

        violates_abs = max_f1_drop_absolute is not None and drop_abs > max_f1_drop_absolute
        violates_rel = (
            max_f1_drop_relative_pct is not None
            and drop_rel_pct is not None
            and drop_rel_pct > max_f1_drop_relative_pct
        )

        row = {
            "class": cls,
            "current_f1": cur_f1,
            "candidate_f1": cand_f1,
            "drop_absolute": drop_abs,
            "drop_relative_pct": drop_rel_pct,
            "violates": violates_abs or violates_rel,
        }
        details.append(row)
        if row["violates"]:
            regressions.append(row)

    return {
        "status": "evaluated",
        "pass": len(regressions) == 0,
        "regressions": regressions,
        "details": details,
    }
