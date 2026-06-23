"""Attacker-proxy resistance evaluation - INTERFACE ONLY, not implemented.

There is currently no chosen attacker proxy model and no dedicated
adversarial eval pool (config/promotion_policy.yaml attacker_proxy_gate is
`required: false` for exactly this reason). This module defines the
contract evaluate_candidate.py / compare_candidate.py call against, so
that whoever picks the proxy model later only has to fill in
`_run_attacker_proxy()` - nothing else changes.

DO NOT fabricate a solve rate here. If the proxy isn't configured, return
status='not_configured' and let promotion_gate.py treat that gate as
non-blocking (see config/promotion_policy.yaml).
"""
from typing import Optional


def evaluate_attacker_proxy(
    candidate_model_path: str,
    proxy_model_path: Optional[str],
    eval_pool_path: Optional[str],
) -> dict:
    """Returns a dict matching the attacker_proxy section of
    contracts/evaluation_result.schema.json."""
    if not proxy_model_path or not eval_pool_path:
        return {
            "status": "not_configured",
            "attacker_solve_rate": None,
            "error_type_breakdown": None,
            "proxy_model_version": None,
            "todo": (
                "config/promotion_policy.yaml attacker_proxy_gate에 "
                "proxy_model_version / eval_pool_path를 채우고, "
                "_run_attacker_proxy()를 실제 어태커 모델 추론으로 구현할 것"
            ),
        }
    return _run_attacker_proxy(candidate_model_path, proxy_model_path, eval_pool_path)


def _run_attacker_proxy(candidate_model_path: str, proxy_model_path: str, eval_pool_path: str) -> dict:
    """TODO: 어태커 프록시 모델/평가 풀이 확정되면 여기를 구현.

    예상되는 작업:
      1. eval_pool_path의 운영 평가 풀(사람도 어태커도 둘 다 거쳐 가는
         별도 문항 세트 - 학습셋과는 분리, MLOPS_OPERATION_DESIGN.md 참고)을 로드
      2. candidate 모델로 각 문항의 "정답"(provisional_emotion)을 만들고,
         어태커 프록시 모델이 같은 문항에서 그 정답을 맞히는 비율을 계산
      3. 오답 유형(어떤 클래스를 어떤 클래스로 혼동하는지)을 집계
      4. {"status": "available", "attacker_solve_rate": ..., ...} 반환
    """
    raise NotImplementedError(
        "attacker proxy 모델/평가 풀이 아직 확정되지 않았습니다. "
        "evaluate_attacker_proxy()의 not_configured 분기를 거치지 않고 "
        "여기로 들어왔다면 호출부 버그입니다."
    )
