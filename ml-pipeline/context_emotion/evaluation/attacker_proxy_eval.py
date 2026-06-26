"""Attacker-proxy resistance evaluation.

VLM 공격 인프라는 context_emotion.captcha_bank 에 구현되어 있다:
  - captcha_bank/common.py          : build_attack_prompt(), parse_model_json()
  - captcha_bank/models.json        : 공격 VLM 목록 (Qwen2.5-VL-3B, SmolVLM2-2.2B 등)
  - captcha_bank/build_review_queue.py : 공격 결과 JSONL → 검수 큐 변환

_run_attacker_proxy()를 구현하려면:
  1. captcha_bank/models.json 의 VLM 중 하나를 proxy_model_path로 지정
  2. eval_pool_path 의 문항 CSV 에서 이미지를 로드하고
     captcha_bank/common.build_attack_prompt()로 프롬프트를 생성
  3. VLM 추론을 돌려 captcha_bank/common.parse_model_json()으로 파싱
  4. 정답(provisional_emotion)과 비교해 solve_rate / error breakdown 계산

config/promotion_policy.yaml 의 attacker_proxy_gate 가 required: false 인
동안은 not_configured 반환으로 충분하다. 구현 완료 후 required: true 로 전환.
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
                "_run_attacker_proxy()를 captcha_bank.common의 "
                "build_attack_prompt/parse_model_json으로 구현할 것"
            ),
        }
    return _run_attacker_proxy(candidate_model_path, proxy_model_path, eval_pool_path)


def _run_attacker_proxy(candidate_model_path: str, proxy_model_path: str, eval_pool_path: str) -> dict:
    """TODO: 구현 필요.

    구현 단계:
      1. eval_pool_path CSV 로드 (captcha_bank.common.read_csv)
      2. 각 이미지에 captcha_bank.common.build_attack_prompt() 적용
      3. captcha_bank/models.json 에서 proxy_model_path에 해당하는 VLM 로드
      4. VLM 추론 → captcha_bank.common.parse_model_json() 파싱
      5. provisional_emotion 과 비교해 solve_rate / error breakdown 집계
      6. {"status": "available", "attacker_solve_rate": ..., ...} 반환
    """
    raise NotImplementedError(
        "attacker proxy 모델/평가 풀이 아직 확정되지 않았습니다. "
        "evaluate_attacker_proxy()의 not_configured 분기를 거치지 않고 "
        "여기로 들어왔다면 호출부 버그입니다."
    )
