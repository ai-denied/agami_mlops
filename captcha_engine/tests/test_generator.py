"""
flashlight_generator 의 핵심 알고리즘만 분리해서 표준 라이브러리로 검증.
Pydantic 의존성 없이도 로직 자체에 결함이 없는지 확인.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import math
import secrets
from datetime import datetime, timedelta, timezone
from app.captcha.flashlight_generator import generate_flashlight_challenge
from app.captcha.challenge_types import Difficulty


OBJECT_CATALOG = [
    {"id": "key", "label": "열쇠", "emoji": "🔑"},
    {"id": "coin", "label": "동전", "emoji": "🪙"},
    {"id": "letter", "label": "편지", "emoji": "✉️"},
    {"id": "gem", "label": "보석", "emoji": "💎"},
    {"id": "ring", "label": "반지", "emoji": "💍"},
    {"id": "scroll", "label": "두루마리", "emoji": "📜"},
]

DIFFICULTY_PROFILES = {
    "easy":   {"flashlight_radius": 0.18, "time_limit_sec": 60, "hint_after_sec": 15,
               "decoy_count": 0, "tolerance": 0.06, "edge_padding": 0.15, "min_separation": 0.25},
    "medium": {"flashlight_radius": 0.12, "time_limit_sec": 45, "hint_after_sec": 18,
               "decoy_count": 1, "tolerance": 0.05, "edge_padding": 0.10, "min_separation": 0.25},
    "hard":   {"flashlight_radius": 0.08, "time_limit_sec": 30, "hint_after_sec": None,
               "decoy_count": 2, "tolerance": 0.04, "edge_padding": 0.08, "min_separation": 0.20},
}


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def generate(difficulty="medium"):
    rng = secrets.SystemRandom()
    profile = DIFFICULTY_PROFILES[difficulty]
    decoy_count = profile["decoy_count"]

    chosen = rng.sample(OBJECT_CATALOG, k=1 + decoy_count)
    correct_obj = chosen[0]
    decoy_objs = chosen[1:]

    pad = profile["edge_padding"]
    min_sep = profile["min_separation"]

    correct_x = pad + (1 - 2 * pad) * rng.random()
    correct_y = pad + (1 - 2 * pad) * rng.random()
    placed = [(correct_x, correct_y)]
    decoys = []

    for obj in decoy_objs:
        for _ in range(50):
            x = pad + (1 - 2 * pad) * rng.random()
            y = pad + (1 - 2 * pad) * rng.random()
            if all(distance((x, y), p) >= min_sep for p in placed):
                placed.append((x, y))
                decoys.append({
                    "object_id": obj["id"],
                    "label": obj["label"],
                    "emoji": obj["emoji"],
                    "x": x,
                    "y": y,
                })
                break
        else:
            raise RuntimeError("decoy placement failed")

    challenge_id = secrets.token_urlsafe(16)
    now = datetime.now(timezone.utc)

    return {
        "challenge_id": challenge_id,
        "difficulty": difficulty,
        "target_hint": {"object_id": correct_obj["id"], "label": correct_obj["label"], "emoji": correct_obj["emoji"]},
        "decoys": decoys,
        "flashlight_radius": profile["flashlight_radius"],
        "time_limit_sec": profile["time_limit_sec"],
        "hint_after_sec": profile["hint_after_sec"],
        "_answer": {
            "correct_object_id": correct_obj["id"],
            "correct_x": correct_x,
            "correct_y": correct_y,
            "tolerance": profile["tolerance"],
        },
    }


def run_invariant_tests(n_iterations=2000):
    """n번 생성하면서 모든 invariant 가 깨지지 않는지 확인."""
    fails = []
    seen_ids = set()
    seen_correct_objs = {"easy": set(), "medium": set(), "hard": set()}

    for i in range(n_iterations):
        diff = ["easy", "medium", "hard"][i % 3]
        c = generate(diff)
        ans = c["_answer"]
        prof = DIFFICULTY_PROFILES[diff]

        # 1. challenge_id 고유성
        if c["challenge_id"] in seen_ids:
            fails.append(f"#{i} duplicate challenge_id")
        seen_ids.add(c["challenge_id"])

        # 2. 좌표 범위
        if not (0 <= ans["correct_x"] <= 1 and 0 <= ans["correct_y"] <= 1):
            fails.append(f"#{i} correct out of [0,1]")

        # 3. edge_padding 준수
        if not (prof["edge_padding"] <= ans["correct_x"] <= 1 - prof["edge_padding"]):
            fails.append(f"#{i} correct_x violates edge_padding")
        if not (prof["edge_padding"] <= ans["correct_y"] <= 1 - prof["edge_padding"]):
            fails.append(f"#{i} correct_y violates edge_padding")

        # 4. 미끼 개수
        if len(c["decoys"]) != prof["decoy_count"]:
            fails.append(f"#{i} decoy count mismatch")

        # 5. 정답-미끼, 미끼-미끼 최소 거리
        all_pos = [(ans["correct_x"], ans["correct_y"])] + [(d["x"], d["y"]) for d in c["decoys"]]
        for a in range(len(all_pos)):
            for b in range(a + 1, len(all_pos)):
                if distance(all_pos[a], all_pos[b]) < prof["min_separation"]:
                    fails.append(f"#{i} min_separation violated: {distance(all_pos[a], all_pos[b]):.3f}")

        # 6. 정답 객체 ID와 미끼 객체 ID 가 겹치지 않음
        decoy_ids = {d["object_id"] for d in c["decoys"]}
        if ans["correct_object_id"] in decoy_ids:
            fails.append(f"#{i} correct object id appears in decoys")

        # 7. 시간 제한 일관성 확인
        if c["time_limit_sec"] != prof["time_limit_sec"]:
            fails.append(f"#{i} time_limit mismatch")

        seen_correct_objs[diff].add(ans["correct_object_id"])

    print(f"총 {n_iterations}회 생성 시도, 실패 {len(fails)}건")
    if fails[:5]:
        for f in fails[:5]:
            print("  -", f)
    print(f"고유 challenge_id 개수: {len(seen_ids)} / {n_iterations}")
    print(f"난이도별 등장 정답 객체 종류:")
    for d, s in seen_correct_objs.items():
        print(f"  {d}: {sorted(s)}")
    return len(fails) == 0


def run_production_bundle_tests(n_iterations: int = 200) -> bool:
    """프로덕션 `generate_flashlight_challenge()` 직접 호출.
    1챌린지 = 3장 묶음 구조 invariant 검증 (production drift 방지).
    """
    fails: list[str] = []
    for i in range(n_iterations):
        difficulty = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD][i % 3]
        spec, answer = generate_flashlight_challenge(difficulty)

        # 1) 묶음 길이 == 3
        if len(spec.sub_challenges) != 3:
            fails.append(f"#{i} spec.sub_challenges len != 3 ({len(spec.sub_challenges)})")
        if len(answer.sub_answers) != 3:
            fails.append(f"#{i} answer.sub_answers len != 3 ({len(answer.sub_answers)})")

        # 2) 인덱스 셋트 == {0,1,2}
        spec_indices = {s.index for s in spec.sub_challenges}
        ans_indices = {a.index for a in answer.sub_answers}
        if spec_indices != {0, 1, 2}:
            fails.append(f"#{i} spec indices {spec_indices} != {{0,1,2}}")
        if ans_indices != {0, 1, 2}:
            fails.append(f"#{i} answer indices {ans_indices} != {{0,1,2}}")

        # 3) sub별 정답 객체 ID가 decoys에 안 들어감
        for sub in spec.sub_challenges:
            decoy_ids = {d.object_id for d in sub.decoys}
            if sub.target_hint.object_id in decoy_ids:
                fails.append(f"#{i} sub{sub.index} target appears in decoys")

        # 4) sub_answers와 sub_challenges가 인덱스로 매칭되며 target_hint와 일치
        for sa in answer.sub_answers:
            matching = next((s for s in spec.sub_challenges if s.index == sa.index), None)
            if matching is None:
                fails.append(f"#{i} sa.index={sa.index} no matching sub_challenge")
                continue
            if matching.target_hint.object_id != sa.correct_object_id:
                fails.append(f"#{i} sub{sa.index} target/answer mismatch")

        # 5) flashlight_radius / time_limit_sec / hint_after_sec는 번들 단위
        if not (0 < spec.flashlight_radius <= 1.0):
            fails.append(f"#{i} flashlight_radius out of range: {spec.flashlight_radius}")
        if spec.time_limit_sec <= 0 or spec.time_limit_sec > 300:
            fails.append(f"#{i} time_limit_sec out of range: {spec.time_limit_sec}")

    print(f"프로덕션 번들 invariant 검증: 총 {n_iterations}회, 실패 {len(fails)}건")
    for f in fails[:5]:
        print("  -", f)
    return len(fails) == 0


if __name__ == "__main__":
    import json

    print("=== 샘플 출력 (medium) — 병렬 알고리즘 검증 ===")
    sample = generate("medium")
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print()
    print("=== 병렬 Invariant 검증 (2000회) ===")
    ok_parallel = run_invariant_tests(2000)
    print("결과:", "PASS" if ok_parallel else "FAIL")
    print()
    print("=== 프로덕션 번들 검증 (200회) ===")
    ok_bundle = run_production_bundle_tests(200)
    print("결과:", "PASS" if ok_bundle else "FAIL")
    sys.exit(0 if (ok_parallel and ok_bundle) else 1)
