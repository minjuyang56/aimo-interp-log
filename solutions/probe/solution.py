"""AIMO Interp — hidden-state probe submission.

2층 구조:
  2층  model_id에 맞는 프로브가 있으면 모델을 한 번 로드해 문제마다 forward 1회 → 판정
  1층  프로브가 없거나 2층이 어떤 이유로든 죽으면 FALLBACK
채점기는 예외를 삼키고 배치 전량을 무효 처리하므로, 여기서 먼저 잡는다.
"""

import probe_lib

FALLBACK = False   # 학습 데이터(DeepSeek 137문제) 기준 다수 클래스. 본선은 균형이라 정보 없음


def are_robust(model_id: str, reasoning_effort: str = "default",
               problems: list[str] | None = None) -> list[bool]:
    # 2026-09-22 계약: 채점기는 are_robust(model_id, reasoning_effort=..., problems=...) 로
    # 키워드 호출한다. 배치 = (model_id, effort) 쌍. 옛 2-인자 호출도 받도록 기본값을 둔다.
    if problems is None:
        problems = []
    if not isinstance(reasoning_effort, str):   # 옛 위치 호출 are_robust(model_id, problems)
        problems, reasoning_effort = list(reasoning_effort), "default"
    fallback = [FALLBACK] * len(problems)
    if not problems:
        return []
    try:
        out = probe_lib.predict(model_id, problems)
    except Exception:
        return fallback
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return fallback
    return out
