"""AIMO Interp — CoT trajectory-geometry submission.

2층  model_id에 맞는 아티팩트가 있으면 모델을 한 번 로드해 문제마다 CoT를 생성하고
     hidden state 경로의 기하 특징으로 판정
1층  아티팩트가 없거나 2층이 죽으면 FALLBACK
시간: 문제당 생성 max_new_tokens(아티팩트에 기록)만큼. 3600초는 배치 전체 합산이라
     아티팩트를 만들 때 이 값을 보수적으로 잡는다.
"""

import traj_lib

FALLBACK = False


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
        out = traj_lib.predict(model_id, problems)
    except Exception:
        return fallback
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return fallback
    return out
