"""AIMO Interp — submission A: common-instrument probe (DeepSeek-8B, layer 18) + within-batch median split.

Pre-registered 2026-09-29. Not tuned against the leaderboard. See inst_lib.py for the design.
"""

import inst_lib

FALLBACK = False


def are_robust(model_id: str, reasoning_effort: str = "default",
               problems: list[str] | None = None) -> list[bool]:
    if problems is None:
        problems = []
    if not isinstance(reasoning_effort, str):          # legacy positional call are_robust(model_id, problems)
        problems, reasoning_effort = list(reasoning_effort), "default"
    if not problems:
        return []
    try:
        out = inst_lib.median_split(inst_lib.scores(problems))
    except Exception:
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
