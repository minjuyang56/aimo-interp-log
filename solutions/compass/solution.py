"""AIMO Interp — submission compass (Small): solve-trajectory compass with fixed per-model thresholds.

Pre-registered 2026-09-29 (directive #26-2). Not tuned against the leaderboard. See compass_lib.py.
"""

import compass_lib

FALLBACK = True


def are_robust(model_id: str, reasoning_effort: str = "default",
               problems: list[str] | None = None) -> list[bool]:
    if problems is None:
        problems = []
    if not isinstance(reasoning_effort, str):          # legacy positional call are_robust(model_id, problems)
        problems, reasoning_effort = list(reasoning_effort), "default"
    if not problems:
        return []
    try:
        out = compass_lib.decide(model_id, list(problems))
    except Exception:
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
