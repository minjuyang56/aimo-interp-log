"""AIMO Interp — submission h15A (Small): rename-surface rule, text only, no generation.

Pre-registered 2026-10-01 (directive #49). Mode A:
  A = raw n_sym · B = n_sym residualised on log word count · C = log word count alone (control).
Thresholds are the midpoint of the two class medians on the official 33 training problems,
fixed before submission and never tuned on the leaderboard. No within-batch median split.
"""

import h15_lib

MODE = "A"
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
        out = h15_lib.decide(list(problems), MODE)
    except Exception:
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
