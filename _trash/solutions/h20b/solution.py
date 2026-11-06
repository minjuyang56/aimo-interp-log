"""AIMO Interp — submission h20b (Small): original-solve length with a budget-aware setting.

Pre-registered 2026-10-02/03 (directives #55, #57). Same signal as h20 — how many tokens the
instrument model spends on the original problem — but the (k, cap) setting is chosen per call to
fit the remaining time, using thresholds fixed per setting on the training problems. Falls back to
the H15A text rule when the budget runs out, and returns True on any error.
"""

import h20b_lib

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
        out = h20b_lib.decide(list(problems))
    except Exception as e:
        h20b_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        try:
            out = h20b_lib.fallback(list(problems))
        except Exception:
            return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
