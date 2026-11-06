"""AIMO Interp — submission h20c (Small): original-solve length, step-budgeted.

Pre-registered 2026-10-03 (directive #60). Decoding is memory-bandwidth bound, so wall time per
call is about (cap x ms/step) and barely depends on batch size or k. The library therefore spends
its per-call budget on the largest cap it can afford, puts every problem of the call into one
batch, and uses as many samples per problem as memory allows. Thresholds are per-cap and fixed on
the training problems; the text rule covers anything the budget cannot reach.
"""

import h20c_lib

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
        out = h20c_lib.decide(list(problems))
    except Exception as e:
        h20c_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        try:
            out = h20c_lib.fallback(list(problems))
        except Exception:
            return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
