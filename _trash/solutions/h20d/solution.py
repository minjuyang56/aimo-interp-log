"""AIMO Interp — submission h20d (Small): original-solve length, budgeted by a measured ms/step model.

Pre-registered 2026-10-03 (directives #60, #61). Worker probe2 refuted the earlier assumption: ms/step rises with batch size
(25.0 at 8, 54.4 at 32, 95.5 at 64 sequences) and roughly triples at 16k context. The library therefore spends
its per-call budget on the largest cap a fitted ms(batch, context) model allows, keeps one batch
because splitting only adds fixed overhead, defaults to two samples per problem, and stops
generation early against a deadline, judging on the steps actually taken. Thresholds are per-cap and fixed on
the training problems; the text rule covers anything the budget cannot reach.
"""

import h20d_lib

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
        out = h20d_lib.decide(list(problems))
    except Exception as e:
        h20d_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        try:
            out = h20d_lib.fallback(list(problems))
        except Exception:
            return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
