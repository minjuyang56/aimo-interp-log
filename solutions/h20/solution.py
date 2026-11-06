"""AIMO Interp — submission h20 (Small): solve the original once, judge by generation length.

Pre-registered 2026-10-02 (directives #55, #56). On the official 33 training problems the mean
number of tokens the model generates for the original problem separates robust from spurious at
AUC 0.850 (0.792 after residualising on problem length), with no perturbation needed.
One instrument model (Skywork) serves every model_id because the labels are problem-level, and a
problem-text cache keeps repeated groups from paying twice. Thresholds were fixed on the training
problems as the midpoint of the two class medians, never tuned on the leaderboard.
"""

import h20_lib

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
        out = h20_lib.decide(list(problems))
    except Exception as e:
        h20_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        try:
            out = h20_lib.fallback(list(problems))
        except Exception:
            return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
