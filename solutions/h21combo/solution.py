"""AIMO Interp — submission h21combo (Small): drop plus short-generation entropy.

Second-choice entry, and **explicitly post-hoc**: the combination was chosen after seeing the per-model results
of H21a-d, so its measured AUC is a design comparison, not a test, and this submission's leaderboard score is
the first confirmation data for it. Score = z(drop) + z(-mean_entropy), standardised within the call, no learned
weights, larger = robust. drop is the h21c quantity (the model's own forced answer re-scored under a
constant-only edit); mean_entropy is the average token entropy over the first 1024 generated tokens.

On the 33 problem-level labels the combination matches drop alone on the two strong models (0.796 vs 0.802;
0.742 vs 0.741) and lifts the two weak ones (Qwen 0.636 to 0.696; Olmo 0.466 to 0.531). The price is the
1024-token generation, about 54 s per model for 35 problems on the competition worker; when the per-call budget
cannot cover it the entry falls back to drop alone, which is exactly h21c.

Judgement is a within-call median split per tier, never an absolute threshold.

The argument order is resolved defensively: the competition has used both
(model_id, reasoning_effort, problems) and (model_id, problems, reasoning_effort).
"""

import h21combo_lib

FALLBACK = True


def _resolve(args, kwargs):
    """위치·키워드 어느 순서로 와도 problems(리스트)와 effort(문자열)를 찾아낸다."""
    problems, effort = None, "default"
    if "problems" in kwargs and kwargs["problems"] is not None:
        problems = list(kwargs["problems"])
    if isinstance(kwargs.get("reasoning_effort"), str):
        effort = kwargs["reasoning_effort"]
    for a in args:
        if problems is None and isinstance(a, (list, tuple)) and not isinstance(a, str):
            problems = list(a)
        elif isinstance(a, str) and effort == "default":
            effort = a
    return (problems or []), effort


def are_robust(model_id: str, *args, **kwargs) -> list[bool]:
    problems, effort = _resolve(args, kwargs)
    if not problems:
        return []
    try:
        out = h21combo_lib.decide(list(problems), model_id=str(model_id), effort=str(effort))
    except Exception as e:
        try:
            h21combo_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        except Exception:
            pass
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
