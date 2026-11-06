"""AIMO Interp — submission h22 (Small): a test-time reproduction of the organisers' own label definition.

Pre-registered 2026-10-10 (directive #H22). The competition paper (arXiv 2607.13899) defines a problem as
spurious when the model solves the original reliably under repeated sampling but gets it wrong on the *majority*
of answer-preserving perturbations; ambiguous cases were excluded from the released labels. The perturbation
families are a mixture (Rephrase, Rename, Domain, Distract, Typos, Expert), not a single edit.

Our earlier entries missed that definition in two ways: they used a drop-magnitude threshold instead of a
majority vote, and they measured a single constant-only edit. This entry reproduces the definition directly:

    score_spurious = P(stable on the original) x (fraction of perturbations whose majority answer is wrong)

The sign is fixed in advance -- a higher score means spurious -- and the decision uses an absolute threshold of
0.5. That threshold is principled here, unlike in our previous entries: the score is the probability stated by
the label definition itself, so 0.5 is exactly the definition's "majority", and it does not assume the final set
is balanced (the organisers warn that it may not be).

Two honest substitutions are documented in h22_lib.py. The gold answer is unknown at test time, so the majority
answer of the original samples serves as a pseudo-gold and the first factor becomes self-consistency rather than
accuracy. Constant-changing edits are excluded from the primary score because the perturbed problem's new answer
cannot be computed at test time; they remain available only as an off-by-default auxiliary term.

Cost is the binding constraint. Under the official FAQ reading of the time limit (3600 s for the entire
prediction run) no configuration of this method fits, so the entry walks a pre-ordered ladder of
(samples, perturbations, samples-per-perturbation, token cap), picks the largest rung that the measured
remaining time and the model's KV-cache footprint both allow, and falls back to a text-only rule when even the
smallest rung is unaffordable. See analysis/h22_budget.py for the table.

The argument order is resolved defensively: the competition has used both
(model_id, reasoning_effort, problems) and (model_id, problems, reasoning_effort).
"""

import h22_lib

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
        out = h22_lib.decide(list(problems), model_id=str(model_id), effort=str(effort))
    except Exception as e:
        try:
            h22_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        except Exception:
            pass
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
