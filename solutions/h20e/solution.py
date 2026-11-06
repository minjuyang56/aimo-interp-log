"""AIMO Interp — submission h20e (Small): per-model generation, within-call rank split.

Pre-registered 2026-10-04 (directive #62). The organisers rebalanced the validation set so the
same problem can be robust for one model and spurious for another. Any method that broadcasts one
per-problem answer across models is therefore capped near 50 percent, which is what our earlier
entries scored. h20e drops the cross-model cache, generates with the model it is asked about,
folds reasoning_effort into the generation budget, and ranks within the call: the longer half of
the originals is called spurious. Everything else — two samples, the fitted ms/step cap model, the
deadline stop, the per-call budget — carries over from h20d.

The argument order is resolved defensively: the competition has used both
(model_id, reasoning_effort, problems) and (model_id, problems, reasoning_effort).
"""

import h20e_lib

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
        out = h20e_lib.decide(list(problems), model_id=str(model_id), effort=str(effort))
    except Exception as e:
        h20e_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        try:
            out = h20e_lib.fallback(list(problems))
        except Exception:
            return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
