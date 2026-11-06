"""AIMO Interp — submission hprobe (Small): a per-model linear probe on one prefill pass.

Pre-registered 2026-10-08 (directive #77, commit e749b2e). For each problem we run the judged model forward
exactly once -- no generation at all -- mean-pool the hidden state over the problem-body tokens at a layer fixed
in advance, and apply that model's own probe weights:

    score = ((h - mean)/scale - pca_mean) @ comps.T @ coef + intercept       (larger = robust)

Judgement is a within-call median split, never an absolute threshold, because the organisers warn the final set
may not be balanced. Models for which we have no labels yet fall back to a text rule (n_sym rank), not to another
model's probe: hidden sizes differ (2560/3584/4096) and a shared instrument predicts per problem, which is capped
near 0.50 on a per-model balanced set.

This is the pipeline, not yet the result. The only weights available today come from 88 self-generated Skywork
labels and their internal CV AUC is 0.454 -- below chance, matching the pilot's nested 0.367. Weights are swapped
by re-running analysis/build_hprobe_artifacts.py once the DeepSeek and Olmo label runs finish; no code changes.

The argument order is resolved defensively: the competition has used both
(model_id, reasoning_effort, problems) and (model_id, problems, reasoning_effort).
"""


import hprobe_lib

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
        out = hprobe_lib.decide(list(problems), model_id=str(model_id), effort=str(effort))
    except Exception as e:
        try:
            hprobe_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        except Exception:
            pass
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
