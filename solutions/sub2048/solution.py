"""AIMO Interp — submission sub2048 (Small): a per-model probe on the mid-solution hidden state.

Directive #SUB-GEN2048 (2026-10-12). For each judged model we let the model itself write up to 2048
greedy tokens of its own solution, mean-pool the hidden state over those solution tokens at layer L//2
(fixed in advance), and apply that model's own weights (StandardScaler -> PCA 24 -> logistic):

    score = ((h - mean)/scale - pca_mean) @ comps.T @ coef + intercept       (larger = robust)

Judgement is a per-model absolute threshold fixed on train (the median of the train scores), not a
within-call median split: the calls here hold only 3-5 problems each, and the logistic's own boundary
would predict True 81% of the time because our labels are skewed to robust (26/4, 21/10, 29/3).

Why window 2048: in the #DS-RELABEL re-evaluation it was the only window that cleared both
pre-registered gates on both label versions (model-mean AUC 0.871 / 0.872, above every baseline),
while the pre-registered main window of 512 failed both. The verdict there stayed "signal not
established" and 2048 was recorded as the one open candidate needing an independent pre-registered
repeat. A leaderboard run is that repeat, on the organisers' held-out set.

Because greedy decoding is deterministic, the first 2048 tokens of a 2048-token generation are
identical to the first 2048 tokens of the 32768-token generations used in training -- so capping
generation does not change the feature.

Models without weights (Qwen3.5-4B, gpt-oss-120b, unknown ids), an insufficient time budget, or any
failure fall back to h15C (-log word count, threshold fixed on train), our best-scoring submission so
far (0.57). The model is not even loaded in that case.

The argument order is resolved defensively: the competition has used both
(model_id, reasoning_effort, problems) and (model_id, problems, reasoning_effort).
"""

import sub2048_lib

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
        out = sub2048_lib.decide(list(problems), model_id=str(model_id), effort=str(effort))
    except Exception as e:
        try:
            sub2048_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        except Exception:
            pass
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
