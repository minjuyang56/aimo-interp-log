"""AIMO Interp — submission h21c (Small): the model's own answer, re-scored under a constant-only edit.

Pre-registered 2026-10-06 (directives #65/#67/#68). For each problem we force the judged model to commit to an
answer (`</think>\nANSWER:` + 8 greedy tokens), then re-score that same answer string against a variant of the
problem in which one constant has been changed by a deterministic rule. The score is

    drop = logp(answer | original) - logp(answer | constant-edited)

with the sign fixed in advance: a larger drop means robust. A model that actually solves the problem abandons its
old answer once the constants change; a model that got there by memory or luck keeps ranking the same answer high.
On the 33 problem-level labels (local 4-bit, permutation null 1000) drop reached AUC 0.802 for DeepSeek-R1-Qwen3-8B
and 0.741 for Skywork-OR1-Math-7B, both outside the null in the registered direction.

Judgement is a within-call median split, never an absolute threshold, because the organisers warn the final set may
not be balanced. Problems whose constant edit cannot be built form a second tier split by answer-token entropy.
Generation is 8 tokens per problem, so the token-cap saturation that crippled the length-based entry cannot occur.

The argument order is resolved defensively: the competition has used both
(model_id, reasoning_effort, problems) and (model_id, problems, reasoning_effort).
"""

import h21cK_lib

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
        out = h21cK_lib.decide(list(problems), model_id=str(model_id), effort=str(effort))
    except Exception as e:
        try:
            h21cK_lib.log(f"decide 실패: {type(e).__name__} {str(e)[:160]}")
        except Exception:
            pass
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
