"""AIMO Interp — submission #TT-PERTURB: test-time perturbation, behaviour-based.

The judged model itself is loaded (bf16), asked to solve the original problem and K answer-preserving
rule-based perturbations (rename x2, typos x2, distractor x2, clause reorder, notation rewrite -- no
LLM rewriter, all deterministic), each with a short chain of thought that is then cut off by forcing
"</think>\nANSWER:". The score is the fraction of perturbations whose forced answer differs from the
original's. Low fraction => robust. The threshold is calibrated on the organisers' public
train-main-v2 problem-level labels so that the predicted True rate matches that label's robust rate.

This is a behavioural measurement, not an interpretation of internal states: nothing is read out of
the hidden layers. Forcing the answer is what makes it affordable -- token-cap saturation, which
killed our generation-length submissions, cannot happen here because every sequence is made to emit
an answer regardless of how far the reasoning got.

Falls back to the length rule (h15C, -log word count, fixed threshold) whenever the model cannot be
opened, the time budget cannot buy the smallest cap, or an answer cannot be extracted.
"""

import ttp_lib

FALLBACK = True


def are_robust(model_id: str, reasoning_effort: str = "default",
               problems: list[str] | None = None) -> list[bool]:
    if problems is None:
        problems = []
    if not isinstance(reasoning_effort, str):      # legacy positional call are_robust(model_id, problems)
        problems, reasoning_effort = list(reasoning_effort), "default"
    problems = list(problems)
    if not problems:
        return []
    try:
        out = ttp_lib.decide(problems, model_id, reasoning_effort)
    except Exception as e:                          # 채점기는 예외를 조용히 삼킨다 → 여기서 반드시 잡는다
        try:
            ttp_lib.log(f"are_robust 예외 {type(e).__name__} {str(e)[:160]}")
            out = ttp_lib.h15c_decide(problems)
        except Exception:
            out = [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        out = [FALLBACK] * len(problems)
    return out
