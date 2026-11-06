"""AIMO Interp — submission #FLIPENT-ZIP: front-of-reasoning token entropy. EXPERIMENTAL.

The judged model itself is loaded and asked to solve the problem greedily for at most 1024 tokens.
A logits processor records the full-vocabulary entropy of every generated token; the score is the
mean entropy over the first 25% of the generated tokens (`ent_front`). HIGHER entropy means robust.
The threshold is fixed per model: the median `ent_front` over the organisers' public train-main-v2
33 problems, measured beforehand with the same generation setting. The official labels are never
used, so this is a prior-free absolute threshold, not an AUC-tuned one.

Sign and threshold were pre-registered in CLAUDE.md `## #FLIP-ENT` (commit 4fcd838) before the
measurement and are not flipped here.

EXPERIMENTAL, and the code says so because the evidence says so: on our own four self-labelled arms
this score did not clear its shuffle null anywhere (AUC 0.721 / 0.562 / 0.337 / 0.526 against null
upper bounds 0.740 / 0.686 / 0.756 / 0.618), and its fixed-threshold accuracy sat below the
all-robust constant on every arm. It is submitted as an experiment at the owner's instruction, as an
explicit exception to our standing rule of only building a zip for methods above 0.645 on the
official 33 -- a set this score was pre-registered never to be judged on.

Falls back to the length rule (h15C, -log word count, fixed threshold) whenever the model has no
stored threshold (e.g. gpt-oss-120b), cannot be opened, the budget cannot buy the full 1024-token
cap, generation fails, or the entropy trace is too short to take a quarter of.
"""

import flipent_lib

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
        out = flipent_lib.decide(problems, model_id, reasoning_effort)
    except Exception as e:                          # 채점기는 예외를 조용히 삼킨다 → 여기서 반드시 잡는다
        try:
            flipent_lib.log(f"are_robust 예외 {type(e).__name__} {str(e)[:160]}")
            out = flipent_lib.h15c_decide(problems)
        except Exception:
            out = [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        out = [FALLBACK] * len(problems)
    return out
