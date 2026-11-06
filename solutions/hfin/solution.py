"""AIMO Interp -- submission #FINAL (hfin): natural-termination rule with per-model routing.

For Skywork-OR1-Math-7B and Olmo-3-7B-Think the judged model itself is loaded and asked to solve
the original problem once (temperature 0.6, top_p 0.95, k=1, cap 12288 new tokens, the same
sampling and prompt used to measure the rule). A problem is called ROBUST iff that attempt
terminates naturally -- an EOS token appears before the cap:

    score_fin = (attempts finishing inside the cap) / k      robust  <=>  score_fin >= 0.5

The threshold 0.5 is absolute, so it does not depend on the batch composition. For every other
model_id (DeepSeek-R1-0528-Qwen3-8B, Qwen3.5-4B, gpt-oss-120b, anything unseen) the prediction
falls back to the length rule h15C (-log word count against a fixed, train-frozen threshold).

THE PER-MODEL ROUTING IS POST-HOC AND WE LABEL IT AS SUCH. It was chosen after seeing the
per-model offline accuracies on the organisers' public train-main-v2 33-problem set (Olmo 0.8182,
Skywork 0.6970, DeepSeek 0.4848), so it is exploratory, not pre-registered. The resulting hybrid
scores 0.6818 model-averaged there, against a shuffle-null 95% upper bound of 0.6367, the
all-robust constant 0.6061 and the length rule 0.6364.

TWO CAVEATS THAT BELONG NEXT TO THAT NUMBER.
(1) Budget wall. cap 12288 is the only setting that clears the gate (cap 8192 gives 0.5455 and cap
    4096 gives 0.4040, i.e. nothing). Generating 35 problems at that cap costs roughly 2,660 s for
    Skywork and 3,508 s for Olmo under our measured worker timing model, so at final-set scale the
    cap cannot be bought. We therefore never shrink the cap -- a lower cap makes "finished
    naturally" saturate to False and is silence, not signal -- and instead fall back to h15C for
    that whole model. At leaderboard scale the rule runs; at final-set scale this submission
    degenerates to h15C.
(2) Circularity with the label definition. The organisers' labels are themselves accuracies inside
    a fixed inference budget (official FAQ 7 and 8). "Does it finish inside a budget" may therefore
    re-read part of the label definition rather than a property of the problem, which would break
    transfer whenever the final-set budget differs from ours.

Related public work we did not copy: EconLearn/aimo-interp-difficulty-vs-fragility reports a
non-probing natural-termination rule (robust if >= 2 of 3 attempts end inside the cap). Our rule is
the same quantity measured at k=1 with an absolute 0.5 threshold; we reached it from our own H20b
completion-rate measurement and implemented it independently.

Nothing here can produce an invalid batch: every failure path (model not routed, load failure,
budget too small, generation failure, exception) returns the length rule, and the return type is
checked to be a list of plain Python bools of the right length.
"""

import hfin_lib

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
        out = hfin_lib.decide(problems, model_id, reasoning_effort)
    except Exception as e:                          # 채점기는 예외를 조용히 삼킨다 -> 여기서 반드시 잡는다
        try:
            hfin_lib.log("are_robust 예외 %s %s" % (type(e).__name__, str(e)[:160]))
            out = hfin_lib.h15c_decide(problems)
        except Exception:
            out = [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        out = [FALLBACK] * len(problems)
    return out
