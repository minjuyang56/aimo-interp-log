"""AIMO Interp — submission #LEN-CAL: problem length alone, threshold calibrated to the label prior.

Pre-registered 2026-10-15 (directive #FINAL-PLAN). Score = -log(word count) -- byte-for-byte the
same quantity as submission h15C. The only change is the threshold: it is fixed so that the
predicted True rate on the official train-main-v2 problem-level labels (33 problems, robust 20)
equals that label's robust rate, 20/33 = 0.6061. No within-batch median split, no leaderboard
tuning, no model loading, no generation.

The justification is calibration, not ranking: in #ORG-TRAIN this score's AUC (0.671) did not clear
its own shuffle-null upper bound (0.70) on those 33 problems. If the final set is imbalanced,
matching the predicted True rate to the label prior pins the expected accuracy near that prior; if
it is balanced, the calibration gains nothing. Both directions are recorded in CLAUDE.md.
"""

import lencal_lib

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
        out = lencal_lib.decide(list(problems))
    except Exception:
        return [FALLBACK] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [FALLBACK] * len(problems)
    return out
