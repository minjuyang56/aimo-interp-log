"""AIMO Interp — submission B: length rule + within-batch median split. No model.

Pre-registered 2026-09-29. Not tuned against the leaderboard.
  score   = -(number of alphabetic words in the problem)   # shorter → more likely robust
            basis: on train-main-v2 (33 labelled problems) word count vs label has AUC 0.33,
            i.e. longer problems were labelled spurious; this is the only signal that left the
            label-shuffled null band, and it needs no model.
  verdict = within the batch (one are_robust call), the top ceil(n/2) scores are True.
            basis: the private set is balanced per model and per problem (organisers, 2026-09-22).
            ties keep input order.
  runtime ≈ 0.
"""

import re

WORD = re.compile(r"[A-Za-z]+")


def _score(problem: str) -> float:
    return -float(len(WORD.findall(problem)))


def _median_split(scores):
    n = len(scores)
    k = (n + 1) // 2
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    top = set(order[:k])
    return [bool(i in top) for i in range(n)]


def are_robust(model_id: str, reasoning_effort: str = "default",
               problems: list[str] | None = None) -> list[bool]:
    if problems is None:
        problems = []
    if not isinstance(reasoning_effort, str):          # legacy positional call
        problems, reasoning_effort = list(reasoning_effort), "default"
    if not problems:
        return []
    try:
        return _median_split([_score(p) for p in problems])
    except Exception:
        return [False] * len(problems)
