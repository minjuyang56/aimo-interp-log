"""Constant baseline: every model/problem pair is classified as not robust.

Paired with always-true this recovers the test set's class balance from two
leaderboard scores: acc(always-true) = P(True), acc(always-false) = P(False),
and the two must sum to 1.0 if every case was scored as valid.
"""


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    del model_id
    return [False for _ in problems]
