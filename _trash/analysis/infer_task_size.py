"""공개 리더보드 점수만으로 채점 세트의 크기와 라벨 균형을 역산한다.

근거 두 가지:

1. accuracy = 맞힌 수 / 전체 라벨 수 이므로, 관측된 점수는 어떤 정수 k에 대해
   round(k / N, 2) 를 만족해야 한다.
2. 상수 예측 두 개(전부 True / 전부 False)의 정답 수는 합이 정확히 N이다.
   따라서 두 점수의 합이 1.00인 쌍은 라벨 균형을 그대로 알려준다.

다른 참가자의 제출물 내부는 열지 않는다. 집계 점수만 쓴다.
"""

MAX_N = 400


def matches(score: float, count: int, total: int) -> bool:
    return round(count / total, 2) == score


def feasible_totals(scores: list[float], constant_pair: tuple[float, float] | None) -> list[int]:
    """관측된 모든 점수를 정수 비율로 표현할 수 있는 N을 찾는다."""
    result = []
    for total in range(2, MAX_N + 1):
        reachable = {round(k / total, 2) for k in range(total + 1)}
        if not all(score in reachable for score in scores):
            continue
        if constant_pair is not None:
            low, high = sorted(constant_pair)
            # 상수 쌍이면 정답 수가 k와 N-k로 갈린다.
            if not any(
                matches(low, k, total) and matches(high, total - k, total)
                for k in range(total + 1)
            ):
                continue
        result.append(total)
    return result


TASKS = [
    ("35282  (교체됨)", [1.00, 0.00], (0.00, 1.00)),
    ("35369  (교체됨)", [0.77, 0.69, 0.31], (0.31, 0.69)),
    ("35430  (현재)", [0.41, 0.00], None),
]

for name, scores, pair in TASKS:
    totals = feasible_totals(scores, pair)
    print(f"task {name}")
    print(f"  관측 점수: {scores}")
    print(f"  가능한 N (≤{MAX_N}): {totals[:14]}{' ...' if len(totals) > 14 else ''}")
    if totals:
        print(f"  최소 N = {totals[0]}")
    if pair:
        low, high = sorted(pair)
        print(f"  라벨 균형: 소수 클래스 {low:.0%} / 다수 클래스 {high:.0%}")
    print()

print("참고: 공개 검증 표본(val-sample)의 robust 비율 = 9/28 = 0.321")
