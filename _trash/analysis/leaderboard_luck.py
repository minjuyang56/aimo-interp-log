"""리더보드 운 분석 — 검증 14문제에서 무작위 제출이 12/14 이상 찍을 확률과, 비상수 제출 수를 감안한 기대 횟수.

가설: 0.86(12/14) 3명은 운으로 설명 가능한가.
사용:  python analysis/leaderboard_luck.py [--n 14] [--nonconst 197] [--hits 3] [--dist "9:33,10:12,11:5,12:3"]
"""
import argparse
from math import comb, exp, factorial

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=14); ap.add_argument("--nonconst", type=int, default=197)
ap.add_argument("--hits", type=int, default=3); ap.add_argument("--k", type=int, default=12)
ap.add_argument("--dist", default="3:1,4:2,5:14,6:32,7:85,8:56,9:33,10:12,11:5,12:3")
a = ap.parse_args()
p_k = lambda k: comb(a.n, k) / 2 ** a.n
tail = sum(p_k(k) for k in range(a.k, a.n + 1))
lam = a.nonconst * tail
p_ge_hits = 1 - sum(exp(-lam) * lam ** i / factorial(i) for i in range(a.hits))
print(f"검증 N={a.n} · 동전 제출 1회가 {a.k}/{a.n} 이상일 확률 = {tail:.4f} (1/{1/tail:.0f})")
print(f"비상수 제출 {a.nonconst}건 → {a.k}/{a.n} 이상 기대 횟수 = {lam:.2f}")
print(f"최소 1건 확률 = {1 - exp(-lam):.2f} · 실제 관측 {a.hits}건 이상일 확률 (Poisson) = {p_ge_hits:.3f}")
print(f"\n관측 vs 동전 기대 (전체 제출 {sum(int(x.split(':')[1]) for x in a.dist.split(','))}건 기준, 상수 포함):")
obs = {int(x.split(":")[0]): int(x.split(":")[1]) for x in a.dist.split(",")}
tot = sum(obs.values())
print(f"{'k/14':>6} {'관측':>5} {'동전기대':>8} {'비율':>6}")
for k in sorted(obs):
    e = tot * p_k(k); print(f"{k:>4}/14 {obs[k]:5d} {e:8.1f} {obs[k] / e:6.2f}")
