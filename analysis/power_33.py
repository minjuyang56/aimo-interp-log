"""검정력 분석 — n 문제로 참 AUC 가 얼마면 라벨 섞기 null 을 넘는가. (지휘 #26-1)

모형: 이항정규(binormal). robust 점수 ~ N(d, 1), spurious ~ N(0, 1), d = sqrt(2)·Phi^-1(AUC_true).
검출 판정: 관측 AUC > 그 시뮬레이션에서 뽑은 null 의 97.5 백분위(단측 alpha=0.025).

빠른 계산의 근거: 라벨을 섞은 null 의 AUC 분포는 **점수와 무관**하다(동점 없을 때).
  AUC = (양성 순위합 - n+(n+1)/2) / (n+ · n-) 이므로, 순위 1..n 중 n+ 개를 무작위로 고르는 것과 같다.
  그래서 null 임계값 분포는 (n, n+, perms) 당 한 번만 뽑아 참 AUC 값들에 재사용한다.
perms 를 30 과 1000 으로 나눠 보는 이유: perms 가 작으면 null 상단 추정이 흔들려 **검출력이 겉보기로 높아진다**(가짜 ▲).

사용:  .venv/Scripts/python.exe analysis/power_33.py [--sims 3000]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

CONFIGS = [(20, 13, "공식 train 33문제 (20/13)"),
           (24, 5, "자체 v3 라벨 29문제 (24/5)"),
           (36, 101, "옛 MATH 137문제 (36/101)"),
           (60, 40, "가상 100문제 (60/40)"),
           (120, 80, "가상 200문제 (120/80)")]
AUCS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.90)


def auc_from_scores(S, n_pos):
    n = S.shape[1]
    order = np.argsort(S, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, n + 1, dtype=np.float64), S.shape).copy(), axis=1)
    return (ranks[:, :n_pos].sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * (n - n_pos))


def null_aucs(m, n, n_pos, rng):
    pick = np.argsort(rng.random((m, n)), axis=1)[:, :n_pos] + 1
    return (pick.sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * (n - n_pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=3000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    out(f"검정력 (이항정규 · 시뮬 {a.sims}회 · 검출 = 관측 AUC > 라벨섞기 null 97.5%, 단측 alpha=0.025)")
    for n_pos, n_neg, tag in CONFIGS:
        n = n_pos + n_neg
        t30 = np.array([np.percentile(null_aucs(30, n, n_pos, rng), 97.5) for _ in range(a.sims)])
        t1k = np.array([np.percentile(null_aucs(1000, n, n_pos, rng), 97.5) for _ in range(a.sims)])
        out(f"\n## {tag} — null 97.5% 상단: perms=30 평균 {t30.mean():.3f}(sd {t30.std():.3f}) · "
            f"perms=1000 평균 {t1k.mean():.3f}(sd {t1k.std():.3f})")
        out(f"{'참 AUC':>7s} {'검출력(perms=30)':>16s} {'검출력(perms=1000)':>18s} {'관측 AUC 평균±sd':>20s}")
        for auc in AUCS:
            d = np.sqrt(2) * norm.ppf(auc)
            S = np.c_[rng.normal(d, 1, (a.sims, n_pos)), rng.normal(0, 1, (a.sims, n_neg))]
            obs = auc_from_scores(S, n_pos)
            out(f"{auc:7.2f} {float((obs > t30).mean()):16.2f} {float((obs > t1k).mean()):18.2f} "
                f"{obs.mean():12.3f} ± {obs.std():.3f}")
    out("\n메모 1. perms=30 열이 1000 열보다 높다 = 적은 섞기로 추정한 null 상단이 낮게 치우쳐 가짜 검출을 만든다.")
    out("메모 2. 공식 33문제(perms=1000): 참 AUC 0.70 에서 검출력 0.51, 0.75 에서 0.70 — 0.70 근처가 동전 던지기다.")
    (Path(__file__).resolve().parent.parent / "experiments" / "results" / "power_33_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
