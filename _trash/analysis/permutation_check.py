"""LOPO AUC가 0.5보다 훨씬 낮게 나올 때 — 진짜 (뒤집힌) 신호인가, leave-one-out 편향인가.

n=33에 class_weight="balanced"로 문제 하나씩 빼면, 뺀 문제의 반대쪽으로 경계가 살짝 밀려
신호가 없어도 AUC가 0.5 아래로 내려간다. 라벨을 섞어 같은 절차를 30번 돌린 분포와 비교한다.

  실제 값이 섞은 분포 범위 안  → 신호 없음 (편향)
  실제 값이 범위 밖           → 진짜 신호. 부호가 반대면 특징 방향이 학습·검증에서 뒤집히는 것

사용:  .venv/Scripts/python.exe analysis/permutation_check.py [--kind conf|hidden] [--n-tokens 512] [--layer 12]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
from extract_hidden import alias  # noqa: E402
from extract_train_v2 import LOCAL_MODELS  # noqa: E402
from train_train_v2 import lopo_margins  # noqa: E402


def kfold_auc(X, y, C, seed=0):
    m = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
        m[te] = clf.decision_function((X[te] - mu) / sd)
    return roc_auc_score(y, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["conf", "hidden"], default="conf")
    ap.add_argument("--n-tokens", type=int, default=512)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--perms", type=int, default=30)
    a = ap.parse_args()
    C = 1.0 if a.kind == "conf" else 0.05
    rng = np.random.default_rng(0)

    for m in LOCAL_MODELS:
        f = (HERE.parent / "experiments" / "results" /
             (f"trainv2_conf_{alias(m)}_N{a.n_tokens}.npz" if a.kind == "conf" else f"trainv2_hidden_{alias(m)}.npz"))
        if not f.is_file():
            continue
        d = np.load(f, allow_pickle=True)
        lab = d["y"] >= 0
        X = (d["F"] if a.kind == "conf" else d["H"][:, a.layer]).astype(np.float32)[lab]
        y = d["y"][lab].astype(int)
        real_l, real_k = roc_auc_score(y, lopo_margins(X, y, C)), kfold_auc(X, y, C)
        pl, pk = [], []
        for _ in range(a.perms):
            yp = rng.permutation(y)
            pl.append(roc_auc_score(yp, lopo_margins(X, yp, C))); pk.append(kfold_auc(X, yp, C))
        lo_l, hi_l = np.percentile(pl, [2.5, 97.5]); lo_k, hi_k = np.percentile(pk, [2.5, 97.5])
        flag_l = "밖" if not (lo_l <= real_l <= hi_l) else "안"
        flag_k = "밖" if not (lo_k <= real_k <= hi_k) else "안"
        print(f"== {alias(m)}  ({a.kind}{'' if a.kind == 'conf' else f' 층 {a.layer}'})")
        print(f"   LOPO   실제 {real_l:.3f}   섞음 평균 {np.mean(pl):.3f} 95%구간 [{lo_l:.2f}, {hi_l:.2f}]  → {flag_l}")
        print(f"   5-fold 실제 {real_k:.3f}   섞음 평균 {np.mean(pk):.3f} 95%구간 [{lo_k:.2f}, {hi_k:.2f}]  → {flag_k}")
    print("\n'안' = 라벨을 섞어도 나올 수 있는 값 = 신호 없음. n=33에서 구간 폭이 ±0.3에 이르는 것 자체가 핵심 사실.")


if __name__ == "__main__":
    main()
