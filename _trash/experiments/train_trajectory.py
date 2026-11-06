"""경로 기하 특징 11개 → 로지스틱 회귀. 특징별 단독 성능도 같이 찍는다.

입력: results/traj_<모델>_L<층>.npz  (extract_trajectory.py)
출력: ../solutions/trajectory/artifacts/<모델>.npz
    layer, max_new_tokens   런타임이 같은 조건으로 생성하도록
    mu, sd, w, b            표준화 + 로지스틱 계수
    cv_bacc

특징이 11개뿐이라 137샘플로도 과적합 걱정이 적다. 대신 "어느 특징이 갈리나"가 발견이다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "solutions" / "trajectory"))
from traj_lib import FEATURES  # noqa: E402

ART = HERE.parent / "solutions" / "trajectory" / "artifacts"


def cv(X, y, C=1.0, seeds=(0, 1, 2)):
    scores = []
    for seed in seeds:
        pred = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
            clf.fit((X[tr] - mu) / sd, y[tr])
            pred[te] = clf.predict((X[te] - mu) / sd)
        scores.append(balanced_accuracy_score(y, pred))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X, y = d["X"].astype(np.float32), d["y"].astype(int)
    print(f"{d['model']}  layer {int(d['layer'])}  max_new_tokens {int(d['max_new_tokens'])}")
    print(f"샘플 {len(y)} · robust {y.sum()}/{len(y)}\n")

    print("특징별 단독 AUC (0.5=무정보, 방향 무관하게 |AUC-0.5|로 보라)")
    for i, name in enumerate(FEATURES):
        auc = roc_auc_score(y, X[:, i])
        print(f"  {name:16s} AUC {auc:.3f}  {'#' * int(abs(auc - 0.5) * 40)}")

    m, s = cv(X, y, C=args.C)
    print(f"\n전체 11특징 cv bacc {m:.3f} ± {s:.3f}   (상수 예측 0.500)")

    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000).fit((X - mu) / sd, y)
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"{str(d['model']).split('/')[-1].replace('.', '_')}.npz"
    np.savez(out, layer=int(d["layer"]), max_new_tokens=int(d["max_new_tokens"]), mu=mu, sd=sd,
             w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]), cv_bacc=m,
             model_id=str(d["model"]))
    print(f"저장 {out}")
    print("계수 (표준화 기준, 양수=robust 쪽):")
    for name, w in sorted(zip(FEATURES, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:16s} {w:+.3f}")


if __name__ == "__main__":
    main()
