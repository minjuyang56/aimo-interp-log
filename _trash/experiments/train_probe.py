"""hidden state → 로지스틱 프로브. 층을 훑어 제일 잘 갈리는 층을 고르고 가중치를 저장한다.

입력: results/hidden_<모델>.npz  (extract_hidden.py 산출물)
출력: ../solutions/probe/artifacts/<모델>.npz
    layer   int      고른 층
    mu, sd  (D,)     표준화 파라미터 (학습셋 기준)
    w, b             로지스틱 회귀 계수. score = w·((h-mu)/sd) + b ; score > 0 → robust
    cv_bacc float    그 층의 교차검증 balanced accuracy

왜 balanced accuracy: 라벨이 36/101로 기울어서 정확도는 상수 예측이 0.74를 먹는다.
왜 class_weight="balanced": 같은 이유. True가 적고 비싸게 검증된 라벨이라 가중치를 준다.
왜 층을 훑나: 주최측 베이스라인이 그렇게 했고(31층), 어느 층에 정보가 있는지가 곧 발견이다.

사용:  uv run python experiments/train_probe.py [--npz results/hidden_DeepSeek-R1-0528-Qwen3-8B.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "solutions" / "probe" / "artifacts"


def standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def cv_layer(X, y, seeds=(0, 1, 2), folds=5, C=0.05):
    """한 층의 교차검증 balanced accuracy. 씨앗 3개 평균."""
    scores = []
    for seed in seeds:
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        pred = np.zeros_like(y)
        for tr, te in skf.split(X, y):
            Xtr, mu, sd = standardize(X[tr])
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
            clf.fit(Xtr, y[tr])
            pred[te] = clf.predict((X[te] - mu) / sd)
        scores.append(balanced_accuracy_score(y, pred))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=HERE / "results" / "hidden_DeepSeek-R1-0528-Qwen3-8B.npz")
    parser.add_argument("--C", type=float, default=0.05, help="L2 역강도. 137샘플×4096차원이라 세게 죈다")
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    H, y, model_id = d["H"].astype(np.float32), d["y"].astype(int), str(d["model"])
    n, L, D = H.shape
    print(f"{model_id}\n샘플 {n} · 층 {L} · 차원 {D} · robust {y.sum()}/{n}"
          f" · 상수 예측 bacc = 0.500 · 정확도 = {max(y.mean(), 1 - y.mean()):.3f}\n")

    print(f"{'층':>3}  {'cv bacc':>8}  {'±':>5}")
    results = []
    for layer in range(L):
        m, s = cv_layer(H[:, layer], y, C=args.C)
        results.append((m, s, layer))
        bar = "#" * int((m - 0.5) * 60) if m > 0.5 else ""
        print(f"{layer:3d}  {m:8.3f}  {s:5.3f}  {bar}", flush=True)

    best_m, best_s, best_layer = max(results)
    print(f"\n최고: 층 {best_layer}  bacc {best_m:.3f} ± {best_s:.3f}")
    if best_m < 0.55:
        print("경고: 어느 층도 상수 예측(0.5)을 뚜렷이 못 넘는다. 신호가 약하다.")

    X, mu, sd = standardize(H[:, best_layer])
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000).fit(X, y)
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"{args.npz.stem.replace('hidden_', '')}.npz"
    np.savez(out, layer=best_layer, mu=mu, sd=sd, w=clf.coef_[0].astype(np.float32),
             b=np.float32(clf.intercept_[0]), cv_bacc=best_m, model_id=model_id)
    insample = balanced_accuracy_score(y, clf.predict(X))
    print(f"저장 {out}  (인샘플 bacc {insample:.3f} — 참고용, cv가 진짜 숫자)")


if __name__ == "__main__":
    main()
