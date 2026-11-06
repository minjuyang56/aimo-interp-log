"""train-main-v2 36문제 벡터로 모델별 프로브를 leave-one-problem-out으로 평가한다.

라벨은 문제 단위(9/22 확인)라 4모델이 같은 y를 공유한다. 문제 하나를 통째로 빼고
나머지로 선을 그어 그 문제를 맞히기를 33번 반복 — 문제가 학습·평가에 동시에 들어가는 일이 없다.

찍는 것:
  [0] 길이 단독 AUC (지름길 검사 — MATH에선 0.75였고 여기선 0.39여야 정상)
  [1] 모델 × 층: LOPO balanced accuracy · AUC(margin)
  [2] 층 선택 편향을 피한 숫자: 층을 미리 고정(중간 층 · 3/4 지점)했을 때의 값
  [3] 모델 간 일치: 같은 문제에 대한 4모델의 margin 상관 — "한 방법이 여러 모델에 일반화되는가"

아티팩트 저장은 하지 않는다 (0.5를 넘는지 먼저 본다). 넘으면 train_probe.py 방식으로 만든다.

사용:  .venv/Scripts/python.exe experiments/train_train_v2.py [--C 0.05]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_train_v2 import LOCAL_MODELS  # noqa: E402
from extract_hidden import alias  # noqa: E402


def lopo_margins(X, y, C):
    """leave-one-problem-out. 각 문제의 out-of-fold margin을 돌려준다."""
    m = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
        m[i] = clf.decision_function(((X[i] - mu) / sd)[None])[0]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.05)
    a = ap.parse_args()

    per_model = {}
    for mname in LOCAL_MODELS:
        f = HERE / "results" / f"trainv2_hidden_{alias(mname)}.npz"
        if not f.is_file():
            print(f"(없음) {f.name}"); continue
        d = np.load(f, allow_pickle=True)
        per_model[mname] = d
    if not per_model:
        print("추출 결과 없음"); return

    first = next(iter(per_model.values()))
    y_all, problems = first["y"].astype(int), first["problems"]
    lab = y_all >= 0
    y = y_all[lab]
    nw = np.array([len(re.findall(r"[A-Za-z]+", p)) for p in problems])[lab]
    lognw = np.log1p(nw)
    len_m = lopo_margins(lognw[:, None], y, 1.0)
    print(f"문제 {len(y_all)} · 라벨 {lab.sum()} (robust {y.sum()} / spurious {len(y) - y.sum()})")
    print(f"[0] 길이 단독 AUC {roc_auc_score(y, lognw):.3f}  (0.5에서 먼 만큼 지름길. 방향 무관)"
          f"   길이만으로 LOPO bacc {balanced_accuracy_score(y, (len_m > 0).astype(int)):.3f}   상수 0.500")
    print("    → 아래 각 층에 '잔차 AUC' = 프로브 margin에서 길이로 설명되는 부분을 뺀 뒤의 AUC. 이게 진짜 숫자.\n")

    margins = {}
    for mname, d in per_model.items():
        H = d["H"].astype(np.float32)[lab]
        L = H.shape[1]
        print(f"== {mname}  층 {L} · 차원 {H.shape[2]}")
        print(f"   {'층':>3} {'bacc':>6} {'AUC':>6} {'ρ(len)':>7} {'잔차AUC':>7}")
        best = (0, 0, None); rows = {}
        for layer in range(0, L, 2):
            mg = lopo_margins(H[:, layer], y, a.C)
            bacc = balanced_accuracy_score(y, (mg > 0).astype(int)); auc = roc_auc_score(y, mg)
            rho = spearmanr(mg, lognw).correlation if mg.std() > 0 else 0.0
            resid = mg - np.polyval(np.polyfit(lognw, mg, 1), lognw) if mg.std() > 0 else mg
            rauc = roc_auc_score(y, resid) if resid.std() > 0 else 0.5
            rows[layer] = (bacc, auc, mg, rauc)
            print(f"   {layer:3d} {bacc:6.3f} {auc:6.3f} {rho:+7.2f} {rauc:7.3f} {'#' * int(max(0, rauc - 0.5) * 40)}")
            if rauc > best[1]: best = (bacc, rauc, layer)
        mid, q3 = (L // 2) // 2 * 2, (3 * L // 4) // 2 * 2
        print(f"   잔차 최고 층 {best[2]}: 잔차 AUC {best[1]:.3f}   (선택 편향 있음)")
        print(f"   고정 층 {mid}: AUC {rows[mid][1]:.3f} 잔차 {rows[mid][3]:.3f} · 고정 층 {q3}: AUC {rows[q3][1]:.3f} 잔차 {rows[q3][3]:.3f}\n")
        margins[mname] = rows[q3][2]

    if len(margins) >= 2:
        print("[3] 모델 간 margin 상관 (고정 3/4 층, 같은 33문제)")
        names = list(margins)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = spearmanr(margins[names[i]], margins[names[j]]).correlation
                print(f"   {alias(names[i]):28s} vs {alias(names[j]):28s} ρ = {r:+.2f}")
        stack = np.mean([margins[n] for n in names], axis=0)
        print(f"   4모델 margin 평균 → AUC {roc_auc_score(y, stack):.3f}  bacc {balanced_accuracy_score(y, (stack > 0).astype(int)):.3f}")


if __name__ == "__main__":
    main()
