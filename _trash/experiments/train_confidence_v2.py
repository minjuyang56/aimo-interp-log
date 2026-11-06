"""생성 자신감 특징(12개)으로 문제 단위 robust 라벨을 leave-one-problem-out으로 맞혀본다.

train_train_v2.py와 같은 평가 틀. 다른 점은 입력이 4096차원 벡터가 아니라 특징 12개라
33문제로도 과적합 걱정이 적고, 특징별로 "무엇이 갈리는가"를 읽을 수 있다.

찍는 것:
  [0] 길이 지름길 (참고)
  [1] 모델별: 특징 단독 AUC · 12특징 LOPO bacc/AUC · 길이 잔차 AUC
  [2] 모델 간 margin 상관 · 4모델 평균

사용:  .venv/Scripts/python.exe experiments/train_confidence_v2.py [--n-tokens 512] [--C 1.0]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias  # noqa: E402
from extract_train_v2 import LOCAL_MODELS  # noqa: E402
from train_train_v2 import lopo_margins  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=512)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()

    data = {}
    for m in LOCAL_MODELS:
        f = HERE / "results" / f"trainv2_conf_{alias(m)}_N{a.n_tokens}.npz"
        if f.is_file():
            data[m] = np.load(f, allow_pickle=True)
    if not data:
        print("결과 없음"); return
    first = next(iter(data.values()))
    y_all, problems, feats = first["y"].astype(int), first["problems"], list(first["features"])
    lab = y_all >= 0; y = y_all[lab]
    lognw = np.log1p([len(re.findall(r"[A-Za-z]+", p)) for p in problems])[lab]
    print(f"문제 {len(y_all)} · 라벨 {lab.sum()} (robust {y.sum()} / spurious {len(y) - y.sum()}) · N={a.n_tokens}")
    print(f"[0] 길이 단독 AUC {roc_auc_score(y, lognw):.3f}   상수 bacc 0.500\n")

    margins = {}
    for m, d in data.items():
        F = d["F"].astype(np.float32)[lab]
        print(f"== {m}")
        print("   특징 단독 AUC (|AUC−0.5|가 클수록 갈림, 방향은 부호):")
        for i, name in enumerate(feats):
            if F[:, i].std() < 1e-9: continue
            auc = roc_auc_score(y, F[:, i]); rho = spearmanr(F[:, i], lognw).correlation
            print(f"     {name:20s} AUC {auc:.3f}  ρ(len) {rho:+.2f}  {'#' * int(abs(auc - 0.5) * 40)}")
        mg = lopo_margins(F, y, a.C)
        bacc, auc = balanced_accuracy_score(y, (mg > 0).astype(int)), roc_auc_score(y, mg)
        resid = mg - np.polyval(np.polyfit(lognw, mg, 1), lognw)
        print(f"   12특징 LOPO  bacc {bacc:.3f}  AUC {auc:.3f}  ρ(len) {spearmanr(mg, lognw).correlation:+.2f}  길이 잔차 AUC {roc_auc_score(y, resid):.3f}\n")
        margins[m] = mg

    if len(margins) >= 2:
        names = list(margins)
        print("[2] 모델 간 margin 상관")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"   {alias(names[i]):28s} vs {alias(names[j]):28s} ρ = {spearmanr(margins[names[i]], margins[names[j]]).correlation:+.2f}")
        stack = np.mean([margins[n] for n in names], axis=0)
        print(f"   4모델 평균 → AUC {roc_auc_score(y, stack):.3f}  bacc {balanced_accuracy_score(y, (stack > 0).astype(int)):.3f}")


if __name__ == "__main__":
    main()
