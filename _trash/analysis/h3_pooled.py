"""H3 — 문제 단위 풀링. DeepSeek-8B를 공용 계측기로 두고 v2 문제(다수결 라벨, None 제외)에서
프로브 층별 · 자신감 특징 · 길이의 leave-one-problem-out bacc/AUC를 라벨 섞기 null과 함께 한 표로.

train_train_v2 / train_confidence_v2 / permutation_check 를 한 모델에 대해 한 표로 합친 것.
사용:  .venv/Scripts/python.exe analysis/h3_pooled.py [--perms 30]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
from extract_hidden import alias  # noqa: E402
from train_train_v2 import lopo_margins  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
RES = HERE.parent / "experiments" / "results"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=30); a = ap.parse_args()
    dh = np.load(RES / f"trainv2_hidden_{alias(MODEL)}.npz", allow_pickle=True)
    dc = np.load(RES / f"trainv2_conf_{alias(MODEL)}_N512.npz", allow_pickle=True)
    lab = dh["y"] >= 0; y = dh["y"][lab].astype(int)
    H, F, feats = dh["H"].astype(np.float32)[lab], dc["F"].astype(np.float32)[lab], list(dc["features"])
    lognw = np.log1p([len(re.findall(r"[A-Za-z]+", p)) for p in dh["problems"]])[lab]
    rng = np.random.default_rng(0)
    print(f"계측기 {MODEL.split('/')[-1]} · v2 문제 {len(y)} (robust {y.sum()} / spurious {len(y) - y.sum()}) · LOPO")
    print(f"  {'특징':22s} {'bacc':>6s} {'AUC':>6s}  null AUC 95% ({a.perms}회)")

    def row(name, X, C, trained=True):
        m = lopo_margins(X, y, C) if trained else X
        bacc = balanced_accuracy_score(y, (m > 0).astype(int)) if trained else max(
            balanced_accuracy_score(y, (m > np.median(m)).astype(int)), balanced_accuracy_score(y, (m < np.median(m)).astype(int)))
        auc = roc_auc_score(y, m)
        null = []
        for _ in range(a.perms):
            yp = rng.permutation(y)
            null.append(roc_auc_score(yp, lopo_margins(X, yp, C) if trained else X))
        lo, hi = np.percentile(null, [2.5, 97.5]); flag = "▲" if auc > hi else ("▼" if auc < lo else " ")
        print(f"  {name:22s} {bacc:6.3f} {auc:6.3f}  [{lo:.2f}, {hi:.2f}] {flag}")

    row("길이 (학습 없음)", lognw, 1.0, trained=False)
    row("길이 (LOPO)", lognw[:, None], 1.0)
    for L in range(2, H.shape[1], 4):
        row(f"프로브 층{L}", H[:, L], 0.05)
    for i, f in enumerate(feats):
        if F[:, i].std() > 0:
            row(f"자신감 {f} (학습 없음)", F[:, i], 1.0, trained=False)
    row("자신감 12특징 (LOPO)", F, 1.0)
    print("\n판정: ▲ 없음 = 어떤 특징도 라벨 섞기와 구분 안 됨. bacc는 학습 없는 행에선 중앙값 분할(유리한 부호).")


if __name__ == "__main__":
    main()
