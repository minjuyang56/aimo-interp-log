"""민감도 프로파일 결과를 길이 통제 아래 다시 읽는다.

sensitivity.py의 층별 AUC 표는 raw 값이다. 프로브가 길이를 배웠던 전례가 있으니
편집별 표현변화도 같은 검사를 거친다:

  ρ(변화, 길이)     편집 자체가 길이의 함수인가
  잔차 AUC          길이로 설명되는 부분을 빼고 남은 신호
  ≥15단어 AUC       길이 지름길이 약한 부분집합에서의 신호

사용:  uv run python analysis/sensitivity_confound.py [--layers 3 10 17 25 31]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import AGG_CSV, require  # noqa: E402

NPZ = Path(__file__).resolve().parent.parent / "experiments" / "results" / "sensitivity_DeepSeek-R1-0528-Qwen3-8B.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 10, 17, 25, 31])
    args = parser.parse_args()
    require(NPZ, AGG_CSV)

    d = np.load(NPZ, allow_pickle=True)
    D, y, pid, edits = d["D"], d["y"].astype(int), list(d["pid"]), list(d["edits"])
    text = {r["problem_id"]: r["original_problem"] for r in csv.DictReader(open(AGG_CSV, newline="", encoding="utf-8"))}
    nw = np.array([len(re.findall(r"[A-Za-z]+", text[p])) for p in pid], float)
    lognw = np.log1p(nw)
    long_mask = nw >= 15

    print(f"문제 {len(y)} · robust {y.sum()} · 길이 단독 AUC {roc_auc_score(y, lognw):.3f}\n")
    for e in edits:
        i = edits.index(e)
        zero = (D[:, i, :].max(1) < 1e-3).mean()
        print(f"[{e}]  변화가 0인 문제 {zero:.0%}")
        print(f"  {'층':>3} {'raw AUC':>8} {'ρ(변화,길이)':>12} {'잔차 AUC':>9} {'≥15단어 AUC':>12}")
        for L in args.layers:
            v = D[:, i, L]
            if v.std() < 1e-9:
                print(f"  {L:3d}   (변화 없음)"); continue
            resid = v - np.polyval(np.polyfit(lognw, v, 1), lognw)
            print(f"  {L:3d} {roc_auc_score(y, v):8.3f} {spearmanr(v, nw).correlation:+12.2f}"
                  f" {roc_auc_score(y, resid):9.3f} {roc_auc_score(y[long_mask], v[long_mask]):12.3f}")
        print()
    print("읽는 법: raw AUC가 0.5에서 멀어도 |ρ|가 크고 잔차 AUC가 0.5로 돌아오면 길이의 대리다.")
    print("        잔차와 ≥15단어가 같은 방향으로 0.5를 벗어나면 길이 너머의 신호다.")


if __name__ == "__main__":
    main()
