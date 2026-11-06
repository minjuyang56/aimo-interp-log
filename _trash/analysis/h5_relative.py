"""H5 상호작용 — 상대 친숙도: (이 모델의 이 문제 NLL/MinK) − (같은 문제에 대한 나머지 3모델 평균).

왜: 주최측 검증셋은 문제별·모델별 균형이라 "문제만 보는" 특징(길이·공용계측기 프로브·풀링 NLL)은
구조적으로 50%다. 남는 건 모델×문제 상호작용뿐. 문제 난이도·길이 효과는 3모델 평균을 빼서 제거한다.

사전등록 부호: 상대적으로 더 익숙(상대 NLL 낮음, 상대 mink 높음) = spurious.
  robust 점수 = +rel_nll, −rel_mink20, −rel_minkpp20, +rel_nll_2nd, +rel_zlib

검증
  (1) v2에서 2개 이상 모델에 걸친 문제의 모델 간 라벨 차이 — H2에서 0건이면 그 사실을 보고 (설명할 분산 없음)
  (2) 모델별 '자기 행'(train-main-v2에서 model_id가 그 모델인 라벨 행): 상대 친숙도 vs 라벨, 섞기 null
  (3) 4모델 자기 행 전부 쌓기 (모델-문제 쌍 단위): 같은 검사

사용:  .venv/Scripts/python.exe analysis/h5_relative.py [--perms 30]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
from extract_hidden import alias  # noqa: E402
from extract_train_v2 import LOCAL_MODELS  # noqa: E402
from h5_memorization import FEATS, SIGN  # noqa: E402

RES = HERE.parent / "experiments" / "results"
SHORT = {"Qwen/Qwen3.5-4B": "Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B": "Skywork-7B",
         "allenai/Olmo-3-7B-Think": "Olmo-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-8B"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=30); a = ap.parse_args()
    rng = np.random.default_rng(0)
    data = {m: np.load(RES / f"h5_{alias(m)}.npz", allow_pickle=True) for m in LOCAL_MODELS}
    ref = data[LOCAL_MODELS[0]]
    for m, d in data.items():
        assert (d["problems"] == ref["problems"]).all(), f"행 정렬 불일치: {m}"
    Fs = {m: d["F"].astype(np.float32) for m, d in data.items()}
    rel = {m: Fs[m] - np.mean([Fs[o] for o in LOCAL_MODELS if o != m], axis=0) for m in LOCAL_MODELS}
    v2 = ref["src"] == "v2"
    problems = ref["problems"]; pidx = ref["pidx"]
    lines = []
    def out(s=""): print(s); lines.append(s)

    # (1) 모델 간 라벨 차이
    df = pd.read_parquet(RES / "train-main-v2.parquet")
    lab = df[df.is_robust.notna()].copy(); lab["y"] = lab.is_robust.astype(bool).astype(int)
    pm = lab.groupby(["problem", "model_id"]).y.mean().unstack()
    multi = pm[pm.notna().sum(1) >= 2]
    n_diff = int((multi.apply(lambda r: r.dropna().round().nunique() > 1, axis=1)).sum())
    out(f"(1) v2에서 2개 이상 모델에 걸친 문제 {len(multi)}개 중 모델 간 라벨이 다른 문제: {n_diff}개")
    if n_diff == 0:
        out("    → 상호작용이 설명할 분산이 v2에는 없다. 아래 (2)(3)은 '문제 라벨을 상대 친숙도로 맞히나'의 검사다.")

    # (2) 모델별 자기 행
    out(f"\n(2) 모델별 자기 행 (train-main-v2 라벨 행, 해당 model_id) · 사전등록 부호 · null {a.perms}회")
    out(f"  {'모델':12s} {'n':>3s} {'rob':>3s}  " + "  ".join(f"{f:>10s}" for f in FEATS))
    stack_y, stack_s = [], {f: [] for f in FEATS}
    text_to_row = {t: i for i, t in enumerate(problems)}
    for m in LOCAL_MODELS:
        own = lab[lab.model_id == m].groupby("problem").y.mean().round().astype(int)
        rows = [text_to_row[t] for t in own.index if t in text_to_row]
        y = own.values
        if len(y) < 4 or y.min() == y.max():
            out(f"  {SHORT[m]:12s} {len(y):3d} {int(y.sum()):3d}  (라벨 한쪽뿐 또는 n<4 — 생략)"); continue
        cells = []
        for j, f in enumerate(FEATS):
            sc = SIGN[f] * rel[m][rows, j]
            auc = roc_auc_score(y, sc)
            null = [roc_auc_score(rng.permutation(y), sc) for _ in range(a.perms)]
            lo, hi = np.percentile(null, [2.5, 97.5])
            cells.append(f"{auc:.2f}{'▲' if auc > hi else ('▼' if auc < lo else ' ')}[{lo:.2f},{hi:.2f}]")
            stack_s[f].extend(sc.tolist())
        stack_y.extend(y.tolist())
        out(f"  {SHORT[m]:12s} {len(y):3d} {int(y.sum()):3d}  " + "  ".join(cells))

    # (3) 쌓기
    y = np.array(stack_y)
    out(f"\n(3) 4모델 자기 행 전부 쌓기 (모델-문제 쌍 {len(y)}개, robust {y.sum()})")
    for f in FEATS:
        sc = np.array(stack_s[f])
        auc = roc_auc_score(y, sc); bacc = balanced_accuracy_score(y, (sc > np.median(sc)).astype(int))
        null = [roc_auc_score(rng.permutation(y), sc) for _ in range(a.perms)]
        lo, hi = np.percentile(null, [2.5, 97.5])
        out(f"  rel_{f:10s} AUC {auc:.3f}  bacc(중앙값) {bacc:.3f}  null[{lo:.2f},{hi:.2f}]{'▲' if auc > hi else ('▼' if auc < lo else ' ')}")
    out("\n판정: 상대 친숙도는 문제 효과를 뺀 것이라, 라벨이 문제 단위면 ~0.5가 나오는 게 정합적이다.")
    (RES / "h5_relative_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
