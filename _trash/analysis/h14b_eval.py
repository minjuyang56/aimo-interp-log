"""H14b 평가 — 사전등록대로(지휘 #44). 부호 고정, 문제 단위 permutation null 1000, 원본 생성길이 잔차화 병기.

지표 (rename − 원본), 점수는 **spurious 를 양성으로 보지 않고 robust=1 기준 AUC** 로 낸다.
  사전등록 부호: rename 이 더 불확실할수록 spurious
    (a) d_logprob  변형에서 더 낮으면 spurious  -> robust 점수 = +d_logprob      (주지표)
    (b) d_entropy  변형에서 더 높으면 spurious  -> robust 점수 = −d_entropy
    (c) d_first_math 변형에서 더 늦으면 spurious -> robust 점수 = −d_first_math
통제: 원본 평균 생성 토큰 수(log), 원문제 문자 길이(log).

사용:  .venv/Scripts/python.exe analysis/h14b_eval.py [--perms 1000]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

KEYS = [("d_logprob", +1, "주지표"), ("d_entropy", -1, "탐색"), ("d_first_math", -1, "탐색")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--glob", default="experiments/results/h14b/**/*_h14b.json")
    a = ap.parse_args()
    hits = sorted(glob.glob(a.glob, recursive=True))
    if not hits:
        print("결과 없음"); return
    rec = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    ok = [r for r in rec if r.get("d_logprob") is not None]
    y = np.array([r["y"] for r in ok])
    base_tok = np.log(np.array([max(1, r["by_kind"]["orig"]["tokens"]) for r in ok], float))
    plen = np.log(np.array([max(1, len(r["problem"])) for r in ok], float))
    rng = np.random.default_rng(0)
    perm_ys = [rng.permutation(y) for _ in range(a.perms)]
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    def resid(x):
        X = np.column_stack([np.ones(len(x)), base_tok, plen])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    out(f"H14b (Skywork · 공식 문제 단위 라벨) · n={len(y)} robust {int(y.sum())} / spurious {int(len(y) - y.sum())} · "
        f"rename 불가로 제외 {len(rec) - len(ok)} · null {a.perms}회 · 통제: log(원본 생성토큰) + log(문제 길이)")
    out(f"{'지표':14s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'잔차null':>13s} {'robust평균':>11s} {'spurious평균':>12s}  비고")
    for key, sign, note in KEYS:
        x = np.array([r[key] for r in ok], float)
        sc = sign * x
        auc = roc_auc_score(y, sc)
        nl = [roc_auc_score(p, sc) for p in perm_ys]
        lo, hi = np.percentile(nl, [2.5, 97.5])
        rs = sign * resid(x)
        ra = roc_auc_score(y, rs)
        nl2 = [roc_auc_score(p, rs) for p in perm_ys]
        lo2, hi2 = np.percentile(nl2, [2.5, 97.5])
        f1 = "▲" if auc > hi else ("▼" if auc < lo else " ")
        f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
        out(f"{key:14s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2} "
            f"{x[y == 1].mean():11.4f} {x[y == 0].mean():12.4f}  {note}")
    Path("experiments/results/h14b_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
