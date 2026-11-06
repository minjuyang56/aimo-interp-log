"""H15 — rename 대상 개수·밀도가 클수록 spurious. 텍스트만, GPU 0. 사전등록 2026-10-01 (지휘 #48).

근거: p13b 판별 실험에서 상한을 16384로 올려도 진짜로 무너진 4건이 **전부 rename** 이었다.
      그러면 "rename 이 건드릴 표면이 많은 문제일수록 깨지기 쉽다"가 텍스트만으로 예측될 수 있다.

지표 (부호 +: 많을수록 spurious → robust 점수 = −지표)
  n_sym   edit_rename 이 실제로 바꾸는 **문자 수**            ← **주지표**
  n_var   수식 안 한 글자 변수의 **종류 수**                  (탐색)
  n_prop  수식 밖 대문자 고유명사(N_topic) 수                 (탐색)
  d_*     위 셋을 문제 단어 수로 나눈 밀도                    (탐색)
검문소: 공식 train-main-v2 문제 단위 라벨 · 문제 단위 permutation null 1000 · 문제 길이(log 단어수) 잔차화 병기.

사용:  .venv/Scripts/python.exe analysis/h15_rename_density.py [--perms 1000]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from h9_essence import units  # noqa: E402
from variants_official import MATH_SPAN, edit_rename  # noqa: E402

KEYS = [("n_sym", "주지표"), ("n_var", "탐색"), ("n_prop", "탐색"),
        ("d_sym", "탐색"), ("d_var", "탐색"), ("d_prop", "탐색")]


def official_problems():
    import pandas as pd
    df = pd.read_parquet(ROOT / "experiments" / "results" / "train-main-v2.parquet")
    rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna()
        if len(lab):
            rows.append({"pidx": i, "problem": p.rstrip(), "y": int(bool(lab.astype(bool).mean() >= 0.5))})
    return rows


def features(text):
    rn = edit_rename(text)
    n_sym = sum(1 for a, b in zip(text, rn) if a != b) if rn and len(rn) == len(text) else 0
    invars = set()
    for m in MATH_SPAN.finditer(text):
        for v in re.finditer(r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])", m.group(0)):
            invars.add(v.group(1))
    n_prop = sum(1 for s, e, b in units(text) if b == "N_topic" and text[s].isupper())
    w = max(1, len(text.split()))
    return {"n_sym": n_sym, "n_var": len(invars), "n_prop": n_prop,
            "d_sym": n_sym / w, "d_var": len(invars) / w, "d_prop": n_prop / w, "words": w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    probs = official_problems()
    F = [dict(features(p["problem"]), y=p["y"], pidx=p["pidx"]) for p in probs]
    y = np.array([f["y"] for f in F])
    lw = np.log(np.array([f["words"] for f in F], float))
    rng = np.random.default_rng(0)
    perm_ys = [rng.permutation(y) for _ in range(a.perms)]
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    out(f"H15 rename 대상 개수·밀도 (텍스트만) · 공식 문제 {len(y)}개 robust {int(y.sum())} / spurious {int(len(y) - y.sum())} · "
        f"null {a.perms}회 · 부호 +(많을수록 spurious) · 통제 log(단어수)")
    out(f"{'지표':8s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'잔차null':>13s} {'robust평균':>11s} {'spurious평균':>12s}  비고")
    for key, note in KEYS:
        x = np.array([f[key] for f in F], float)
        sc = -x                                        # 많을수록 spurious → robust 점수는 음수
        auc = roc_auc_score(y, sc)
        nl = [roc_auc_score(p, sc) for p in perm_ys]
        lo, hi = np.percentile(nl, [2.5, 97.5])
        rs = -resid(x)
        ra = roc_auc_score(y, rs)
        nl2 = [roc_auc_score(p, rs) for p in perm_ys]
        lo2, hi2 = np.percentile(nl2, [2.5, 97.5])
        f1 = "▲" if auc > hi else ("▼" if auc < lo else " ")
        f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
        out(f"{key:8s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2} "
            f"{x[y == 1].mean():11.3f} {x[y == 0].mean():12.3f}  {note}")
    out()
    out(f"참고: 단어 수 중앙 {int(np.median(np.exp(lw)))} · n_sym 중앙 {int(np.median([f['n_sym'] for f in F]))} · "
        f"rename 이 아무것도 못 바꾼 문제 {sum(1 for f in F if f['n_sym'] == 0)}")
    (ROOT / "experiments" / "results" / "h15_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
