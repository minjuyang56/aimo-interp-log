"""H15 외부검증 — 파라미터 고정, 재학습 없음. (지휘 #49-1)

H15 특징(n_sym · d_sym · log단어수)을 **부호만 고정한 채** 우리 합성 라벨에 그대로 적용한다.
문제셋이 공식 train(올림피아드 52단어 중앙)과 다르므로(AIME·MATH level 3~5) 길이 교란 분리에 도움이 된다.

라벨 집합
  v3-Skywork  : genlab v2/v3 유효 라벨 29건 (robust 24 / 진짜 spurious 4 + 오답 우세 1) — 절단 아티팩트 제외
  v1-run      : genlab v1 Skywork level 3~4 공식 라벨 85건 (robust 83 / spurious 2) — spurious 2개라 사실상 판정 불가
  v1-l5       : v1 level 5 보충 17건 (robust 17 / spurious 0) — 한 클래스뿐이라 평가 불가
부호: n_sym·d_sym 는 **많을수록 spurious**(robust 점수 = −지표), log단어수는 **길수록 spurious**(robust 점수 = −log단어수).

사용:  .venv/Scripts/python.exe analysis/h15_external.py [--perms 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))
from h15_rename_density import features, official_problems  # noqa: E402

RES = ROOT / "experiments" / "results"


def v3_set():
    lp = sorted(RES.glob("genlab/**/Skywork-OR1-Math-7B_v3_labels.json"))
    pp = sorted(RES.glob("genlab/**/Skywork-OR1-Math-7B_probe13.json"))
    if not lp:
        return []
    lab = json.loads(lp[-1].read_text(encoding="utf-8"))
    verdict = {int(r["pidx"]): r["verdict"] for r in json.loads(pp[-1].read_text(encoding="utf-8"))} if pp else {}
    rows = []
    for s in lab:
        if s.get("label") is None:
            continue
        if s["label"] is False and verdict.get(int(s["pidx"])) == "truncation_artifact":
            continue
        rows.append({"problem": s["problem"], "y": int(bool(s["label"]))})
    return rows


def v1_set(pattern):
    import numpy as _np
    hits = sorted(RES.glob(pattern))
    if not hits:
        return []
    store = list(_np.load(hits[-1], allow_pickle=True)["store"])
    return [{"problem": s["problem"], "y": int(bool(s["label_official"]))}
            for s in store if s.get("label_official") is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    sets = [("공식 train (참조)", [{"problem": p["problem"], "y": p["y"]} for p in official_problems()]),
            ("v3-Skywork (AIME+L5)", v3_set()),
            ("v1-run (MATH L3~4)", v1_set("genlab/*run/genlab/Skywork-OR1-Math-7B.npz")),
            ("v1-l5 (MATH L5)", v1_set("genlab/*l5/genlab/Skywork-OR1-Math-7B-l5.npz"))]
    out(f"H15 외부검증 (부호 고정·재학습 없음) · null {a.perms}회")
    out(f"{'라벨 집합':24s} {'n':>4s} {'rob/spu':>8s} {'n_sym':>7s} {'null95%':>13s} {'d_sym':>7s} {'log단어수':>9s} {'단어수 중앙':>10s}")
    for name, rows in sets:
        if not rows:
            out(f"{name:24s}  (데이터 없음)"); continue
        y = np.array([r["y"] for r in rows])
        if len(y) < 8 or y.min() == y.max():
            out(f"{name:24s} {len(y):4d} {f'{int(y.sum())}/{int(len(y) - y.sum())}':>8s}  한 클래스뿐 — 평가 불가"); continue
        F = [features(r["problem"]) for r in rows]
        res = {}
        for key in ("n_sym", "d_sym"):
            x = np.array([f[key] for f in F], float)
            res[key] = roc_auc_score(y, -x)
        lw = np.log(np.array([f["words"] for f in F], float))
        res["lw"] = roc_auc_score(y, -lw)
        nl = [roc_auc_score(rng.permutation(y), -np.array([f["n_sym"] for f in F], float)) for _ in range(a.perms)]
        lo, hi = np.percentile(nl, [2.5, 97.5])
        flag = "▲" if res["n_sym"] > hi else ("▼" if res["n_sym"] < lo else " ")
        out(f"{name:24s} {len(y):4d} {f'{int(y.sum())}/{int(len(y) - y.sum())}':>8s} {res['n_sym']:7.3f} "
            f"[{lo:.2f},{hi:.2f}]{flag} {res['d_sym']:7.3f} {res['lw']:9.3f} {int(np.median(np.exp(lw))):10d}")
    out()
    out("해석 메모: 문제셋마다 단어 수 중앙이 다르다. n_sym 이 길이 대리라면 집합을 옮길 때 log단어수 AUC 와 같이 움직인다.")
    (RES / "h15_external_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
