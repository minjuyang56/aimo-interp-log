"""H14 평가 — 사전등록대로(지휘 #43). 부호 고정 +(길이비 큼 = spurious), 문제 단위 permutation null 1000,
원본 생성길이 잔차화 병기. 보조: rename 정확도 하락(max_drop)의 공식 라벨 AUC, 미완주율.

주의: H14 커널은 정답 배열(answers)을 저장했으므로 rename max_drop 은 여기서 계산한다.
      gold 가 없으므로 '원본 다수답'을 의사정답으로 쓴다 — 공식 라벨과 독립적이고, 원본이 서로 다른 답을 내면 제외한다.

사용:  .venv/Scripts/python.exe analysis/h14_eval.py [--perms 1000]
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--glob", default="experiments/results/h14/**/*_h14.json")
    a = ap.parse_args()
    hits = sorted(glob.glob(a.glob, recursive=True))
    if not hits:
        print("결과 없음"); return
    rec = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    rows = []
    for r in rec:
        bk = r["by_kind"]
        if r.get("length_ratio") is None:
            continue
        ans = [x for x in bk["orig"]["answers"] if x is not None]
        gold = Counter(ans).most_common(1)[0][0] if ans else None
        base_acc = (sum(1 for x in bk["orig"]["answers"] if x == gold) / len(bk["orig"]["answers"])) if gold is not None else None
        vaccs = [sum(1 for x in bk[kd]["answers"] if x == gold) / len(bk[kd]["answers"])
                 for kd in ("rename", "rename2") if not bk[kd]["degenerate"]] if gold is not None else []
        k = len(bk["orig"]["answers"])
        unfin = 1 - (bk["orig"]["n_fin"] + sum(bk[kd]["n_fin"] for kd in ("rename", "rename2"))) / (3 * k)
        rows.append({"y": r["y"], "lr": r["length_ratio"], "base_tok": bk["orig"]["mean_tokens"],
                     "plen": len(r["problem"]), "gold": gold, "base_acc": base_acc,
                     "max_drop": (base_acc - min(vaccs)) if (base_acc is not None and vaccs) else None,
                     "unfin": unfin})
    y = np.array([r["y"] for r in rows])
    bt = np.log(np.array([max(1.0, r["base_tok"]) for r in rows]))
    pl = np.log(np.array([max(1, r["plen"]) for r in rows], float))
    rng = np.random.default_rng(0)
    perm_ys = [rng.permutation(y) for _ in range(a.perms)]

    def resid(x):
        X = np.column_stack([np.ones(len(x)), bt, pl])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    def band(sc, yy=None, ps=None):
        yy = y if yy is None else yy
        ps = perm_ys if ps is None else ps
        auc = roc_auc_score(yy, sc)
        nl = [roc_auc_score(p, sc) for p in ps]
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

    out(f"H14 (Skywork · 공식 문제 단위 라벨) · n={len(y)} robust {int(y.sum())} / spurious {int(len(y) - y.sum())} · "
        f"rename 불가 제외 {len(rec) - len(rows)} · null {a.perms}회 · 통제: log(원본 생성토큰) + log(문제 길이)")
    out(f"{'지표':18s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'잔차null':>13s} {'robust평균':>11s} {'spurious평균':>12s}  비고")
    # 주지표: 길이비 큼 = spurious -> robust 점수 = −길이비
    lr = np.array([r["lr"] for r in rows])
    auc, lo, hi, f1 = band(-lr)
    rs = -resid(lr)
    ra, lo2, hi2, f2 = band(rs)
    out(f"{'길이비(부호+)':18s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2} "
        f"{lr[y == 1].mean():11.3f} {lr[y == 0].mean():12.3f}  주지표")
    # 보조: rename max_drop (큼 = spurious -> robust 점수 = −max_drop)
    ok = [i for i, r in enumerate(rows) if r["max_drop"] is not None]
    if len(ok) >= 10 and len(set(y[ok])) == 2:
        md = np.array([rows[i]["max_drop"] for i in ok])
        yy = y[ok]
        ps = [rng.permutation(yy) for _ in range(a.perms)]
        auc, lo, hi, f1 = band(-md, yy, ps)
        out(f"{'rename max_drop':18s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {'':>8s} {'':>13s} "
            f"{md[yy == 1].mean():11.3f} {md[yy == 0].mean():12.3f}  보조 (n={len(ok)}, 의사정답)")
    uf = np.array([r["unfin"] for r in rows])
    auc, lo, hi, f1 = band(-uf)
    out(f"{'미완주율':18s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {'':>8s} {'':>13s} "
        f"{uf[y == 1].mean():11.3f} {uf[y == 0].mean():12.3f}  보조")
    out()
    out(f"참고: 원본 평균 생성 토큰 {np.exp(bt).mean():.0f} · 전체 미완주율 {uf.mean():.2f} · "
        f"의사정답 확보 {len(ok)}/{len(rows)}")
    Path("experiments/results/h14_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
