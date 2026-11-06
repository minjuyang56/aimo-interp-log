"""H20 설정 격자 — k(샘플 수) × cap(상한) 별 AUC 와 임계값. GPU 0. (지휘 #57a)

H14 커널이 저장한 **원본 샘플의 토큰 수**만 쓴다. 더 싼 설정을 사후에 흉내 낼 수 있다:
  cap c 에서의 토큰 수 = min(t, c),  완주 = (t < c)
  k=1 은 첫 샘플만, k=2 는 두 샘플 평균.
지표(사전등록 부호): H20a 평균 토큰 — 길수록 spurious / H20b 완주율 — 낮을수록 spurious.
임계값은 설정마다 **두 클래스 중앙값의 중점**으로 고정해 아티팩트에 담는다(제출물이 설정을 바꿔도 보정이 맞도록).

사용:  .venv/Scripts/python.exe analysis/h20_grid.py [--perms 1000] [--write-art]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CAPS = (4096, 6144, 8192, 12288, 16384)
KS = (1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--write-art", action="store_true")
    a = ap.parse_args()
    hits = sorted(glob.glob(str(ROOT / "experiments/results/h14/**/*_h14.json"), recursive=True))
    rec = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    rows = [{"y": r["y"], "toks": r["by_kind"]["orig"]["tokens"], "words": len(r["problem"].split())}
            for r in rec if r["by_kind"]["orig"]["tokens"]]
    y = np.array([r["y"] for r in rows])
    lw = np.log(np.array([max(1, r["words"]) for r in rows], float))
    rng = np.random.default_rng(0)
    perm_ys = [rng.permutation(y) for _ in range(a.perms)]
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    def band(sc):
        auc = roc_auc_score(y, sc)
        nl = [roc_auc_score(p, sc) for p in perm_ys]
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

    out(f"H20 설정 격자 (H14 저장 데이터, GPU 0) · 문제 {len(y)} (robust {int(y.sum())} / spurious {int(len(y) - y.sum())}) · null {a.perms}회")
    out(f"{'k':>2s} {'cap':>6s} {'지표':6s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'임계값':>9s} {'train정확도':>10s} {'최악 토큰비용':>12s}")
    art = {"thr": {}, "fit": {}}
    for k in KS:
        for cap in CAPS:
            tk = np.array([np.mean([min(t, cap) for t in r["toks"][:k]]) for r in rows], float)
            fin = np.array([np.mean([t < cap for t in r["toks"][:k]]) for r in rows], float)
            for tag, x, sign in (("H20a", tk, -1), ("H20b", fin, +1)):
                auc, lo, hi, f1 = band(sign * x)
                ra = roc_auc_score(y, sign * resid(x))
                thr = float((np.median(x[y == 1]) + np.median(x[y == 0])) / 2)
                pred = (x <= thr) if sign < 0 else (x >= thr)
                acc = float((pred == (y == 1)).mean())
                key = f"{tag}_k{k}_c{cap}"
                art["thr"][key] = thr
                art["fit"][key] = {"acc": acc, "auc": float(auc), "pred_true_rate": float(pred.mean())}
                out(f"{k:2d} {cap:6d} {tag:6s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} {thr:9.1f} {acc:10.3f} "
                    f"{k * cap:12d}")
    out()
    out("최악 토큰비용 = 문제당 k×cap (전부 상한까지 갔을 때). 실제로는 완주하는 문제가 있어 더 적다.")
    out(f"실측 평균(H14, k=2, 상한 16384): 문제당 {np.mean([sum(r['toks']) for r in rows]):.0f} 토큰 "
        f"(= k 합산). k=1 이면 {np.mean([r['toks'][0] for r in rows]):.0f} 토큰.")
    if a.write_art:
        f = ROOT / "solutions" / "h20" / "artifacts" / "h20_grid.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(art, indent=2), encoding="utf-8")
        out(f"아티팩트 저장 {f}")
    (ROOT / "experiments" / "results" / "h20_grid_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
