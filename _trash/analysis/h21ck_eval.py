"""H21cK 평가 — K별 AUC·null·길이잔차·부트스트랩 CI, K 안정화 곡선, 규칙 개선 전/후. GPU 0. (지휘 #69)

사전등록 (측정 전 기록, CLAUDE.md "H21cK")
  주지표 = 변형 K개의 **drop 평균** · 보조 = 중앙값 · 부호: **클수록 robust**(뒤집지 않는다).
  규칙 개선(지수·첨자 제외, v2)은 **별개 변수**로 K=8 에서 전(v1)/후(v2)를 따로 본다.
  검문소: 문제 단위 permutation null 1000 · log(단어수) 잔차 · 부트스트랩 95% CI(문제 재추출 1000).

K 안정화 곡선을 보는 이유: h21c(K=1)의 리더보드 0.43 은 신호가 약한 것과 **변형 하나에 의존해 분산이 큰 것**을
구분할 수 없다. K 를 늘릴 때 AUC 가 올라가다 평평해지면 후자이고, 평평하지 않으면 K 를 더 늘려도 소용없다.

사용:  .venv/Scripts/python.exe analysis/h21ck_eval.py [--perms 1000] [--boot 1000]
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "experiments" / "results" / "h21ck"
KS = (1, 2, 4, 8, 16)


def load_runs():
    out = {}
    for f in sorted(glob.glob(str(H / "*.json"))):
        out[Path(f).stem] = json.loads(Path(f).read_text(encoding="utf-8"))
    return out


def take(rec, rule, k):
    """그 문제의 rule 변형 중 **앞쪽 k개** drop. 저장 순서가 생성 순서다(결정적)."""
    vals = [v for lab, v in rec["drops"].items() if lab.startswith(rule + ":")]
    return vals[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    runs = load_runs()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    if not runs:
        out(f"결과 없음: {H}/*.json")
        return
    alias0 = sorted(runs)[0]
    probs = [r["problem"] for r in runs[alias0]]
    y = np.array([r["y"] for r in runs[alias0]])
    lw = np.log(np.array([max(1, len(re.findall(r"[A-Za-z]+", p))) for p in probs], float))
    rng = np.random.default_rng(0)
    PY = np.array([rng.permutation(y) for _ in range(a.perms)], float)
    boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(a.boot)]

    def ranks(x):
        x = np.asarray(x, float)
        o = x.argsort(kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(1, len(x) + 1, dtype=float)
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
        return r

    def auc_of(score, yy=None):
        yy = y if yy is None else yy
        r = ranks(score)
        n1, n0 = float(yy.sum()), float((1 - yy).sum())
        if n1 == 0 or n0 == 0:
            return np.nan
        return (float(r[yy == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)

    def band(score):
        r = ranks(score)
        n1, n0 = float(y.sum()), float((1 - y).sum())
        auc = (float(r[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
        nl = (PY @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

    def boot_ci(score):
        s = np.asarray(score, float)
        vals = [auc_of(s[i], y[i]) for i in boot_idx]
        vals = [v for v in vals if not np.isnan(v)]
        return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, np.asarray(x, float), rcond=None)
        return np.asarray(x, float) - X @ b

    def series(rec_list, rule, k, agg):
        """문제별 점수. 변형이 없으면 nan → 그 문제는 중앙값으로 메운다(순위에 영향 최소)."""
        v = []
        for r in rec_list:
            d = take(r, rule, k)
            v.append((np.mean(d) if agg == "mean" else np.median(d)) if d else np.nan)
        v = np.array(v, float)
        if np.isnan(v).any():
            v[np.isnan(v)] = np.nanmedian(v) if not np.isnan(v).all() else 0.0
        return v

    out(f"H21cK · 모델 {len(runs)}종 · 문제 {len(y)} (robust {int(y.sum())}/spurious {int(len(y) - y.sum())}) "
        f"· null {a.perms} · 부트스트랩 {a.boot} · 주지표 = drop 평균(큼 = robust)")
    cover = {al: np.mean([len(take(r, 'v2', 16)) for r in rec]) for al, rec in runs.items()}
    out("변형 생성 수(v2, 문제 평균): " + " · ".join(f"{al.split('-')[0]} {c:.1f}" for al, c in cover.items()))
    out()
    out("A) K 안정화 곡선 — 개선 규칙(v2) · 평균 drop")
    out(f"{'모델':26s} " + "".join(f"{'K=' + str(k):>22s}" for k in KS))
    curve = {}
    for al, rec in sorted(runs.items()):
        cells = []
        for k in KS:
            sc = series(rec, "v2", k, "mean")
            auc, lo, hi, flag = band(sc)
            blo, bhi = boot_ci(sc)
            curve[(al, k)] = auc
            cells.append(f"{auc:.3f}{flag}[{blo:.2f},{bhi:.2f}]".rjust(22))
        out(f"{al:26s} " + "".join(cells))
    out("   표기: AUC▲[부트스트랩 95% CI] · ▲ = 문제단위 permutation null 95% 상단 초과")
    out()
    out("B) K=8 · 지표·잔차·null 상세 (v2)")
    out(f"{'모델':26s} {'지표':10s} {'AUC':>6s} {'null95%':>14s} {'부트 95%CI':>16s} {'길이잔차':>8s}")
    for al, rec in sorted(runs.items()):
        for agg in ("mean", "median"):
            sc = series(rec, "v2", 8, agg)
            auc, lo, hi, flag = band(sc)
            blo, bhi = boot_ci(sc)
            ra, *_ = band(resid(sc))
            star = "*" if agg == "mean" else " "
            out(f"{al:26s} {agg:10s}{star}{auc:6.3f} [{lo:.2f},{hi:.2f}]{flag} "
                f"[{blo:5.2f},{bhi:5.2f}] {ra:8.3f}")
    out("   * = 사전등록 주지표")
    out()
    out("C) 규칙 개선 전/후 (K=8, 평균 drop) — v1 = 지수·첨자 포함(h21c 규칙) · v2 = 제외")
    out(f"{'모델':26s} {'v1 AUC':>14s} {'v2 AUC':>14s} {'차이':>8s}")
    for al, rec in sorted(runs.items()):
        a1 = band(series(rec, "v1", 8, "mean"))
        a2 = band(series(rec, "v2", 8, "mean"))
        out(f"{al:26s} {a1[0]:8.3f}{a1[3]:5s} {a2[0]:8.3f}{a2[3]:5s} {a2[0] - a1[0]:+8.3f}")
    out()
    out("D) 모델 간 순위상관 (v2, K=8, 평균 drop) — 낮으면 모델별 성분에 가깝다")
    als = sorted(runs)
    S = {al: ranks(series(runs[al], "v2", 8, "mean")) for al in als}
    out("   " + " " * 26 + "".join(f"{b[:12]:>14s}" for b in als))
    for ai in als:
        out(f"   {ai:26s}" + "".join(f"{np.corrcoef(S[ai], S[aj])[0, 1]:14.3f}" for aj in als))
    out()
    best = max(curve.items(), key=lambda kv: kv[1])
    mean_by_k = {k: np.mean([curve[(al, k)] for al in als]) for k in KS}
    out("E) 제출 설정 선택 (train 기준, 리더보드 미사용)")
    out("   K별 모델평균 AUC: " + " · ".join(f"K={k} {v:.3f}" for k, v in mean_by_k.items()))
    kbest = max(mean_by_k, key=mean_by_k.get)
    out(f"   → 모델평균이 가장 높은 K = **{kbest}** (단일 최고는 {best[0][0]} K={best[0][1]} {best[1]:.3f})")
    (ROOT / "experiments" / "results" / "h21ck_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
