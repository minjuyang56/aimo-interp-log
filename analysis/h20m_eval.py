"""H20-M 평가 — 모델별 원본 생성길이가 robust/spurious 를 가르는가. GPU 0. (지휘 #63 사전등록대로)

사전등록 (2026-10-04, 커널 push 전 기록 · CLAUDE.md "H20-M 모델별 원본 생성길이")
  부호 고정: 원본 평균 생성 토큰 수가 **길수록 spurious** (점수 = −토큰수). 뒤집지 않는다.
  주지표: **cap 12288** · 보조: cap 8192 (둘 다 min(t, cap) 절단 시뮬로 같은 생성에서 낸다).
  검문소: 문제 단위 permutation null 1000 · log(단어수) 잔차 AUC 병기 · 모델 간 Spearman 순위상관.
  판정: 모델 간 상관 중앙값 **>=0.8** 이면 "모델별 생성은 같은 문제 순위를 재는 것" → h20e 의 모델별 근거 기각.
        **<0.5** 면 모델별 성분 실재 → h20e 지명 후보. 0.5~0.8 은 보류.

왜 이걸 보는가: h20e 는 "모델마다 길이가 달라 모델별 판정이 가능하다" 를 전제한다. 학습셋 라벨은 문제 단위라
모델 간 라벨 차이로는 검증할 수 없으므로, 같은 지표를 모델마다 재서 **순위가 얼마나 겹치는지**로 전제를 시험한다.

사용:  .venv/Scripts/python.exe analysis/h20m_eval.py [--perms 1000]
       (커널 출력이 experiments/results/h20m/<slug>/ 또는 h10b/<slug>/ 아래 *_h14.json 으로 있어야 한다)
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CAPS = (8192, 12288)
PRIMARY = 12288


def find_runs():
    """H20-M 커널 출력(slug 에 h20m 이 든 것)만 모은다. 모델 별칭 -> 레코드."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "experiments/results/**/*_h14.json"), recursive=True)):
        if "h20m" not in f.replace("\\", "/").lower():
            continue
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        alias = Path(f).name.replace("_h14.json", "")
        out[alias] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    runs = find_runs()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    if not runs:
        out("H20-M 커널 출력이 없다 (experiments/results/**/*h20m*/*_h14.json). 먼저 내려받을 것.")
        return

    # 문제 단위 라벨·텍스트는 모델마다 같은 33문제여야 한다 — 교집합으로 맞춘다.
    keys = None
    for alias, rec in runs.items():
        k = {r["problem"].strip(): r["y"] for r in rec if r["by_kind"].get("orig", {}).get("tokens")}
        keys = k if keys is None else {p: y for p, y in keys.items() if p in k}
    probs = sorted(keys)
    y = np.array([keys[p] for p in probs])
    words = np.array([max(1, len(re.findall(r"[A-Za-z]+", p))) for p in probs], float)
    lw = np.log(words)
    rng = np.random.default_rng(0)
    PY = np.array([rng.permutation(y) for _ in range(a.perms)], float)
    n1, n0 = float(y.sum()), float((1 - y).sum())

    def ranks(x):
        x = np.asarray(x, float)
        o = x.argsort(kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(1, len(x) + 1, dtype=float)
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
        return r

    def band(score):
        r = ranks(score)
        auc = (float(r[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
        nl = (PY @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    out(f"H20-M 모델별 원본 생성길이 · 공통 문제 {len(probs)} (robust {int(y.sum())} / spurious {int(len(y) - y.sum())}) "
        f"· null {a.perms}회 · 주지표 cap {PRIMARY}")
    out(f"모델 {len(runs)}종: " + ", ".join(sorted(runs)))
    out()
    out(f"{'모델':26s} {'cap':>6s} {'평균토큰':>9s} {'robust':>8s} {'spurious':>9s} {'AUC':>6s} {'null95%':>14s} {'길이잔차':>8s}")
    tok = {}                                        # (alias, cap) -> 문제별 평균 토큰
    for alias, rec in sorted(runs.items()):
        by = {r["problem"].strip(): r["by_kind"]["orig"]["tokens"] for r in rec
              if r["by_kind"].get("orig", {}).get("tokens")}
        for cap in CAPS:
            t = np.array([np.mean([min(x, cap) for x in by[p]]) for p in probs], float)
            tok[(alias, cap)] = t
            auc, lo, hi, flag = band(-t)            # 부호: 길수록 spurious
            ra, *_ = band(-resid(t))
            out(f"{alias:26s} {cap:6d} {t.mean():9.0f} {t[y == 1].mean():8.0f} {t[y == 0].mean():9.0f} "
                f"{auc:6.3f} [{lo:.2f},{hi:.2f}]{flag} {ra:8.3f}")
    out()

    aliases = sorted(runs)
    for cap in CAPS:
        out(f"모델 간 Spearman 순위상관 (cap {cap}) — 사전등록 판정 기준이 여기다")
        out("   " + " " * 26 + "".join(f"{b[:12]:>14s}" for b in aliases))
        vals = []
        for i, aa in enumerate(aliases):
            cells = []
            for j, bb in enumerate(aliases):
                rho = spearmanr(tok[(aa, cap)], tok[(bb, cap)]).statistic
                cells.append(f"{rho:14.3f}")
                if i < j:
                    vals.append(rho)
            out(f"   {aa:26s}" + "".join(cells))
        if vals:
            med = float(np.median(vals))
            verdict = ("기각: 모델별 생성은 같은 문제 순위를 잰다 (h20e 의 모델별 근거 없음)" if med >= 0.8
                       else "모델별 성분 실재 → h20e 지명 후보" if med < 0.5 else "보류 (0.5~0.8)")
            out(f"   쌍 {len(vals)}개 · 중앙값 **{med:.3f}** · 최소 {min(vals):.3f} · 최대 {max(vals):.3f} → {verdict}")
        out()

    # 풀링(모델 평균 z) 도 참고로
    out("참고: 모델 평균(모델별 z 표준화 후 평균) 점수")
    for cap in CAPS:
        z = np.mean([(tok[(aa, cap)] - tok[(aa, cap)].mean()) / (tok[(aa, cap)].std() + 1e-9) for aa in aliases], axis=0)
        auc, lo, hi, flag = band(-z)
        ra, *_ = band(-resid(z))
        out(f"   cap {cap}: AUC {auc:.3f} [{lo:.2f},{hi:.2f}]{flag} · 길이잔차 {ra:.3f}")
    out()
    out("모델 자기 라벨(참고, n 작음): train-main-v2 의 해당 모델 행만으로 본 AUC 는 별도 분석 필요 — 여기서는 문제 단위 라벨만 쓴다.")
    (ROOT / "experiments" / "results" / "h20m_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
