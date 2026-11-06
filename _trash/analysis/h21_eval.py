"""H21 평가 — 자기 불확실성 3종. 사전등록 부호 고정. GPU 0. (지휘 #65)

사전등록 (측정 전 기록, CLAUDE.md "H21 자기 불확실성")
  H21a mean_entropy : 클수록 spurious        → 점수 = −entropy
  H21b ans_logprob  : 낮을수록 spurious      → 점수 = +logprob
  H21c drop         : 클수록 robust          → 점수 = +drop
  H21d d_irrel      : 작을수록 robust        → 점수 = −d_irrel   (무관 편집에 둔감해야 한다)
  H21p = drop − d_irrel : 클수록 robust      ← **민감도 프로파일 주지표** (출처: [author] 가설)
  보조(탐색): mean_logprob · mink_logprob · hi_ent_frac
검문소: 문제 단위 permutation null 1000 · log(단어수) 잔차 AUC 병기 ·
       **모델 간 분리력**(같은 문제에서 robust 모델 점수가 spurious 모델보다 높은 쌍의 비율) 별도 표.

모델 간 분리력이 이 실험의 핵이다. 문제 단위 AUC 는 h20 계열에서도 높게 나왔지만(최대 0.85)
리더보드에서는 0.57 에 머물렀다 — 검증셋이 모델별로 라벨을 섞어 놓았기 때문이다.
모델×문제 성분을 재려면 **같은 문제 안에서 모델을 비교**해야 한다.

사용:  .venv/Scripts/python.exe analysis/h21_eval.py [--stage both] [--perms 1000]
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H21 = ROOT / "experiments" / "results" / "h21"
SPECS = {                      # 지표 -> (부호, 단계, 주지표 여부)
    "mean_entropy": (-1, "gen", True),
    "mean_logprob": (+1, "gen", False),
    "mink_logprob": (+1, "gen", False),
    "hi_ent_frac": (-1, "gen", False),
    "ans_logprob": (+1, "conf", True),
    "drop": (+1, "conf", True),
    # H21d·H21p (지휘 #67) — [author]의 민감도 프로파일 가설을 출력 확률로 재구현한 것
    "h21p": (+1, "sens", True),            # drop − d_irrel · 큼 = robust  ← 주지표
    "d_irrel": (-1, "sens", False),        # 무관 편집에서의 |logp 변화| 평균 · 작을수록 robust
    "d_irrel_signed": (-1, "sens", False),
    "d_rename": (-1, "sens", False),
    "d_typos": (-1, "sens", False),
}


def load(stage):
    out = {}
    for f in sorted(glob.glob(str(H21 / f"*_{stage}.json"))):
        alias = Path(f).name.replace(f"_{stage}.json", "")
        out[alias] = json.loads(Path(f).read_text(encoding="utf-8"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="both", choices=["conf", "gen", "sens", "both"])
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    stages = ["conf", "gen", "sens"] if a.stage == "both" else [a.stage]
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    for stage in stages:
        runs = load(stage)
        if not runs:
            out(f"[{stage}] 결과 파일 없음 ({H21}/*_{stage}.json)")
            continue
        alias0 = sorted(runs)[0]
        probs = [r["problem"] for r in runs[alias0]]
        y = np.array([r["y"] for r in runs[alias0]])
        lw = np.log(np.array([max(1, len(re.findall(r"[A-Za-z]+", p))) for p in probs], float))
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

        def band(score, mask):
            s, yy = np.asarray(score, float)[mask], y[mask]
            r = ranks(s)
            k1, k0 = float(yy.sum()), float((1 - yy).sum())
            if k1 == 0 or k0 == 0:
                return None
            auc = (float(r[yy == 1].sum()) - k1 * (k1 + 1) / 2) / (k1 * k0)
            nl = (PY[:, mask] @ r - k1 * (k1 + 1) / 2) / (k1 * k0) if mask.all() else None
            if nl is None:                      # 부분집합이면 그 안에서 다시 섞는다
                nl = np.array([(ranks(s)[rng.permutation(yy) == 1].sum() - k1 * (k1 + 1) / 2) / (k1 * k0)
                               for _ in range(a.perms)])
            lo, hi = np.percentile(nl, [2.5, 97.5])
            return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

        def resid(x, mask):
            X = np.column_stack([np.ones(int(mask.sum())), lw[mask]])
            b, *_ = np.linalg.lstsq(X, np.asarray(x, float)[mask], rcond=None)
            r = np.asarray(x, float).copy()
            r[mask] = np.asarray(x, float)[mask] - X @ b
            return r

        keys = [k for k, (_, st, _) in SPECS.items() if st == stage]
        out(f"[{stage}] 모델 {len(runs)}종 · 문제 {len(y)} (robust {int(y.sum())}/spurious {int(len(y) - y.sum())}) "
            f"· null {a.perms}회 · ▲▼ 는 사전등록 부호 기준")
        out(f"{'모델':26s} {'지표':14s} {'n':>3s} {'robust평균':>10s} {'spurious평균':>12s} {'AUC':>6s} {'null95%':>14s} {'길이잔차':>8s}")
        store = {}
        for alias, rec in sorted(runs.items()):
            for k in keys:
                sign, _, primary = SPECS[k]
                v = np.array([np.nan if r.get(k) is None else float(r[k]) for r in rec])
                mask = ~np.isnan(v)
                if mask.sum() < 8 or len(set(y[mask])) < 2:
                    out(f"{alias:26s} {k:14s} {int(mask.sum()):3d}  (판정 불가)")
                    continue
                store[(alias, k)] = (v, mask)
                b = band(sign * np.nan_to_num(v), mask)
                rv = resid(sign * np.nan_to_num(v), mask)
                rb = band(rv, mask)
                star = "*" if primary else " "
                out(f"{alias:26s} {k:14s}{star}{int(mask.sum()):3d} {v[mask & (y == 1)].mean():10.3f} "
                    f"{v[mask & (y == 0)].mean():12.3f} {b[0]:6.3f} [{b[1]:.2f},{b[2]:.2f}]{b[3]} {rb[0]:8.3f}")
        out("   * = 사전등록 주지표")
        out()

        # ---- 모델 간 분리력: 같은 문제에서 모델 자기 라벨이 갈리는 쌍 ----
        out(f"[{stage}] 모델 간 분리력 — 같은 문제에서 robust 모델 점수 > spurious 모델 점수인 쌍의 비율")
        per = {}                                 # (pidx, alias) -> (label, {지표: 값})
        for alias, rec in runs.items():
            full = alias.replace("_", ".")       # 별칭 복원용(매칭은 접미사로 한다)
            for r in rec:
                lab = None
                for mid, v in r.get("per_model", {}).items():
                    if mid.split("/")[-1].replace(".", "_") == alias:
                        lab = bool(v)
                if lab is None:
                    continue
                per[(r["pidx"], alias)] = (lab, {k: r.get(k) for k in keys}, full)
        for k in keys:
            sign = SPECS[k][0]
            byp = {}
            for (pidx, alias), (lab, vals, _) in per.items():
                if vals.get(k) is None:
                    continue
                byp.setdefault(pidx, []).append((lab, sign * float(vals[k]), alias))
            ok = tot = 0
            for pidx, items in byp.items():
                for i in range(len(items)):
                    for j in range(len(items)):
                        if items[i][0] and not items[j][0]:
                            tot += 1
                            ok += int(items[i][1] > items[j][1])
            out(f"   {k:14s} 갈리는 쌍 {tot:3d} · 맞춘 비율 {('-' if tot == 0 else f'{ok / tot:.3f}')}"
                + ("  ← 0.5 초과면 모델×문제 성분의 증거" if tot else "  (모델 자기 라벨이 갈리는 같은 문제가 없다)"))
        out()
    (ROOT / "experiments" / "results" / "h21_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
