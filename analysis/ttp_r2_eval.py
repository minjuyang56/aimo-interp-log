"""#LOOP R2 평가 — 주최측 `max_drop` 흉내. 공식 33문제. 비용 $0 (R1 파드가 산 샘플 재사용).

사전등록(CLAUDE.md '#LOOP R2 를 R1 파드에 합친다') 그대로:
```
pseudo_gold  = 원본 N샘플의 다수결 (다수결 없으면 greedy 원본의 답)
base_acc     = 원본 N샘플 중 pseudo_gold 와 같은 비율            (지휘 (3) 확신도 항)
var_acc[t]   = 유형 t 의 N샘플 중 pseudo_gold 와 같은 비율
max_drop_hat = base_acc − min over t of var_acc[t]               높을수록 spurious
robust  <=>  max_drop_hat <= THR     (THR 은 모델별 · 문제단위 LOO 안에서만 교정)
```
2차 규칙 `max_drop_hat <= 0.10` 문자 그대로 · 보조 `combo = max_drop_hat + (1 − base_acc)`.
R1 과 **같은 LOO·같은 셔플 null·같은 길이 대조군**을 쓴다(`ttp_eval` 에서 그대로 import).

사용: .venv/Scripts/python.exe analysis/ttp_r2_eval.py --dir <out 디렉터리>
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))
from ttp_eval import auc, loo_acc, null_band, pidx_map, reindex          # R1 과 같은 절차를 재사용한다

REPORT = ROOT / "experiments/results/ttp_r2_report.txt"
PRIMARY_CKPT = "512"      # --ckpt 로 덮어쓴다 (R2 사전등록 주지표는 1024)
# 쓸 수 있는 섭동 유형이 이보다 적으면 판정 불가 (R1 의 K_MIN 과 같은 역할).
# **R2 는 유형이 2종이므로 2 다** — 3 으로 두면 전 문제가 판정 불가가 되어 길이 폴백만 남는다.
# 즉 '두 유형 다 답이 나왔을 때만 판정한다'. 사전등록에 이 값을 박아 둔다.
T_MIN = int(__import__("os").environ.get("TTP_R2_TMIN", 2))
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def majority(vals, fallback=None):
    """결정적 다수결. 동점이면 fallback(greedy 답), 그것도 없으면 가장 작은 값."""
    v = [x for x in vals if x is not None]
    if not v:
        return None
    c = Counter(v).most_common()
    top = c[0][1]
    tied = sorted(x for x, n in c if n == top)
    if len(tied) == 1:
        return tied[0]
    if fallback is not None and fallback in tied:
        return fallback
    return tied[0]


def r2_scores(samp, rows, ckpt):
    """pidx → dict(max_drop_hat, base_acc, combo, n_types, gold)"""
    greedy = {r["pidx"]: r["val"].get(ckpt) for r in rows if r["kind"] == "orig"}
    by = {}
    for r in samp:
        by.setdefault(r["pidx"], {}).setdefault(r["kind"], []).append(r["val"].get(ckpt))
    res = {}
    for p, d in by.items():
        gold = majority(d.get("orig", []), greedy.get(p))
        if gold is None:
            res[p] = dict(md=None, base=None, combo=None, n_types=0, gold=None)
            continue
        ov = [x for x in d.get("orig", []) if x is not None]
        base = (sum(1 for x in ov if x == gold) / len(ov)) if ov else None
        vacc = {}
        for t, vs in d.items():
            if t == "orig":
                continue
            u = [x for x in vs if x is not None]
            if u:
                vacc[t] = sum(1 for x in u if x == gold) / len(u)
        if base is None or len(vacc) < T_MIN:
            res[p] = dict(md=None, base=base, combo=None, n_types=len(vacc), gold=gold)
            continue
        md = base - min(vacc.values())
        res[p] = dict(md=md, base=base, combo=md + (1.0 - base), n_types=len(vacc), gold=gold,
                      vacc=vacc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--ckpt", default=None, help="주 체크포인트 (R2 사전등록 = 1024)")
    a = ap.parse_args()
    d = Path(a.dir)
    global PRIMARY_CKPT
    if a.ckpt:
        PRIMARY_CKPT = str(a.ckpt)
    from h21_uncertainty import problems
    rows_lab = problems()
    y = np.array([r["y"] for r in rows_lab], int)
    txt = [r["problem"] for r in rows_lab]
    n = len(y)
    rate = float(y.mean())
    lw = np.array([math.log(max(1, len(t.split()))) for t in txt], float)

    p2i = pidx_map(txt)
    out("#LOOP R2 결과 — 주최측 max_drop 흉내 (공식 33문제, 비용 $0)")
    out(f"   라벨 robust {int(y.sum())}/{n} · 상수 정확도 {rate:.4f} · 주 체크포인트 CoT {PRIMARY_CKPT}")
    out(f"   지표 max_drop_hat = base_acc − min_t var_acc[t] · 임계는 모델별 · LOO 안에서만")
    out()

    models = {}
    for f in sorted(d.glob("*_ttp_samp.json")):
        nm = f.name.replace("_ttp_samp.json", "")
        rf = d / f"{nm}_ttp_rows.json"
        if not rf.exists():
            continue
        models[nm] = (json.loads(f.read_text(encoding="utf-8")),
                      json.loads(rf.read_text(encoding="utf-8")))
    if not models:
        out(f"!! {d} 에 *_ttp_samp.json 이 없다")
        REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
        return

    ckpts = sorted({c for s, _ in models.values() for r in s for c in r["val"]}, key=int)
    acc_len, _ = loo_acc(lw, y)
    out(f"## 대조군 — 길이 h15C(LOO) 정확도 {acc_len:.4f} · 상수 {rate:.4f}")
    out()

    summary = {}
    for metric in ("md", "combo"):
        out(f"## 지표 {metric}" + ("  (주지표)" if metric == "md" else "  (보조)"))
        out(f"   {'ckpt':>5} " + " ".join(f"{m[:14]:>17}" for m in models) +
            f"{'모델평균':>12}{'null상단':>10}")
        for ck in ckpts:
            accs, aucs, nulls, cells = [], [], [], []
            for nm, (samp, rows) in models.items():
                sc = reindex(r2_scores(samp, rows, ck), p2i)
                vals, nfb = [], 0
                for i in range(n):
                    v = sc.get(i, {}).get(metric)
                    if v is None:
                        nfb += 1
                        vals.append(lw[i] / 100.0 + 0.5)        # 판정 불가 → 길이 폴백 (제출물과 같다)
                    else:
                        vals.append(float(v))
                sv = np.array(vals, float)
                ac, _ = loo_acc(sv, y)
                au = auc(sv, y)
                (_, ha), (_, hu) = null_band(sv, y, a.reps, 23)
                accs.append(ac); aucs.append(au); nulls.append(ha)
                cells.append(f"{ac:.3f}/{au:.3f}" + (f"(fb{nfb})" if nfb else ""))
            out(f"   {ck:>5} " + " ".join(f"{c:>17}" for c in cells) +
                f"{np.mean(accs):>7.3f}/{np.mean(aucs):.3f}{np.mean(nulls):>10.3f}")
            if metric == "md":
                summary[ck] = dict(acc=float(np.mean(accs)), auc=float(np.mean(aucs)),
                                   null_acc=float(np.mean(nulls)),
                                   per_model=dict(zip(models, accs)))
        out()

    p = summary.get(PRIMARY_CKPT)
    if p:
        out("## 사전등록 판정 (R2 주지표 max_drop_hat · ckpt " + PRIMARY_CKPT + ")")
        out(f"   모델평균 정확도 **{p['acc']:.4f}** · 셔플 null 95% 상단 {p['null_acc']:.3f} · "
            f"AUC {p['auc']:.4f}")
        out(f"   상수 {rate:.4f} 대비 {p['acc'] - rate:+.4f} · 길이 {acc_len:.4f} 대비 "
            f"{p['acc'] - acc_len:+.4f}")
        out(f"   zip 빌드 조건 >0.645(H22) → {'충족' if p['acc'] > 0.645 else '**미달 — zip 만들지 않는다**'}")
        out(f"   라운드 통과 >=0.706 → {'통과' if p['acc'] >= 0.706 else '**미달**'} · "
            f"루프 목표 >=0.90 → {'달성' if p['acc'] >= 0.90 else '**미달**'}")
        out()

    out("## 사전지정 2차 규칙 — max_drop_hat <= 0.10 (주최측 임계 문자 그대로)")
    for ck in ckpts:
        accs = []
        for nm, (samp, rows) in models.items():
            sc = reindex(r2_scores(samp, rows, ck), p2i)
            pred = np.array([(sc.get(i, {}).get("md") is not None and sc[i]["md"] <= 0.10)
                             for i in range(n)])
            accs.append(float((pred == (y == 1)).mean()))
        out(f"   ckpt {ck:>5}: " + " ".join(f"{m[:10]} {v:.3f}" for m, v in zip(models, accs)) +
            f" · 모델평균 {np.mean(accs):.4f}")
    out()

    out("## 진단 — 해상도 (사전등록에 적어 둔 천장)")
    for nm, (samp, rows) in models.items():
        sc = reindex(r2_scores(samp, rows, PRIMARY_CKPT), p2i)
        md = [v["md"] for v in sc.values() if v["md"] is not None]
        bs = [v["base"] for v in sc.values() if v["base"] is not None]
        nt = [v["n_types"] for v in sc.values()]
        out(f"   {nm:<30} 판정 {len(md)}/{n} · max_drop_hat 서로 다른 값 {len(set(np.round(md, 4)))} "
            f"{sorted(set(np.round(md, 3)))[:8]} · base_acc 평균 "
            f"{np.mean(bs) if bs else float('nan'):.3f} · 유형 수 평균 {np.mean(nt):.2f}")
    out()
    out("주최측 max_drop 격자는 0.01(≈100샘플)이고 우리는 N=3 이라 0.333 이다 — 33배 거칠다.")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(summary, open(ROOT / "experiments/results/ttp_r2_summary.json", "w"), indent=2)
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
