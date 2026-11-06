"""#LOOP R2 오프라인 분해 — R2 가 산 데이터로 추가 GPU 비용 0 에 세 가지를 본다 (탐색).

**전부 탐색이다**(R2 결과를 본 뒤의 계산). 판정에 쓰지 않고 **R3 설계 판단**에만 쓴다:
1. **유형별 분해** — 사전등록 때 경고한 '`rename2` 가 반신호가 가장 센 유형' 이 R2 를 끌어내렸나
2. **`base_acc` 단독** — 원본 확신도만으로 예측하면 어떻게 되나 (지휘 R2 (3) 의 단독 효과)
3. **부호를 뒤집으면** — R3 가설('흔들림 큼 = robust')을 같은 33문제에서 보면 얼마가 되나
   (※ 이건 **이미 본 데이터**라 R3 검증이 아니다. 뒤집은 값이 얼마나 되는지 '규모' 만 본다)

사용: .venv/Scripts/python.exe analysis/ttp_r2_offline.py --dir <out> --ckpt 1024
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
from h21_uncertainty import problems
from ttp_eval import auc, loo_acc, null_band, pidx_map, reindex
from ttp_r2_eval import majority

REPORT = ROOT / "experiments/results/ttp_r2_offline.txt"
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def scores(samp, rows, ckpt, types=None):
    """pidx → dict(md, base, vacc). types 를 주면 그 유형만으로 min 을 잡는다."""
    greedy = {r["pidx"]: r["val"].get(ckpt) for r in rows if r["kind"] == "orig"}
    by = {}
    for r in samp:
        by.setdefault(r["pidx"], {}).setdefault(r["kind"], []).append(r["val"].get(ckpt))
    res = {}
    for p, d in by.items():
        gold = majority(d.get("orig", []), greedy.get(p))
        ov = [x for x in d.get("orig", []) if x is not None]
        if gold is None or not ov:
            res[p] = dict(md=None, base=None, vacc={})
            continue
        base = sum(1 for x in ov if x == gold) / len(ov)
        vacc = {}
        for t, vs in d.items():
            if t == "orig" or (types is not None and t not in types):
                continue
            u = [x for x in vs if x is not None]
            if u:
                vacc[t] = sum(1 for x in u if x == gold) / len(u)
        res[p] = dict(md=(base - min(vacc.values())) if vacc else None, base=base, vacc=vacc)
    return res


def ev(sv, y, reps, seed):
    a, _ = loo_acc(sv, y)
    u = auc(sv, y)
    (_, ha), (_, hu) = null_band(sv, y, reps, seed)
    return a, u, ha, hu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpt", default="1024")
    ap.add_argument("--reps", type=int, default=600)
    a = ap.parse_args()
    d = Path(a.dir)
    rows_lab = problems()
    y = np.array([r["y"] for r in rows_lab], int)
    txt = [r["problem"] for r in rows_lab]
    n = len(y)
    p2i = pidx_map(txt)
    lw = np.array([math.log(max(1, len(t.split()))) for t in txt], float)
    models = {}
    for f in sorted(d.glob("*_ttp_samp.json")):
        nm = f.name.replace("_ttp_samp.json", "")
        rf = d / f"{nm}_ttp_rows.json"
        if rf.exists():
            models[nm] = (json.loads(f.read_text(encoding="utf-8")),
                          json.loads(rf.read_text(encoding="utf-8")))

    out(f"#LOOP R2 오프라인 분해 (ckpt {a.ckpt} · 모델 {len(models)} · **전부 탐색**)")
    out(f"   상수 {y.mean():.4f} · 길이 h15C(LOO) {loo_acc(lw, y)[0]:.4f}")
    out()

    def fill(sc, key, flip=False):
        v = []
        for i in range(n):
            x = sc.get(i, {}).get(key)
            v.append(lw[i] / 100.0 + 0.5 if x is None else (-float(x) if flip else float(x)))
        return np.array(v, float)

    # 1) 유형별 분해
    out("## 1. 유형별 분해 — 사전등록 경고('rename2 가 반신호') 가 맞았나")
    out(f"   {'유형 집합':<20}" + "".join(f"{m[:13]:>15}" for m in models) + f"{'모델평균':>10}{'null상단':>10}")
    for nm_set, ts in (("둘 다(사전등록)", None), ("rename2 만", {"rename2"}), ("typos0 만", {"typos0"})):
        accs, nulls = [], []
        for mn, (samp, rws) in models.items():
            sc = reindex(scores(samp, rws, a.ckpt, ts), p2i)
            acc, u, ha, hu = ev(fill(sc, "md"), y, a.reps, 41)
            accs.append(acc); nulls.append(ha)
        out(f"   {nm_set:<20}" + "".join(f"{v:>15.3f}" for v in accs) +
            f"{np.mean(accs):>10.3f}{np.mean(nulls):>10.3f}")
    out()

    # 2) base_acc 단독
    out("## 2. `base_acc` 단독 — 원본 확신도만으로 (부호 둘 다 본다)")
    out(f"   {'부호':<20}" + "".join(f"{m[:13]:>15}" for m in models) + f"{'모델평균':>10}")
    for nm_s, fl in (("낮으면 spurious", True), ("높으면 spurious", False)):
        accs, aucs = [], []
        for mn, (samp, rws) in models.items():
            sc = reindex(scores(samp, rws, a.ckpt), p2i)
            sv = fill(sc, "base", flip=fl)
            acc, u, ha, hu = ev(sv, y, a.reps, 43)
            accs.append(acc); aucs.append(u)
        out(f"   {nm_s:<20}" + "".join(f"{v:>15.3f}" for v in accs) +
            f"{np.mean(accs):>10.3f}   AUC {np.mean(aucs):.3f}")
    out()

    # 3) 부호 뒤집기 (규모만)
    out("## 3. 부호를 뒤집으면 — R3 가설의 '규모' 만 본다 (**이미 본 데이터라 검증 아니다**)")
    out(f"   {'방향':<24}" + "".join(f"{m[:13]:>15}" for m in models) + f"{'모델평균':>10}{'AUC':>8}")
    for nm_s, fl in (("사전등록(하락 큼=spur)", False), ("**뒤집음**(하락 큼=robust)", True)):
        accs, aucs = [], []
        for mn, (samp, rws) in models.items():
            sc = reindex(scores(samp, rws, a.ckpt), p2i)
            acc, u, ha, hu = ev(fill(sc, "md", flip=fl), y, a.reps, 47)
            accs.append(acc); aucs.append(u)
        out(f"   {nm_s:<24}" + "".join(f"{v:>15.3f}" for v in accs) +
            f"{np.mean(accs):>10.3f}{np.mean(aucs):>8.3f}")
    out()
    out("   (뒤집어도 상수 0.606 을 못 넘으면 '부호 문제' 로도 설명되지 않는다는 뜻이다.)")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
