"""#LOOP R4 — 원본 3샘플 **자기불일치**. 정답도 의사정답도 필요 없다. $0 오프라인.

사전등록(CLAUDE.md '#LOOP R4') 그대로:
```
disagree = 1 − (원본 3샘플 중 자기 다수결과 일치하는 비율)   ∈ {0, 1/3, 2/3}
```
* **주지표 부호: 큼 = spurious** · 사전지정 2차: 반대 부호 · 보조: 서로 다른 답의 개수
* 평가 둘: **공식 33**(R2 데이터 · 3모델 — **이미 본 값의 재보고**) · **R3 19문제**(새 칸)
* LOO 사전확률 교정 · 셔플 null 1000 · 길이 대조군·상수 병기 · ckpt 1024 주지표, 512 민감도

사용: .venv/Scripts/python.exe analysis/ttp_r4_eval.py
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
from h21_uncertainty import problems
from ttp_eval import auc, loo_acc, null_band, pidx_map, reindex

R2DIR = ROOT / "experiments/results/runpod/ttperturb/ngksp4ki8uagky/out"
R3DIR = ROOT / "experiments/results/runpod/ttperturb/og7itzdazdjk9y/out"
REPORT = ROOT / "experiments/results/ttp_r4_report.txt"
REPS = 1000
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def disagree_map(samp, ckpt):
    """pidx → (disagree, 서로 다른 답 개수). 원본 샘플만 쓴다. 답이 2개 미만이면 None."""
    by = {}
    for r in samp:
        if r["kind"] == "orig":
            by.setdefault(r["pidx"], []).append(r["val"].get(ckpt))
    res = {}
    for p, vs in by.items():
        u = [v for v in vs if v is not None]
        if len(u) < 2:
            res[p] = (None, None)
            continue
        top = Counter(u).most_common(1)[0][1]
        res[p] = (1.0 - top / len(u), len(set(u)))
    return res


def cells(vals, y, lw, seed, label):
    """양방향 + 길이 폴백 처리. (정확도, AUC, null상단, 폴백 수) 두 쌍."""
    outp = {}
    for nm, flip in (("큼=spurious", False), ("큼=robust", True)):
        sv = np.array([lw[i] / 100.0 + 0.5 if vals[i] is None
                       else (-float(vals[i]) if flip else float(vals[i]))
                       for i in range(len(y))], float)
        a, _ = loo_acc(sv, y)
        u = auc(sv, y)
        (_, ha), (_, hu) = null_band(sv, y, REPS, seed)
        outp[nm] = (a, u, ha, int(sum(1 for v in vals if v is None)))
    return outp


def main():
    out("#LOOP R4 — 원본 3샘플 자기불일치 (정답 불필요) · $0 오프라인")
    out("   **공식 33 칸은 R2 오프라인에서 이미 본 값의 재보고다**(disagree = 1 − base_pseudo 의 단조 변환).")
    out("   **R3 19문제 칸이 유일한 새 측정이다.** 다중비교 8칸 — 사전등록에 적어 둔 그대로.")
    out()

    # ---------------- 공식 33 (R2 데이터)
    rows_lab = problems()
    y33 = np.array([r["y"] for r in rows_lab], int)
    txt33 = [r["problem"] for r in rows_lab]
    lw33 = np.array([math.log(max(1, len(t.split()))) for t in txt33], float)
    p2i = pidx_map(txt33)
    al33, _ = loo_acc(lw33, y33)
    out(f"## 공식 33 (R2 데이터 · 3모델) — 상수 {y33.mean():.4f} · 길이 {al33:.4f}")
    out(f"   {'ckpt':>5} {'모델':<28}{'큼=spur 정확도/AUC':>22}{'큼=robust':>18}{'null상단':>10}{'폴백':>6}")
    agg = {}
    for ck in ("512", "1024"):
        acc_s, acc_r, nul = [], [], []
        for f in sorted(R2DIR.glob("*_ttp_samp.json")):
            nm = f.name.replace("_ttp_samp.json", "")
            dm = reindex(disagree_map(json.loads(f.read_text(encoding="utf-8")), ck), p2i)
            vals = [dm.get(i, (None, None))[0] for i in range(len(y33))]
            c = cells(vals, y33, lw33, 71, nm)
            acc_s.append(c["큼=spurious"][0]); acc_r.append(c["큼=robust"][0]); nul.append(c["큼=spurious"][2])
            out(f"   {ck:>5} {nm[:28]:<28}"
                f"{c['큼=spurious'][0]:>13.3f}/{c['큼=spurious'][1]:.3f}"
                f"{c['큼=robust'][0]:>11.3f}/{c['큼=robust'][1]:.3f}"
                f"{c['큼=spurious'][2]:>10.3f}{c['큼=spurious'][3]:>6}")
        agg[ck] = (float(np.mean(acc_s)), float(np.mean(acc_r)), float(np.mean(nul)))
        out(f"   {ck:>5} {'**모델평균**':<28}{np.mean(acc_s):>13.3f}      {np.mean(acc_r):>11.3f}"
            f"      {np.mean(nul):>10.3f}")
    out()

    # ---------------- R3 19문제 (새 칸)
    lab = json.loads((ROOT / "experiments/results/r3_labels.json").read_text(encoding="utf-8"))
    keep = [r for r in lab if r["label"] is not None]
    y19 = np.array([1 if r["label"] is True else 0 for r in keep], int)
    pid19 = [r["pidx"] for r in keep]
    lw19 = np.array([math.log(max(1, len(r["problem"].split()))) for r in keep], float)
    al19, _ = loo_acc(lw19, y19)
    out(f"## R3 19문제 (Olmo 자체 라벨) — **새 칸** · 상수 {y19.mean():.4f} · 길이 {al19:.4f}")
    out(f"   {'ckpt':>5} {'큼=spur 정확도/AUC':>22}{'큼=robust':>18}{'null상단':>10}{'폴백':>6}")
    f3 = R3DIR / "Olmo-3-7B-Think_ttp_samp.json"
    r3cells = {}
    for ck in ("512", "1024"):
        dm = disagree_map(json.loads(f3.read_text(encoding="utf-8")), ck)
        vals = [dm.get(p, (None, None))[0] for p in pid19]
        c = cells(vals, y19, lw19, 73, "Olmo")
        r3cells[ck] = c
        out(f"   {ck:>5} {c['큼=spurious'][0]:>13.3f}/{c['큼=spurious'][1]:.3f}"
            f"{c['큼=robust'][0]:>11.3f}/{c['큼=robust'][1]:.3f}"
            f"{c['큼=spurious'][2]:>10.3f}{c['큼=spurious'][3]:>6}")
        nd = [dm.get(p, (None, None))[1] for p in pid19]
        out(f"         서로 다른 답 개수 분포 {dict(sorted(Counter(x for x in nd if x).items()))}"
            f" · disagree 값 {sorted(set(round(v, 3) for v in vals if v is not None))}")
    out()

    # ---------------- 판정
    out("## 사전등록 판정 (주지표 = 공식 33 · ckpt 1024 · 큼=spurious)")
    a_s, a_r, nl = agg["1024"]
    out(f"   공식 33 모델평균: 큼=spurious **{a_s:.4f}** · 큼=robust {a_r:.4f} · null 상단 {nl:.3f}")
    out(f"      상수 {y33.mean():.4f} 대비 {a_s - y33.mean():+.4f} · 길이 {al33:.4f} 대비 {a_s - al33:+.4f}")
    c = r3cells["1024"]
    out(f"   R3 19문제 (새 칸): 큼=spurious **{c['큼=spurious'][0]:.4f}** · "
        f"큼=robust {c['큼=robust'][0]:.4f} · null 상단 {c['큼=spurious'][2]:.3f}")
    out(f"      상수 {y19.mean():.4f} 대비 {c['큼=spurious'][0] - y19.mean():+.4f} · "
        f"길이 {al19:.4f} 대비 {c['큼=spurious'][0] - al19:+.4f}")
    passed = (a_s > y33.mean() and a_s > al33)
    out()
    out(f"   → 상수·길이 둘 다 초과? **{'그렇다' if passed else '아니다'}** → "
        f"{'라운드 통과' if passed else '**기각. 지휘대로 루프를 종료한다.**'}")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
