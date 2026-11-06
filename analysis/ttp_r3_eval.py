"""#LOOP R3 평가 — **정답(gold)을 아는 첫 측정.** Olmo T=1.0 자체 라벨 20문제. 비용 $0.

사전등록(CLAUDE.md '#LOOP R3 사전등록 보완') 그대로:
```
base_acc_hat   = 원본 3샘플 중 **gold 와 같은** 답 비율      (= cap 1024 에서의 진짜 정확도)
var_acc_hat[t] = 유형 t 의 3샘플 중 gold 와 같은 답 비율
md_gold        = base_acc_hat − min over t of var_acc_hat[t]  (주최측 max_drop 의 테스트타임 판)
```
* **주지표 A** `md_gold` 큼 = spurious (주최측 정의 그대로)
* **주지표 B** `md_gold` 큼 = robust (#LABEL-AUDIT 반대 방향 가설)
  → 같은 점수의 양방향이라 **동시 통과가 불가능**하다. 둘 다 null 을 못 넘으면 '부호 없음'.
* 보조: `md_pseudo`(의사정답판 — R1·R2 와 비교) · `base_acc_hat` 단독([author] 지적의 직접 측정)
임계는 문제 단위 **LOO 안에서만** 사전확률 교정 · 셔플 null 1000 · 길이 대조군 · 상수 병기.

사용: .venv/Scripts/python.exe analysis/ttp_r3_eval.py --dir <out> --ckpt 1024
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from ttp_eval import auc, loo_acc, null_band
from ttp_r2_eval import majority

REPORT = ROOT / "experiments/results/ttp_r3_report.txt"
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def build(samp, rows, ckpt, gold):
    """pidx → dict(md_gold, md_pseudo, base_gold, base_pseudo, vacc_gold, n_types)"""
    greedy = {r["pidx"]: r["val"].get(ckpt) for r in rows if r["kind"] == "orig"}
    by = {}
    for r in samp:
        by.setdefault(r["pidx"], {}).setdefault(r["kind"], []).append(r["val"].get(ckpt))
    res = {}
    for p, d in by.items():
        ov = [x for x in d.get("orig", []) if x is not None]
        g = gold.get(p)
        # --- 정답 기준
        bg = (sum(1 for x in ov if x == g) / len(ov)) if (ov and g is not None) else None
        vg = {}
        for t, vs in d.items():
            if t == "orig":
                continue
            u = [x for x in vs if x is not None]
            if u and g is not None:
                vg[t] = sum(1 for x in u if x == g) / len(u)
        md_g = (bg - min(vg.values())) if (bg is not None and len(vg) >= 2) else None
        # --- 의사정답 기준 (R1·R2 와 같은 방식)
        pg = majority(d.get("orig", []), greedy.get(p))
        bp = (sum(1 for x in ov if x == pg) / len(ov)) if (ov and pg is not None) else None
        vp = {}
        for t, vs in d.items():
            if t == "orig":
                continue
            u = [x for x in vs if x is not None]
            if u and pg is not None:
                vp[t] = sum(1 for x in u if x == pg) / len(u)
        md_p = (bp - min(vp.values())) if (bp is not None and len(vp) >= 2) else None
        res[p] = dict(md_gold=md_g, md_pseudo=md_p, base_gold=bg, base_pseudo=bp,
                      n_types=len(vg), gold=g, pseudo=pg)
    return res


def cell(vals, y, lw, reps, seed, flip=False):
    """None 은 길이 폴백으로 메운다(제출물과 같은 처리). flip 이면 부호를 뒤집는다."""
    sv = np.array([lw[i] / 100.0 + 0.5 if vals[i] is None
                   else (-float(vals[i]) if flip else float(vals[i]))
                   for i in range(len(y))], float)
    a, _ = loo_acc(sv, y)
    u = auc(sv, y)
    (_, ha), (_, hu) = null_band(sv, y, reps, seed)
    return a, u, ha, hu, int(sum(1 for v in vals if v is None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpt", default="1024")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    d = Path(a.dir)
    lab = json.loads((ROOT / "experiments/results/r3_labels.json").read_text(encoding="utf-8"))
    keep = [r for r in lab if r["label"] is not None]
    y = np.array([1 if r["label"] is True else 0 for r in keep], int)
    pid = [r["pidx"] for r in keep]
    gold = {r["pidx"]: r["gold"] for r in lab}
    lw = np.array([math.log(max(1, len(r["problem"].split()))) for r in keep], float)
    n = len(y)
    rate = float(y.mean())

    out("#LOOP R3 결과 — 정답(gold)을 아는 첫 테스트타임 측정 · Olmo T=1.0 자체 라벨")
    out(f"   라벨 {n}문제 (robust {int(y.sum())} / spurious {int(n - y.sum())}) · "
        f"상수 '전부 robust' 정확도 **{rate:.4f}** · 주 체크포인트 CoT {a.ckpt}")
    out(f"   **#LOOP 표의 '공식 33' 열과 직접 비교 불가** — 라벨 셋이 다르다(사전등록에 적음)")
    out()

    f = d / "Olmo-3-7B-Think_ttp_samp.json"
    rf = d / "Olmo-3-7B-Think_ttp_rows.json"
    if not (f.exists() and rf.exists()):
        out(f"!! 산출물이 없다: {f.name} / {rf.name}")
        REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
        return
    sc = build(json.loads(f.read_text(encoding="utf-8")),
               json.loads(rf.read_text(encoding="utf-8")), a.ckpt, gold)

    # 대조군
    al, _ = loo_acc(lw, y)
    out(f"## 대조군 — 길이(LOO 임계) 정확도 {al:.4f} · AUC {auc(lw, y):.4f} · 상수 {rate:.4f}")
    out()

    out("## 주지표 · 보조 (정확도 / AUC / null 95% 상단 / 폴백 수)")
    out(f"   {'지표':<34}{'정확도':>8}{'AUC':>8}{'null상단':>10}{'폴백':>6}{'판정':>12}")
    rowsout = {}
    for nm, key, flip, tag in (
            ("**주지표 A** md_gold 큼=spurious", "md_gold", False, "A"),
            ("**주지표 B** md_gold 큼=robust", "md_gold", True, "B"),
            ("보조 md_pseudo 큼=spurious", "md_pseudo", False, ""),
            ("보조 md_pseudo 큼=robust", "md_pseudo", True, ""),
            ("보조 base_acc_hat 낮으면 spurious", "base_gold", True, ""),
            ("보조 base_acc_hat 높으면 spurious", "base_gold", False, "")):
        vals = [sc.get(p, {}).get(key) for p in pid]
        acc, u, ha, hu, nfb = cell(vals, y, lw, a.reps, 61, flip)
        verd = ("**null 초과**" if acc > ha else "미달") if tag else ("초과" if acc > ha else "미달")
        out(f"   {nm:<34}{acc:>8.3f}{u:>8.3f}{ha:>10.3f}{nfb:>6}{verd:>12}")
        if tag:
            rowsout[tag] = (acc, u, ha)
    out()

    out("## 사전등록 판정")
    A, B = rowsout.get("A"), rowsout.get("B")
    if A and B:
        pa = A[0] > A[2]
        pb = B[0] > B[2]
        out(f"   주지표 A (주최측 정의 그대로): 정확도 {A[0]:.4f} vs null 상단 {A[2]:.3f} → "
            f"{'**통과**' if pa else '미달'}")
        out(f"   주지표 B (반대 방향 가설):     정확도 {B[0]:.4f} vs null 상단 {B[2]:.3f} → "
            f"{'**통과**' if pb else '미달'}")
        if pa and not pb:
            out("   → **부호 확정: 주최측 정의 방향(하락 큼 = spurious).** 반대 방향 가설은 기각된다.")
        elif pb and not pa:
            out("   → **부호 확정: 반대 방향(하락 큼 = robust) 1건.** 새 데이터에서의 확증이다.")
        elif pa and pb:
            out("   → 둘 다 통과? 같은 점수의 양방향이므로 **구현을 의심해야 한다** — 멈추고 감사한다.")
        else:
            out("   → **부호 없음.** 어느 방향도 null 을 넘지 못했다 — R2 오프라인 분해(AUC 0.50 근처)가")
            out("     가리킨 쪽이고, 이것으로 '섭동 흔들림' 계열을 닫는다.")
        out(f"   상수 {rate:.4f} 대비 A {A[0] - rate:+.4f} · B {B[0] - rate:+.4f} · "
            f"길이 {al:.4f} 대비 A {A[0] - al:+.4f}")
    out()

    out("## 진단 — [author] 지적의 직접 측정 (cap 1024 에서의 **진짜** 정확도)")
    bg = [sc[p]["base_gold"] for p in pid if sc.get(p, {}).get("base_gold") is not None]
    bp = [sc[p]["base_pseudo"] for p in pid if sc.get(p, {}).get("base_pseudo") is not None]
    lab_base = [r["base_acc"] for r in keep]
    out(f"   테스트타임 `base_acc_hat`(정답 기준, 원본 3샘플 · cap 1024) 평균 **{np.mean(bg):.3f}** · "
        f"중앙 {np.median(bg):.3f} · 0 인 문제 {sum(1 for v in bg if v == 0)}/{len(bg)}")
    out(f"   의사정답 기준 `base_acc` 평균 {np.mean(bp):.3f} — 정답 기준보다 "
        f"{np.mean(bp) - np.mean(bg):+.3f} 높다(일관되게 틀리면 의사정답이 과대평가한다)")
    out(f"   라벨 생성 때의 `base_acc`(k=8 · cap **32768**) 평균 **{np.mean(lab_base):.3f}**")
    out(f"   → **예산 차이가 정확도를 {np.mean(lab_base) - np.mean(bg):+.3f} 만큼 가른다.**")
    mdg = [sc[p]["md_gold"] for p in pid if sc.get(p, {}).get("md_gold") is not None]
    out(f"   `md_gold` 서로 다른 값 {len(set(np.round(mdg, 4)))} {sorted(set(np.round(mdg, 3)))}")
    out(f"   `md_gold` robust 평균 {np.mean([v for v, t in zip(mdg, y) if t == 1]):.3f} vs "
        f"spurious 평균 {np.mean([v for v, t in zip(mdg, y) if t == 0]):.3f}")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
