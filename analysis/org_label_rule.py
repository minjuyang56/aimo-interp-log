"""#ORG-LABEL 시험1 — 주최측 공개 라벨로 **우리 라벨 규칙**을 검증한다 (지휘 #ORG-LABEL, 2026-10-15).

## 지휘가 요구한 쌍별 비교는 왜 불가능한가 (실행 전 측정으로 확정)
지휘는 "겹치는 (모델,문제)마다 주최측 max_drop vs 우리 max_drop" 을 요구했다. 그런데 우리 라벨
JSON **19개 전수**에서 공식 36문제와 완전 일치가 **0건**이고 `source_leak_audit.txt` 의 8-그램
자카드 최댓값도 **0.000** 이다. 원인은 설계다 — `genlab_v2.load_pool` 이
`rows = [r for r in rows if r["problem"] not in v2]` 로 공식 문제를 누출 방지 차원에서 **명시적으로
배제**한다. 즉 맞댈 쌍이 존재하지 않고 $0 으로는 만들 수도 없다. 그 사실을 이 스크립트가 먼저
다시 센다(주장 말고 계측).

## 그래서 같은 질문을 $0 으로 답하는 두 각도
**(1a) 임계 규칙 역추정 검증 — 주지표.** 우리는 `robust ⟺ max_drop ≤ 0.10` ·
`spurious ⟺ ≥ 0.25` · 사이 제외를 `train-main-v2` 에서 **역추정해** 모든 자체 라벨에 써 왔다
(FAQ 7: 공식 기준은 비공개). 그런데 그 규칙이 주최측 82행의 `is_robust` 를 **몇 건 재현하는지
숫자로 적은 적이 없다**. 주최측 표에 `max_drop` 과 `is_robust` 가 같이 있으므로 직접 검증이다.
  판정(사전등록): 일치율 ≥0.95 "같은 규칙" · 0.80~0.95 "근사" · <0.80 "다른 규칙".

**(1b) 분포 비교 — 보조, 판정 없음.** 문제집합·k·cap·디코딩이 전부 달라 기술통계만 적는다.

사용: .venv/Scripts/python.exe analysis/org_label_rule.py
"""

import glob
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "experiments/results/train-main-v2.parquet"
OUT = ROOT / "experiments/results/org_label_rule.txt"
LO, HI = 0.10, 0.25                      # 우리 규칙 (바꾸지 않는다)
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def our_rule(md, lo=LO, hi=HI):
    """우리 주라벨 규칙. 반환 'robust' / 'spurious' / 'excluded'."""
    if md is None or (isinstance(md, float) and math.isnan(md)):
        return "excluded"
    return "robust" if md <= lo else ("spurious" if md >= hi else "excluded")


def org_state(v):
    """주최측 `is_robust` 의 3상태. 결측 = 그들이 제외한 행."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "excluded"
    return "robust" if bool(v) else "spurious"


def resolution(vals):
    """값들의 격자 간격을 추정한다 — 라벨이 k샘플 정확도 차이면 1/k 의 배수여야 한다."""
    cents = sorted({int(round(v * 1000)) for v in vals})
    g = 0
    for c in cents:
        g = math.gcd(g, abs(c))
    return g / 1000.0, len(cents)


def load_ours():
    """우리 라벨 묶음별 max_drop. 라벨 JSON 에 있는 것만(npz 에는 max_drop 이 없는 묶음도 있다)."""
    sets = {}
    for f in sorted(glob.glob(str(ROOT / "experiments/results/runpod/**/*_labels.json"), recursive=True)) + \
            sorted(glob.glob(str(ROOT / "experiments/results/genlab/**/*labels*.json"), recursive=True)):
        try:
            rows = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list) or not rows:
            continue
        md = [(r.get("max_drop"), r.get("label"), r.get("gen_k"), r.get("gen_cap"))
              for r in rows if isinstance(r, dict) and r.get("max_drop") is not None]
        if len(md) < 10:
            continue
        rel = Path(f).relative_to(ROOT).as_posix()
        sets[rel] = md
    return sets


def main():
    df = pd.read_parquet(V2)
    out("#ORG-LABEL 시험1 — 주최측 공개 라벨로 우리 라벨 규칙 검증 (비용 $0)")
    out(f"출처 train-main-v2.parquet · {len(df)}행 · 고유 문제 {df.problem.nunique()} · "
        f"모델 {df.model_id.nunique()}종")
    out()

    # ---------- 쌍별 비교가 불가능하다는 것을 다시 센다 ----------
    off = {str(p).rstrip() for p in df.problem}
    ours = load_ours()
    hits = {k: sum(1 for r in json.load(open(ROOT / k, encoding="utf-8"))
                   if isinstance(r, dict) and str(r.get("problem", "")).rstrip() in off)
            for k in ours}
    out("## 지휘의 쌍별 비교 — **판정 불가 (겹침 0)**")
    out(f"   우리 라벨 묶음 {len(ours)}개에서 공식 36문제와 **완전 일치 {sum(hits.values())}건**")
    out("   (`source_leak_audit.txt`: 8-그램 자카드 최댓값도 다섯 묶음 전부 0.000 = 근사 중복도 없다)")
    out("   원인은 설계다 — `genlab_v2.load_pool:74` 가 `problem not in v2` 로 공식 문제를 배제한다.")
    out("   → 주최측 max_drop 과 우리 max_drop 을 같은 (모델,문제) 에서 맞댈 데이터가 **없다**.")
    out("     만들려면 공식 33문제에 우리 파이프라인을 새로 돌려야 하고(약 $5·1.6h) 지휘가 (1)을")
    out("     $0 으로 못 박았으므로 하지 않는다. 아래 두 각도로 대체한다.")
    out()

    # ---------- (1a) 임계 규칙 역추정 검증 ----------
    out("## (1a) 임계 규칙 역추정 검증 — **주지표**")
    out(f"   우리 규칙: max_drop ≤ {LO} → robust · ≥ {HI} → spurious · 사이 → 제외")
    md_all = [float(v) for v in df.max_drop]
    step, nuniq = resolution(md_all)
    out(f"   주최측 max_drop 해상도: 격자 간격 **{step:g}** · 서로 다른 값 {nuniq}개 · "
        f"범위 [{min(md_all):+.3f}, {max(md_all):+.3f}]")
    neg = sum(1 for v in md_all if v < 0)
    out(f"   음수 max_drop {neg}행 — 변형이 **원본보다 정확**했던 경우다(우리 규칙은 robust 로 본다)")
    out()

    cm = {}
    rows = []
    for r in df.itertuples():
        a, b = our_rule(float(r.max_drop)), org_state(r.is_robust)
        cm[(a, b)] = cm.get((a, b), 0) + 1
        rows.append((str(r.model_id), float(r.max_drop), a, b))
    states = ["robust", "spurious", "excluded"]
    out("   혼동표 (행 = 우리 규칙 · 열 = 주최측 is_robust)")
    out("   " + " " * 12 + "".join(f"{s:>11}" for s in states) + "       합")
    for a in states:
        rowtot = sum(cm.get((a, b), 0) for b in states)
        out(f"   {a:<12}" + "".join(f"{cm.get((a, b), 0):>11}" for b in states) + f"{rowtot:>10}")
    out("   " + "합".ljust(12) + "".join(f"{sum(cm.get((a, b), 0) for a in states):>11}" for b in states)
        + f"{len(df):>10}")
    agree = sum(cm.get((s, s), 0) for s in states)
    rate = agree / len(df)
    verdict = "같은 규칙" if rate >= 0.95 else ("근사" if rate >= 0.80 else "다른 규칙")
    out()
    out(f"   **일치율 {agree}/{len(df)} = {rate:.3f}** → 사전등록 판정: **{verdict}**")
    out("     (기준: ≥0.95 같은 규칙 · 0.80~0.95 근사 · <0.80 다른 규칙)")
    bad = [r for r in rows if r[2] != r[3]]
    if bad:
        out(f"   어긋난 {len(bad)}행 전부 (모델 · max_drop · 우리 → 주최측):")
        for m, v, a, b in sorted(bad, key=lambda x: x[1]):
            out(f"      {v:+.3f}  {a:<9} → {b:<9}  {m}")
    out()

    # 주최측 제외 행이 우리 중간대역과 같은 구간인가
    exc = [float(r.max_drop) for r in df.itertuples() if org_state(r.is_robust) == "excluded"]
    inc = [float(r.max_drop) for r in df.itertuples() if org_state(r.is_robust) != "excluded"]
    out(f"   주최측이 제외한 {len(exc)}행의 max_drop: "
        f"[{min(exc):.3f}, {max(exc):.3f}] · 값 {sorted(set(exc))}")
    out(f"   주최측이 라벨을 준 {len(inc)}행의 max_drop 범위: [{min(inc):.3f}, {max(inc):.3f}]")
    out(f"   → 우리 중간대역 ({LO}, {HI}) 과 비교: 제외 행이 전부 그 안에 드는가 = "
        f"{all(LO < v < HI for v in exc)}")
    out()

    # 임계 격자 쓸기 (탐색 — 이 값으로 우리 라벨을 다시 만들지 않는다)
    out("   [탐색] 주최측 라벨을 가장 잘 재현하는 임계쌍 — **우리 규칙을 이 값으로 바꾸지 않는다**")
    grid = [i / 100 for i in range(0, 51)]
    best = []
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            ok = sum(1 for r in df.itertuples()
                     if our_rule(float(r.max_drop), lo, hi) == org_state(r.is_robust))
            best.append((ok, lo, hi))
    best.sort(key=lambda x: (-x[0], x[2] - x[1]))
    top = best[0][0]
    ties = [(lo, hi) for ok, lo, hi in best if ok == top]
    out(f"      최고 일치 {top}/{len(df)} = {top / len(df):.3f} · 그 값을 내는 임계쌍 {len(ties)}개")
    out(f"      가장 좁은 것 5개: {ties[:5]}")
    out(f"      우리 규칙 ({LO}, {HI}) 의 일치 = {agree} → 최고와 차 {top - agree}행")
    out()

    # ---------- (1b) 분포 비교 ----------
    out("## (1b) 분포 비교 — **보조, 판정 없음** (문제집합·k·cap·디코딩이 전부 다르다)")
    out(f"   {'묶음':<58}{'n':>4}{'k':>3}{'해상도':>8}{'중앙':>8}{'spur%':>7}{'제외%':>7}")

    def line(name, md, k):
        vals = [v for v, *_ in md]
        st, _ = resolution(vals)
        lab = [our_rule(v) for v in vals]
        sp = sum(1 for x in lab if x == "spurious") / len(lab)
        ex = sum(1 for x in lab if x == "excluded") / len(lab)
        med = sorted(vals)[len(vals) // 2]
        out(f"   {name:<58}{len(vals):>4}{str(k):>3}{st:>8.3f}{med:>8.3f}{sp * 100:>7.1f}{ex * 100:>7.1f}")

    line("주최측 train-main-v2 (10샘플 · 5모델 · 36문제)", [(v,) for v in md_all], 10)
    for k, md in sorted(ours.items()):
        ks = {x[2] for x in md if x[2] is not None}
        line(k.replace("experiments/results/", ""), md, (sorted(ks)[0] if len(ks) == 1 else "?"))
    out()
    out("   읽는 법: `해상도` 는 max_drop 값들의 격자 간격이다. k샘플 정확도의 차이라면 1/k 의")
    out("   배수여야 한다 — 우리 k=8 은 0.125, k=4 는 0.25 다. 주최측은 그 둘과 다르고(위 숫자),")
    out("   그래서 **분포를 같은 축에 놓고 비교할 수 없다**. `spur%`·`제외%` 는 우리 규칙을 그 묶음에")
    out("   적용했을 때의 비율이고, 묶음마다 문제 난이도·모델·cap·디코딩이 달라 **원인 분해가 안 된다**.")
    out("   이 표의 용도는 '우리 라벨이 주최측과 같은 방향인가' 의 답이 아니라, 그 질문에 답할 수")
    out("   있는 데이터가 없다는 사실의 정량화다.")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
