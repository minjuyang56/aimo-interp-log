"""라벨 품질 게이트 — k=4 의 해상도와 'max over 섭동유형' 편향을 잰다. GPU 0. (지휘 #KCHK)

왜 게이트가 필요한가
  우리 라벨 규칙은 `max_drop = base_acc − min_over_types(var_acc)` 이고 `≥0.25 → spurious` 다.
  k=4 면 var_acc 의 해상도가 0.25 라서, **base_acc=1.0 인 문제는 변형 12샘플(3유형×4) 중
  하나만 틀려도 곧바로 spurious** 가 된다. 샘플당 실패확률 p 에 대해
      P(spurious | base_acc=1.0) = 1 − (1−p)^12      ← k=4, '유형 중 최솟값' 규칙
  p=0.05 에서 46% · p=0.10 에서 **72%** · p=0.15 에서 86% 다. 즉 **잡음만으로도 대부분이
  spurious 가 된다.** Olmo 선측정이 4/4 spurious 였던 것이 이 범위 안이다.

  게다가 `min over types`(= max drop)는 **세 유형 중 가장 나쁜 것만 본다** → 유형별 잡음을
  위로 끌어올리는 **최댓값 선택 편향**이 있다. 주최측 논문 정의는 '섭동의 **과반**' 이므로
  (지휘 #H22 에서 확인) 우리 max 규칙은 **더 시끄럽고 덜 충실하다**.

  좋은 소식: `var_acc` 가 **유형별로** 저장돼 있어서 규칙은 **사후에 다시 계산할 수 있다**.
  GPU 를 다시 쓸 필요가 있는 것은 **k(샘플 해상도)** 뿐이다.

보고 항목 (지휘 #KCHK)
  (1) 모델별 spurious 비율
  (2) `max_drop == 0.25` 이고 **결정 변형에서 실패가 정확히 1샘플**인 '경계 판정' 비율
  (3) 같은 문제들의 원본 `base_acc` 분포
  (4) [추가] max 규칙 vs 과반 규칙(논문 정의)으로 라벨을 다시 매기면 몇 건이 뒤집히나
  (5) [추가] 결정 변형 유형의 분포 — 잡음이면 유형이 고르게 퍼지고, 실재하면 한 유형에 몰린다

판정 규칙 (지휘 #KCHK)
  spurious 비율 > 70% **또는** 경계 판정이 spurious 의 절반 이상 → 남은 잡은 **k=8** 로 올린다.

사용:  .venv/Scripts/python.exe analysis/label_quality_gate.py [라벨.json ...]
       인자가 없으면 genlab/ 과 results/runpod/ 아래의 라벨 파일을 전부 찾는다.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "label_quality_gate.txt"
DROP_ROBUST, DROP_SPUR = 0.10, 0.25


def find_files():
    pats = [ROOT / "experiments" / "results" / "genlab",
            ROOT / "experiments" / "results" / "runpod"]
    out = []
    for base in pats:
        if base.is_dir():
            out += [p for p in base.rglob("*labels*.json")]
    return sorted(set(out))


def load(p):
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return []
    return [r for r in d if isinstance(r.get("label"), bool)]


def deciding(r):
    """라벨을 정한 변형 유형과 그 실패 상세."""
    va = r.get("var_acc") or {}
    vd = r.get("var_detail") or {}
    if not va:
        return None, {}
    kind = min(va, key=lambda k: (va[k] if va[k] is not None else 9))
    return kind, (vd.get(kind) or {})


def majority_label(r):
    """논문 정의에 가까운 규칙: **섭동의 과반**에서 base_acc 보다 떨어지면 spurious.
    (유형별 var_acc 가 저장돼 있어 사후 계산이 된다 — GPU 재실행이 필요 없다.)"""
    va = {k: v for k, v in (r.get("var_acc") or {}).items() if v is not None}
    if not va:
        return None
    base = r.get("base_acc")
    if base is None:
        return None
    broken = sum(1 for v in va.values() if (base - v) >= DROP_SPUR)
    return False if broken > len(va) / 2 else True


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("라벨 품질 게이트 — k 해상도와 'max over 섭동유형' 편향 (지휘 #KCHK)")
    out()
    out("산수: base_acc=1.0 이면 변형 12샘플(3유형×k4) 중 **하나만** 틀려도 max_drop=0.25 = spurious.")
    out("      P(spurious) = 1 − (1−p)^12 → p=0.05 46% · p=0.10 **72%** · p=0.15 86%.")
    out("      k=8 이면 임계가 '한 유형에서 2샘플 실패' 로 올라가 p=0.10 에서 46% 로 내려간다.")
    out()

    files = [Path(a) for a in sys.argv[1:]] or find_files()
    if not files:
        out("라벨 파일을 못 찾았다.")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    verdicts = []
    for f in files:
        rows = load(f)
        if not rows:
            continue
        sp = [r for r in rows if not r["label"]]
        caps = Counter(r.get("gen_cap") for r in rows)
        ks = Counter(r.get("gen_k") for r in rows)
        out(f"## {f.name}  ({f.parent.name})")
        out(f"   라벨 {len(rows)} · gen_cap {dict(caps)} · gen_k {dict(ks)}")
        out()
        # (1) spurious 비율
        rate = len(sp) / len(rows)
        out(f"   **(1) spurious 비율 {len(sp)}/{len(rows)} = {rate:.0%}**")
        # (2) 경계 판정
        boundary = []
        for r in sp:
            kind, det = deciding(r)
            fails = (det.get("n_wrong", 0) or 0) + (det.get("n_unfin", 0) or 0)
            if abs(float(r.get("max_drop", 0)) - 0.25) < 1e-6 and fails == 1:
                boundary.append((r.get("pidx"), kind))
        bshare = len(boundary) / max(1, len(sp))
        out(f"   **(2) 경계 판정(max_drop=0.25 & 결정변형 실패 1샘플) {len(boundary)}/{len(sp)} "
            f"= spurious 의 {bshare:.0%}**")
        # (3) base_acc 분포
        out(f"   (3) base_acc 분포 {dict(sorted(Counter(r.get('base_acc') for r in rows).items()))}")
        out(f"       max_drop 분포 {dict(sorted(Counter(round(float(r['max_drop']), 2) for r in rows).items()))}")
        # (4) 과반 규칙으로 다시 매기면
        flips = 0
        maj_sp = 0
        for r in rows:
            m = majority_label(r)
            if m is None:
                continue
            if not m:
                maj_sp += 1
            if bool(m) != bool(r["label"]):
                flips += 1
        out(f"   (4) **과반 규칙**(논문 정의)으로 다시 매기면 spurious {maj_sp}/{len(rows)} "
            f"= {maj_sp / len(rows):.0%} · 현재 라벨과 **{flips}건 뒤집힘** ({flips / len(rows):.0%})")
        # (5) 결정 변형 유형 분포
        kinds = Counter(deciding(r)[0] for r in sp)
        out(f"   (5) spurious 결정 변형 유형 {dict(kinds)} "
            f"— 잡음이면 고르게 퍼지고 실재하면 한 유형에 몰린다")
        out()
        gate = (rate > 0.70) or (bshare >= 0.50 and len(sp) > 0)
        out(f"   → 게이트: {'**k=8 로 올린다**' if gate else 'k=4 유지'} "
            f"(기준: spurious >70% 또는 경계 판정이 spurious 의 절반 이상)")
        out()
        verdicts.append((f.name, rate, bshare, gate))

    out("## 종합")
    out()
    out("| 파일 | spurious | 경계 판정 | 게이트 |")
    out("|---|---:|---:|---|")
    for name, rate, bshare, gate in verdicts:
        out(f"| {name} | {rate:.0%} | {bshare:.0%} | {'**k=8**' if gate else 'k=4 유지'} |")
    out()
    if any(g for _, _, _, g in verdicts):
        out("→ 하나라도 게이트에 걸리면 남은 잡(Olmo off40/80 · ③ · (b))은 **k=8** 로 돌린다.")
        out("  k 가 바뀐 라벨은 `gen_k` 메타로 구분된다(지휘 #KCHK).")
    out()
    out("## 규칙 교체는 GPU 없이 된다 (중요)")
    out()
    out("`var_acc` 가 **유형별로** 저장돼 있어서 max 규칙 → 과반 규칙 교체는 **사후 재계산**이면 끝이다.")
    out("GPU 를 다시 써야 하는 것은 **k(샘플 해상도)** 뿐이다. 그래서 (4) 의 뒤집힘 수가 크면,")
    out("k 를 올리기 전에 **규칙을 과반으로 바꾸는 것만으로** 라벨 품질이 개선될 수 있다 —")
    out("그 편이 공짜이고 주최측 정의에도 더 가깝다(지휘 #H22 에서 확인한 논문 정의가 '섭동의 과반').")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
