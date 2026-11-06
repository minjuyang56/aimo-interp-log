"""라벨이 생성 예산에 얼마나 달려 있나 — 같은 문제를 cap 16384 / 32768 로 만든 라벨을 맞댄다. GPU 0.

왜 이게 리포트에 들어가는가
  우리 라벨 규칙은 **미완주를 오답으로 센다**(개정 #37). 근거는 '주최측 라벨도 정해진 예산 안의
  정확도' 였다(FAQ 7·8). 그 선택의 대가가 얼마인지는 지금까지 **간접 증거**뿐이었다:
  24문제 판별 실험에서 cap 8192→16384 로 올리니 두 라벨 정의(`label` vs `label_fin`)의 일치율이
  56%→90% 가 됐다는 것. 그건 '정의끼리' 의 일치율이고 '같은 정의를 다른 예산으로' 잰 게 아니다.

  (b)(지휘 #CAP-B)로 **DeepSeek 0–79 를 cap 16384(Kaggle)와 cap 32768(RunPod) 두 번** 만든다.
  같은 문제·같은 모델·같은 변형 규칙·같은 k 에서 **cap 만 다르다** → 라벨 뒤집힘을 직접 센다.
  이 숫자가 크면 '우리 라벨은 문제의 성질이 아니라 예산의 함수' 라는 뜻이고, 그건 h20 계열이
  죽은 이유(포화)와 같은 병이 **라벨 쪽에도** 있다는 증거가 된다.

무엇을 내는가
  (a) 라벨 뒤집힘 수와 방향 (robust→spurious / spurious→robust)
  (b) 미완주율 변화 · spurious 결정 변형의 미완주 비중 변화
  (c) max_drop 의 변화 분포
  (d) 두 cap 에서 **둘 다 라벨이 있는** 문제만 모은 일치율 (= 라벨의 예산 의존도)

사용:
  .venv/Scripts/python.exe analysis/label_cap_dependence.py \
      --a experiments/results/genlab/<16384 라벨>.json --b experiments/results/genlab/<32768 라벨>.json
  (인자 없이 돌리면 genlab/ 에서 gen_cap 이 다른 같은 모델 파일을 자동으로 짝짓는다)
"""

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENLAB = ROOT / "experiments" / "results" / "genlab"
OUT = ROOT / "experiments" / "results" / "label_cap_dependence.txt"


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return [r for r in d if isinstance(r.get("label"), bool)]


def cap_of(rows):
    caps = {r.get("gen_cap") for r in rows if r.get("gen_cap")}
    return (caps.pop() if len(caps) == 1 else f"혼합{sorted(caps)}") if caps else "미기록"


def unfin_share(rows):
    """spurious 를 결정한 변형의 실패 중 미완주 비중."""
    cu = cw = 0
    for r in rows:
        if r["label"]:
            continue
        va, vd = r.get("var_acc") or {}, r.get("var_detail") or {}
        if not va or not vd:
            continue
        worst = min(va, key=lambda k: va[k])
        v = vd.get(worst) or {}
        cu += v.get("n_unfin", 0)
        cw += v.get("n_wrong", 0)
    return cu, cw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="낮은 cap 라벨 파일")
    ap.add_argument("--b", help="높은 cap 라벨 파일")
    a = ap.parse_args()

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    if not (a.a and a.b):
        cands = sorted(GENLAB.glob("*_labels*.json"))
        out("## 짝지을 파일 후보 (gen_cap 기준)")
        for p in cands:
            try:
                rows = load(p)
            except Exception:
                continue
            if rows:
                out(f"  {p.name}  라벨 {len(rows):3d}  gen_cap {cap_of(rows)}")
        out()
        out("→ --a(낮은 cap) --b(높은 cap) 로 두 파일을 지정해 다시 돌릴 것.")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    A, B = load(a.a), load(a.b)
    ka = {r["problem"]: r for r in A}
    kb = {r["problem"]: r for r in B}
    both = [p for p in ka if p in kb]
    out("라벨의 생성 예산 의존도 — 같은 문제를 두 cap 으로")
    out()
    out(f"A {Path(a.a).name}: 라벨 {len(A)} · gen_cap {cap_of(A)}")
    out(f"B {Path(a.b).name}: 라벨 {len(B)} · gen_cap {cap_of(B)}")
    out(f"양쪽에 라벨이 있는 문제 **{len(both)}건** "
        f"(A 에만 {len(A) - len(both)} · B 에만 {len(B) - len(both)})")
    out()
    if not both:
        out("겹치는 문제가 없다 — 범위를 확인할 것.")
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    flips = [(p, ka[p]["label"], kb[p]["label"]) for p in both if ka[p]["label"] != kb[p]["label"]]
    r2s = sum(1 for _, x, y in flips if x and not y)
    s2r = sum(1 for _, x, y in flips if not x and y)
    out(f"## (d) 일치율 **{1 - len(flips) / len(both):.0%}** · 뒤집힘 {len(flips)}/{len(both)}건")
    out(f"    robust→spurious {r2s} · spurious→robust {s2r}")
    out()
    out("## (b) 미완주")
    for tag, rows in (("A", A), ("B", B)):
        tu = tn = 0
        for r in rows:
            for v in (r.get("var_detail") or {}).values():
                tu += v.get("n_unfin", 0)
                tn += v.get("n_fin", 0) + v.get("n_unfin", 0)
        cu, cw = unfin_share(rows)
        out(f"    {tag}: 변형 샘플 미완주 {tu}/{tn} ({tu / max(1, tn):.0%}) · "
            f"spurious 결정 변형의 미완주 비중 **{cu / max(1, cu + cw):.0%}** (미완주 {cu} · 오답 {cw})")
    out()
    out("## (c) max_drop 변화 (B − A), 양쪽에 있는 문제만")
    delta = collections.Counter()
    for p in both:
        d = round(float(kb[p]["max_drop"]) - float(ka[p]["max_drop"]), 2)
        delta[d] += 1
    out(f"    {dict(sorted(delta.items()))}")
    same = sum(v for k, v in delta.items() if abs(k) < 1e-9)
    out(f"    max_drop 가 그대로인 문제 {same}/{len(both)} ({same / len(both):.0%})")
    out()
    out("## (a) 뒤집힌 문제 상세")
    for p, x, y in flips:
        out(f"    pidx {ka[p].get('pidx')} · {'robust' if x else 'spurious'} → "
            f"{'robust' if y else 'spurious'} · max_drop {ka[p]['max_drop']} → {kb[p]['max_drop']}")
        out(f"      A var_acc {ka[p].get('var_acc')}")
        out(f"      B var_acc {kb[p].get('var_acc')}")
    out()
    out("## 읽는 법")
    out("일치율이 높으면(>90%) 우리 라벨은 예산에 둔감하다는 뜻이고, 개정 #37('미완주 = 오답')의")
    out("대가가 작다는 증거다. 낮으면 **라벨이 문제의 성질이 아니라 예산의 함수**이고, h20 계열을")
    out("죽인 포화와 같은 병이 라벨 쪽에도 있다는 뜻이다 — 그러면 #77 프로브의 학습 대상 자체가")
    out("흔들리므로 리포트에 한계로 명시해야 한다.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
