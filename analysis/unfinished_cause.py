"""자체 라벨의 미완주 원인 분해 — spurious 가 '틀렸다' 인가 '예산을 넘겼다' 인가. GPU 0. (지휘 #78-2)

왜: cap 8192 에서는 spurious 8건 중 7건이 라벨 정의를 바꾸면 뒤집혔다(미완주 91%). cap 16384 로 올려
24문제 판별 실험에서 두 정의 일치율이 56%→90% 가 됐고 본실행을 그 설정으로 돌렸다. 그 본실행 88문제에서
**실제로 얼마나 남았는지**를 여기서 잰다. 이 숫자가 Olmo·DeepSeek 커널 설정을 바꿀지 말지의 근거다.

세 가지를 구분해서 센다:
  (a) 전체 변형 샘플의 미완주율 — 설정(cap)이 전반적으로 충분한가
  (b) **spurious 라벨을 결정한 변형**(var_acc 가 최저인 변형)의 실패 원인 — 라벨이 예산에 끌려갔는가
  (c) 사전등록 민감도 부분집합(n_wrong >= n_unfin)의 크기 — 그 통제가 검정력을 갖는가

사용:  .venv/Scripts/python.exe analysis/unfinished_cause.py
"""

import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "experiments" / "results" / "genlab"
OUT = ROOT / "experiments" / "results" / "unfinished_cause.txt"


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    for f in sorted(MERGED.glob("*_labels_merged.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        lab = [r for r in d if isinstance(r.get("label"), bool)]
        if not lab:
            continue
        out(f"## {f.stem} — 라벨 {len(lab)}건 "
            f"(robust {sum(1 for r in lab if r['label'])} / spurious {sum(1 for r in lab if not r['label'])})")
        out()
        tu = tw = tn = 0
        for r in lab:
            for v in (r.get("var_detail") or {}).values():
                tu += v.get("n_unfin", 0)
                tw += v.get("n_wrong", 0)
                tn += v.get("n_fin", 0) + v.get("n_unfin", 0)
        out(f"(a) 변형 샘플 {tn}개 중 미완주 {tu} ({tu / max(1, tn):.0%}) · 완주하고 오답 {tw} ({tw / max(1, tn):.0%})")
        fin = statistics.mean(r["n_fin"] for r in lab)
        out(f"    원본 쪽: 완주 평균 {fin:.2f}/4 → 미완주율 {1 - fin / 4:.0%}")
        sp = [r for r in lab if not r["label"]]
        cu = cw = dom = 0
        for r in sp:
            worst = min(r["var_detail"], key=lambda k: r["var_acc"][k])
            v = r["var_detail"][worst]
            cu += v.get("n_unfin", 0)
            cw += v.get("n_wrong", 0)
            dom += 1 if v.get("n_unfin", 0) >= v.get("n_wrong", 0) else 0
        out(f"(b) spurious {len(sp)}건의 **결정 변형**: 미완주 {cu} · 오답 {cw} → 미완주 비중 "
            f"**{cu / max(1, cu + cw):.0%}** · 미완주 우세 {dom}/{len(sp)}건")
        out(f"(c) 사전등록 민감도 부분집합(오답 우세 spurious)의 크기 = **{len(sp) - dom}건** "
            f"→ 이 수가 10 미만이면 그 통제는 검정력이 없다")
        flip = [r for r in lab if r.get("label_fin") is not None and bool(r["label"]) != bool(r["label_fin"])]
        out(f"    두 라벨 정의 불일치 {len(flip)}/{len(lab)} ({len(flip) / len(lab):.0%})")
        out(f"    max_drop 분포 {dict(sorted(collections.Counter(round(r['max_drop'], 2) for r in lab).items()))}")
        out(f"    base_acc 분포 {dict(sorted(collections.Counter(r['base_acc'] for r in lab).items()))} "
            f"· 문제풀 {dict(collections.Counter(r['level'] for r in lab))}")
        out()
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
