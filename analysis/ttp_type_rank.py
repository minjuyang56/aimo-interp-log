"""R2 섭동 부분집합 선정 — 지휘 '정답률 하락이 큰 2~3종만'. R1 데이터로 센다. 비용 $0.

**이 선택은 R1 결과를 본 뒤에 한다** — 지휘가 그 규칙을 지정했으므로 따르지만, 사후 선택임을
사전등록에 명시한다. 유형별로 재는 양: 섭동의 강제 즉답이 원본 답과 **달라진 비율**(= 하락 대리).
커버리지(그 유형이 붙는 문제 수)도 같이 본다 — 커버리지가 낮으면 K 가 조용히 줄어든다.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
from h21_uncertainty import problems
from ttp_eval import pidx_map

REPORT = ROOT / "experiments/results/ttp_type_rank.txt"
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    rows_lab = problems()
    y = np.array([r["y"] for r in rows_lab], int)
    p2i = pidx_map([r["problem"] for r in rows_lab])
    D = Path(sys.argv[1] if len(sys.argv) > 1
             else "experiments/results/runpod/ttperturb/kkm0qcksxp8z4y/out")
    agg = defaultdict(lambda: {"n": 0, "flip": 0, "rb": [], "sp": []})
    for f in sorted(D.glob("*_ttp_rows.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        og = {p2i[r["pidx"]]: r["val"]["512"] for r in rows
              if r["kind"] == "orig" and r["pidx"] in p2i}
        for r in rows:
            if r["kind"] == "orig" or r["pidx"] not in p2i:
                continue
            i = p2i[r["pidx"]]
            o, v = og.get(i), r["val"]["512"]
            if o is None or v is None:
                continue
            a = agg[r["kind"]]
            a["n"] += 1
            fl = int(v != o)
            a["flip"] += fl
            (a["rb"] if y[i] == 1 else a["sp"]).append(fl)
    out("#LOOP R2 섭동 유형 순위 — '답이 달라진 비율'(하락 대리), CoT 512 · 4모델 합산")
    out(f"   {'유형':<12}{'관측':>7}{'달라짐':>9}{'비율':>8}{'robust':>9}{'spur':>8}{'차(부호)':>10}")
    rank = []
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["flip"] / max(1, kv[1]["n"])):
        rate = a["flip"] / max(1, a["n"])
        rb = np.mean(a["rb"]) if a["rb"] else float("nan")
        sp = np.mean(a["sp"]) if a["sp"] else float("nan")
        rank.append((k, rate, a["n"]))
        out(f"   {k:<12}{a['n']:>7}{a['flip']:>9}{rate:>8.3f}{rb:>9.3f}{sp:>8.3f}{rb - sp:>+10.3f}")
    out()
    top = [k for k, _, _ in rank[:3]]
    out(f"   상위 3종: {top}")
    out("   '차(부호)' 가 양수면 그 유형에서 robust 쪽이 더 많이 흔들린다(사전등록과 반대).")
    out("   비율이 높아도 차가 0 근처면 그 유형은 **라벨과 무관하게** 흔든다는 뜻이다.")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
