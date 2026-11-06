"""#LOOP R2 보완 — R1 의 **원본 답** 을 체크포인트별로 전부 적는다 (지휘 #LOOP R2 보완). 비용 $0.

**먼저 못 하는 것을 적는다: 정답률은 이 33문제에서 측정 불가다.**
`experiments/results/train-main-v2.parquet` 의 컬럼은 `model_id · reasoning_effort · problem ·
is_robust · max_drop` 뿐이고 **정답 컬럼이 없다**(#ORG-LABEL·H22 에서 이미 기록한 한계).
#ORG-LABEL 이 사 둔 cap 2048 greedy 풀이도 33/33 **전부 절단**이라 그 안에도 답이 없다.
→ '원본 정답률' 을 쓸 수 없으므로 **그 자리에 올 수 있는 측정 가능한 양 넷**을 적는다:

1. **추출 성공률** — 강제 즉답에서 정수 답이 나온 비율 (이 설계의 전제)
2. **답의 체크포인트 간 안정성** — 섭동 없이 CoT 길이만 바꿨을 때 답이 그대로인 비율.
   지휘가 의심한 '예산이 짧아 답이 안 정해졌다' 를 **직접** 재는 양이다.
3. **자기일관성(ck512 기준)** — ck512 답이 ck256·ck0 답과 같은 비율
4. **답의 쏠림** — 가장 흔한 답이 차지하는 비율(문제 텍스트의 숫자를 그대로 베끼는지 보는 가늠자)

사용: .venv/Scripts/python.exe analysis/ttp_orig_stats.py [out 디렉터리]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
from h21_uncertainty import problems
from ttp_eval import pidx_map

REPORT = ROOT / "experiments/results/ttp_orig_stats.txt"
NUM = re.compile(r"-?\d+")
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    rows_lab = problems()
    y = np.array([r["y"] for r in rows_lab], int)
    txt = [r["problem"] for r in rows_lab]
    p2i = pidx_map(txt)
    nums_in = [set(NUM.findall(t)) for t in txt]
    D = Path(sys.argv[1] if len(sys.argv) > 1
             else "experiments/results/runpod/ttperturb/kkm0qcksxp8z4y/out")

    out("#LOOP R2 보완 — R1 원본 답의 체크포인트별 통계 (공식 33문제)")
    out()
    out("**정답률은 측정 불가다**: train-main-v2 에 정답 컬럼이 없고(model_id·reasoning_effort·")
    out("problem·is_robust·max_drop 뿐) #ORG-LABEL 의 cap 2048 풀이도 33/33 전부 절단이라 답이 없다.")
    out("그래서 아래 넷으로 대신한다. **R1 은 체크포인트 0·256·512 만 있다** — 1024 칸은 예산 때문에")
    out("R2 재료와 맞바꿨다(그 결정은 결과 보기 전에 기록했다).")
    out()

    for f in sorted(D.glob("*_ttp_rows.json")):
        nm = f.name.replace("_ttp_rows.json", "")
        rows = json.loads(f.read_text(encoding="utf-8"))
        og = {p2i[r["pidx"]]: r for r in rows if r["kind"] == "orig" and r["pidx"] in p2i}
        cks = sorted(next(iter(og.values()))["val"].keys(), key=int)
        out(f"## {nm}")
        out(f"   {'ckpt':>6} {'추출':>8} {'문제숫자 베낌':>14} {'최빈답 비율':>12} "
            f"{'서로다른답':>11} {'robust 군':>10} {'spur 군':>9}")
        vals = {}
        for ck in cks:
            v = [og[i]["val"][ck] for i in range(len(y)) if i in og]
            vals[ck] = v
            got = [x for x in v if x is not None]
            copy = sum(1 for i, x in zip(range(len(y)), v) if x is not None and str(x) in nums_in[i])
            cnt = Counter(got)
            rb = [x for i, x in enumerate(v) if x is not None and y[i] == 1]
            sp = [x for i, x in enumerate(v) if x is not None and y[i] == 0]
            out(f"   {ck:>6} {len(got):>4}/{len(v):<3} {copy:>10}/{len(got):<3} "
                f"{cnt.most_common(1)[0][1] / max(1, len(got)):>12.3f} {len(cnt):>11} "
                f"{len(rb):>10} {len(sp):>9}")
        # 안정성
        keys = [k for k in cks]
        same_all = sum(1 for i in range(len(y))
                       if i in og and len({vals[k][i] for k in keys}) == 1)
        same_512_256 = sum(1 for i in range(len(y))
                           if i in og and vals["512"][i] == vals["256"][i])
        same_512_0 = sum(1 for i in range(len(y)) if i in og and vals["512"][i] == vals["0"][i])
        n = len(og)
        out(f"   안정성: 세 ckpt 전부 같은 답 **{same_all}/{n} = {same_all / n:.1%}** · "
            f"ck512==ck256 {same_512_256}/{n} · ck512==ck0 {same_512_0}/{n}")
        # 라벨군별 안정성 — 지휘가 의심한 '능력 부족이 섞였나' 를 가르는 칸
        st = np.array([1 if (i in og and len({vals[k][i] for k in keys}) == 1) else 0
                       for i in range(len(y))])
        out(f"   라벨군별 안정성: robust {st[y == 1].mean():.3f} · spurious {st[y == 0].mean():.3f} "
            f"(차 {st[y == 1].mean() - st[y == 0].mean():+.3f})")
        out()

    out("## 읽는 법")
    out("* '안정성' 이 낮으면 **예산이 짧아 답이 아직 안 정해졌다**는 뜻이다 — 지휘가 의심한 그 원인.")
    out("* '문제숫자 베낌' 은 즉답이 문제 본문의 숫자를 그대로 내놓은 비율이다. 높으면 모델이 풀지 않고")
    out("  베낀 것이라 섭동 비교가 '숫자 베끼기의 안정성' 을 재게 된다.")
    out("* '라벨군별 안정성' 의 차가 크면 안정성 자체가 라벨과 상관된다는 뜻이고, 그건 R2 의")
    out("  `base_acc`(원본 확신도) 항이 신호를 가질 가능성을 가리킨다.")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
