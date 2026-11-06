"""#TT-PERTURB 실패 원인 진단 — 정렬(pidx→라벨인덱스)을 맞춰 센다. 비용 $0.

왜 이 표가 결과보다 중요한가: 메커니즘(강제 즉답)은 작동했는데 신호가 없다. 그러면 '무엇이
흔들렸고 그 흔들림이 라벨과 어떤 관계인가' 를 숫자로 적어야 다음 라운드를 설계할 수 있다.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
from h21_uncertainty import problems
from ttp_eval import pidx_map, reindex, score_rows

REPORT = ROOT / "experiments/results/ttp_diag.txt"
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    rows_lab = problems()
    y = np.array([r["y"] for r in rows_lab], int)
    txt = [r["problem"] for r in rows_lab]
    p2i = pidx_map(txt)
    D = Path(sys.argv[1] if len(sys.argv) > 1
             else "experiments/results/runpod/ttperturb/kkm0qcksxp8z4y/out")
    out("#TT-PERTURB 진단 — 메커니즘은 작동했는데 왜 신호가 없나")
    out()
    out(f"{'모델':<28}{'원본답 ck간 불안정':>18}{'섭동답 일치율':>14}"
        f"{'robust 평균':>12}{'spur 평균':>11}{'차(부호)':>10}")
    diffs = []
    for f in sorted(D.glob("*_ttp_rows.json")):
        nm = f.name.replace("_ttp_rows.json", "")
        rows = json.loads(f.read_text(encoding="utf-8"))
        og = {p2i[r["pidx"]]: r for r in rows if r["kind"] == "orig" and r["pidx"] in p2i}
        unst = sum(1 for r in og.values()
                   if len({r["val"]["0"], r["val"]["256"], r["val"]["512"]}) > 1)
        sc = reindex(score_rows(rows, "512"), p2i)
        rb = [sc[i][0] for i in range(len(y)) if sc.get(i, (None,))[0] is not None and y[i] == 1]
        sp = [sc[i][0] for i in range(len(y)) if sc.get(i, (None,))[0] is not None and y[i] == 0]
        agree = []
        for i in range(len(y)):
            o = sc.get(i, (None, 0, None))[2]
            if o is None:
                continue
            for r in rows:
                if r["pidx"] in p2i and p2i[r["pidx"]] == i and r["kind"] != "orig" \
                        and r["val"]["512"] is not None:
                    agree.append(r["val"]["512"] == o)
        d = np.mean(rb) - np.mean(sp)
        diffs.append(d)
        out(f"{nm:<28}{unst:>10}/{len(og):<7}{np.mean(agree):>14.3f}"
            f"{np.mean(rb):>12.3f}{np.mean(sp):>11.3f}{d:>+10.3f}")
    out()
    out("원본답 ck간 불안정 = **섭동 없이** CoT 길이만 0/256/512 로 바꿨을 때 강제 즉답이 달라진 문제 수")
    out("섭동답 일치율      = 섭동의 강제 즉답이 원본 답과 같았던 비율 (1 이면 흔들림이 아예 안 생긴다)")
    out("robust/spur 평균   = score_spurious(갈린 섭동 비율) 의 군별 평균")
    out("차(부호)           = robust − spurious. **양수면 사전등록 부호와 반대**(robust 쪽이 더 흔들린다)")
    out()
    out(f"사전등록 부호와 반대인 모델 {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
