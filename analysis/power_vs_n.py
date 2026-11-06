"""#KAGGLE-0 사전 계산 — **n 을 얼마까지 늘려야 검정력 천장이 뚫리나.** $0.

왜 이걸 먼저 하나: 라벨을 더 만드는 계획의 가치는 '행이 몇 개 더 생기나' 가 아니라 **'그 행 수에서
신호를 되찾을 수 있나'** 로 정해진다. #LABEL-AUDIT 이 n=33 에서 d=400 일 때 **20σ 로 심은 신호도
AUC 0.696** 이 천장임을 쟀다. 같은 경로에 **n 만 바꿔** 심어 보면 목표 n 이 나온다.
그 숫자 없이 GPU 를 쓰면 '몇 주 걸린다' 는 견적이 무의미하다.

방법: `probe_final_eval.make_plan` / `nested_scores` **실제 경로**에 한 차원 신호를 심고
n ∈ {33, 60, 100, 150, 200, 300} · d ∈ {400, 4096} · shift ∈ {3σ, 8σ} 로 AUC 를 본다.
라벨 비율은 공식 33 과 같게(robust 0.606) 유지한다. 반복 5회 평균(시드 고정).

사용: .venv/Scripts/python.exe analysis/power_vs_n.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from probe_final_eval import auc_of, make_plan, nested_scores

REPORT = ROOT / "experiments/results/power_vs_n.txt"
NS = (33, 60, 100, 150, 200, 300)
RATE = 0.606
REPEATS = 5
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def one(n, d, shift, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, int)
    y[: int(round(RATE * n))] = 1
    rng.shuffle(y)
    X = rng.normal(0, 1, (n, d))
    X[:, 0] += shift * y
    plan = make_plan({0: X}, y, [0])
    if plan is None:
        return float("nan")
    s = nested_scores(y, [0], plan)
    s = s[0] if isinstance(s, tuple) else s
    return auc_of(y, s)


def main():
    out("#KAGGLE-0 사전 계산 — 행 수 n 에 따른 검정력 (실제 nested CV 경로에 신호를 심는다)")
    out(f"   라벨 비율 robust {RATE} 고정 · PCA 24 · 층·C 폴드 내부 CV · 반복 {REPEATS}회 평균")
    out("   읽는 법: 심은 신호를 **되찾은** AUC 다. 0.75 를 넘지 못하면 그 n 에서는 그 크기의")
    out("   신호도 우연과 구분되지 않는다(이 프로젝트의 중심 문장과 같은 기준).")
    out()
    for d in (400, 4096):
        out(f"## 특징 차원 d = {d}")
        out(f"   {'n':>6}" + "".join(f"{f'{sh}σ':>10}" for sh in (3.0, 8.0)))
        for n in NS:
            cells = []
            for sh in (3.0, 8.0):
                vals = [one(n, d, sh, 1000 + 7 * k) for k in range(REPEATS)]
                cells.append(float(np.nanmean(vals)))
            star = "  ← 공식 33" if n == 33 else ""
            out(f"   {n:>6}" + "".join(f"{c:>10.3f}" for c in cells) + star)
        out()
    out("## 읽고 나서 정할 것")
    out("   목표 n = 심은 8σ 신호가 0.75 를 넘는 가장 작은 n. 그 값이 라벨 생성 계획의 N 을 정한다.")
    out("   (8σ 를 기준으로 삼는 이유: 3σ 는 d=400 에서도 n=300 까지 안 올라올 가능성이 크고,")
    out("    20σ 는 현실적인 효과 크기가 아니다. 8σ 는 '있으면 반드시 보여야 하는' 크기다.)")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
