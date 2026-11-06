"""#KAGGLE-0 사전 계산 2 — **파이프라인이 병목인가, 행 수가 병목인가.** $0.

`power_vs_n.py` 가 보여준 것: d=4096 에서 8σ 로 심은 1차원 신호를 `StandardScaler → PCA 24 →
로지스틱` 경로가 **n=300 에서도 못 찾는다**(AUC 0.508). 그러면 라벨을 더 만드는 계획의 가치가
의심된다 — 행이 늘어도 이 파이프라인은 못 본다는 뜻이기 때문이다.

그래서 **같은 합성 데이터에 파이프라인만 바꿔** 본다. 핵심 의심: 표준화가 분산 스파이크를 없애고
PCA 는 분산으로 성분을 고르므로, 4096 차원 중 신호 1개가 24성분에 들어갈 이유가 없다.
대안은 **폴드 학습분에서 단변량 연관으로 차원을 고르는 것**(y 를 보지만 **폴드 안에서만** 보므로
누출이 아니다 — PCA 를 폴드 안에서 적합하는 것과 같은 자격이다).

비교 세 가지 (전부 같은 외부 폴드 · 같은 합성 데이터):
  A  StandardScaler → PCA 24 → 로지스틱         (우리가 #77·#ORG-TRAIN·#CMP4·R5 에서 쓴 경로)
  B  StandardScaler → 로지스틱 (전 차원, L2)      (PCA 없음)
  S  StandardScaler → **폴드 안 단변량 상위 24 차원** → 로지스틱

사용: .venv/Scripts/python.exe analysis/power_select.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from probe_final_eval import auc_of

REPORT = ROOT / "experiments/results/power_select.txt"
RATE, REPEATS, CS = 0.606, 5, (0.003, 0.01, 0.03, 0.1, 0.3)
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def oof(X, y, mode, npc=24, seed=0):
    sc = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        st = StandardScaler().fit(X[tr])
        Ztr, Zte = st.transform(X[tr]), st.transform(X[te])
        if mode == "A":
            pc = PCA(n_components=min(npc, len(tr) - 1), random_state=seed).fit(Ztr)
            Ztr, Zte = pc.transform(Ztr), pc.transform(Zte)
        elif mode == "S":
            # **폴드 학습분에서만** 단변량 상관으로 차원 선택 (테스트 행을 보지 않는다)
            yc = y[tr] - y[tr].mean()
            num = Ztr.T @ yc
            den = np.linalg.norm(Ztr, axis=0) * np.linalg.norm(yc) + 1e-12
            r = np.abs(num / den)
            keep = np.argsort(-r)[:npc]
            Ztr, Zte = Ztr[:, keep], Zte[:, keep]
        best, bs = -1.0, None
        for C in CS:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Ztr, y[tr])
            s = m.decision_function(Ztr)
            a = auc_of(y[tr], s)
            if a > best:
                best, bs = a, m
        sc[te] = bs.decision_function(Zte)
    return auc_of(y, sc)


def main():
    out("#KAGGLE-0 사전 계산 2 — 파이프라인이 병목인가 (심은 8σ 1차원 신호, d=4096)")
    out("   A = StandardScaler→PCA24→LR (우리 경로) · B = PCA 없음 · S = 폴드 안 단변량 상위 24")
    out(f"   라벨 비율 robust {RATE} · 반복 {REPEATS}회 평균 · 0.75 를 넘으면 '되찾았다'")
    out()
    out(f"   {'n':>6}{'A (PCA24)':>12}{'B (전차원)':>12}{'S (선별24)':>12}")
    for n in (33, 60, 100, 200):
        res = {m: [] for m in "ABS"}
        for k in range(REPEATS):
            rng = np.random.default_rng(2000 + 11 * k)
            y = np.zeros(n, int)
            y[: int(round(RATE * n))] = 1
            rng.shuffle(y)
            X = rng.normal(0, 1, (n, 4096))
            X[:, 0] += 8.0 * y
            for m in "ABS":
                res[m].append(oof(X, y, m, seed=k))
        out(f"   {n:>6}" + "".join(f"{np.mean(res[m]):>12.3f}" for m in "ABS"))
    out()
    out("## 해석 규칙 (결과 보기 전 정해 둔다)")
    out("   * S 가 n=33 에서도 0.75 를 넘으면 → **병목은 파이프라인이고 $0 으로 고쳐진다.**")
    out("     그러면 라벨을 더 만들기 전에 기존 데이터를 선별 경로로 다시 봐야 한다.")
    out("   * S 도 n 이 커질 때만 넘으면 → **행 수가 병목이고 Kaggle 계획에 가치가 있다.**")
    out("   * S 가 어디서도 못 넘으면 → 1차원 8σ 조차 이 설계로는 못 찾는다는 뜻이고,")
    out("     라벨을 늘려도 프로브 노선은 닫힌다(리포트의 검정력 절로 보낸다).")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
