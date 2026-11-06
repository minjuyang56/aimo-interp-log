"""#LOOP R5 평가 — 방향은 자체 screen 행에서(정답률 타깃), 시험은 공식 33 에서 1차원 한 번. $0.

사전등록(CLAUDE.md '#LOOP R5') 그대로:
* 학습: 모델별 `StandardScaler → Ridge(alpha 는 학습 행 안 5-fold CV)` 로 **`base_acc`** 를 회귀
* 시험: 그 scaler·그 w 로 공식 33 특징을 투영한 1차원 점수. **부호 큼 = robust** (뒤집지 않는다)
* 임계는 문제 단위 **LOO 사전확률 교정** · 셔플 null **1000** · 길이 대조군·상수 병기
* 주지표 = **층 L//2 · Ridge(PCA 없음) · 모델평균 정확도**. 나머지 층과 PCA24 판은 사전지정 민감도
* 채택: 모델평균 정확도 ≥ **0.686** AND 자기 null 95% 상단 초과

**먼저 찍는 진단**: 학습 쪽 교차검증 상관. 방향이 in-domain 에서도 `base_acc` 를 못 맞히면
전이 시험은 공허하다 — 그 경우 '전이 실패' 가 아니라 '방향이 안 잡혔다' 로 읽어야 한다.

사용: .venv/Scripts/python.exe analysis/r5_eval.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
from h21_uncertainty import problems
from ttp_eval import auc, loo_acc, null_band

P = ROOT / "experiments/results/probe"
REPORT = ROOT / "experiments/results/r5_report.txt"
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
MODELS = {"Skywork-OR1-Math-7B": 14, "Olmo-3-7B-Think": 16, "DeepSeek-R1-0528-Qwen3-8B": 18}
REPS = 1000
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def fit_direction(X, t, npc=None, seed=0):
    """학습 행에서만 방향을 잡는다. (변환 함수, 고른 alpha, 학습 CV 상관)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    st = StandardScaler().fit(X)
    Z = st.transform(X)
    pc = None
    if npc:
        pc = PCA(n_components=min(npc, min(Z.shape) - 1), random_state=seed).fit(Z)
        Z = pc.transform(Z)
    # alpha 는 학습 행 안 5-fold CV 로만 (공식 33 미사용)
    kf = list(KFold(5, shuffle=True, random_state=seed).split(Z))
    best, best_a = -2.0, ALPHAS[0]
    for a in ALPHAS:
        oof = np.zeros(len(t))
        for tr, te in kf:
            oof[te] = Ridge(alpha=a).fit(Z[tr], t[tr]).predict(Z[te])
        r = np.corrcoef(oof, t)[0, 1] if np.std(oof) > 1e-12 else 0.0
        if np.isnan(r):
            r = 0.0
        if r > best:
            best, best_a = r, a
    m = Ridge(alpha=best_a).fit(Z, t)

    def tf(Xq):
        Zq = st.transform(Xq)
        if pc is not None:
            Zq = pc.transform(Zq)
        return m.predict(Zq)
    return tf, best_a, best


def main():
    rows_lab = problems()
    y = np.array([r["y"] for r in rows_lab], int)
    txt = [r["problem"] for r in rows_lab]
    lw = np.array([math.log(max(1, len(t.split()))) for t in txt], float)
    n = len(y)
    al, _ = loo_acc(lw, y)

    out("#LOOP R5 — 방향은 33행 밖(정답률), 시험은 공식 33 에서 1차원 한 번 · $0")
    out(f"   공식 33: robust {int(y.sum())}/{n} · 상수 {y.mean():.4f} · 길이 h15C(LOO) {al:.4f}")
    out(f"   채택 기준: 모델평균 정확도 ≥ **0.686** AND 자기 null 95% 상단 초과")
    out()

    missing = [m for m in MODELS if not (P / f"{m}_r5screen.npz").exists()]
    if missing:
        out(f"!! 특징 파일 없음: {missing}")
        REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
        return

    out("## 학습 쪽 진단 — 방향이 in-domain 에서 `base_acc` 를 맞히나 (이게 0 이면 전이 시험은 공허하다)")
    out(f"   {'모델':<28}{'행':>4}{'타깃 sd':>9}{'alpha':>9}{'학습 CV 상관':>13}")
    direc = {}
    for m, lay in MODELS.items():
        d = np.load(P / f"{m}_r5screen.npz", allow_pickle=True)
        X, t = np.asarray(d[f"h{lay}"], float), np.asarray(d["base_acc"], float)
        tf, a, r = fit_direction(X, t)
        direc[m] = (tf, lay)
        out(f"   {m[:26]:<28}{len(t):>4}{t.std():>9.3f}{a:>9.0f}{r:>13.3f}")
    out()

    out("## 주지표 — 공식 33 전이 (층 L//2 · Ridge · PCA 없음)")
    out(f"   {'모델':<28}{'정확도':>8}{'AUC':>8}{'null상단':>10}{'점수 sd':>9}{'True비율':>9}")
    accs, aucs, nuls, scores = [], [], [], {}
    for m, (tf, lay) in direc.items():
        do = np.load(P / f"{m}_official.npz", allow_pickle=True)
        Xo = np.asarray(do[f"h{lay}"], float)
        s = np.asarray(tf(Xo), float)
        scores[m] = s
        # 부호: 점수 큼 = robust → spur = −s
        a_, pred = loo_acc(-s, y)
        u = auc(-s, y)
        (_, ha), _ = null_band(-s, y, REPS, 81)
        accs.append(a_); aucs.append(u); nuls.append(ha)
        out(f"   {m[:26]:<28}{a_:>8.3f}{u:>8.3f}{ha:>10.3f}{s.std():>9.4f}{pred.mean():>9.3f}")
    ma, mu, mn = float(np.mean(accs)), float(np.mean(aucs)), float(np.mean(nuls))
    out(f"   {'**모델평균**':<28}{ma:>8.3f}{mu:>8.3f}{mn:>10.3f}")
    out()

    out("## 사전등록 판정")
    thr = max(al, float(y.mean())) + 0.05
    ok1, ok2 = ma >= thr, ma > mn
    out(f"   기준1 모델평균 {ma:.4f} ≥ {thr:.4f} (= max(길이 {al:.3f}, 상수 {y.mean():.3f}) + 0.05) → "
        f"{'통과' if ok1 else '**미달**'}")
    out(f"   기준2 모델평균 {ma:.4f} > null 상단 {mn:.3f} → {'통과' if ok2 else '**미달**'}")
    out(f"   → **{'채택' if (ok1 and ok2) else '기각'}**")
    out(f"   상수 대비 {ma - float(y.mean()):+.4f} · 길이 대비 {ma - al:+.4f}")
    out()

    out("## (c) 모델 간 부호 일치 — 진짜 신호면 모델을 넘어 반복돼야 한다")
    from scipy.stats import spearmanr
    ms = list(scores)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            r, p = spearmanr(scores[ms[i]], scores[ms[j]])
            out(f"   ρ({ms[i][:12]}, {ms[j][:12]}) = {r:+.3f} (p={p:.3f})")
    side = ["위" if u > 0.5 else "아래" for u in aucs]
    out(f"   AUC 가 0.5 의 같은 쪽인가: {dict(zip([m[:12] for m in ms], side))} → "
        f"{'일치' if len(set(side)) == 1 else '**갈린다**'}")
    out()

    out("## 사전지정 민감도 (판정에 쓰지 않는다)")
    out(f"   {'설정':<24}" + "".join(f"{m[:11]:>13}" for m in MODELS) + f"{'모델평균':>10}")
    for nm, npc, lays in (("PCA24 · L//2", 24, None),
                          ("층 L//4", None, "q1"), ("층 3L//4", None, "q3"), ("층 L", None, "q4")):
        row, acc2 = [], []
        for m, lay in MODELS.items():
            d = np.load(P / f"{m}_r5screen.npz", allow_pickle=True)
            keys = sorted(int(k[1:]) for k in d.files if k.startswith("h") and k[1:].isdigit())
            use = lay if lays is None else {"q1": keys[0], "q3": keys[2], "q4": keys[3]}[lays]
            X, t = np.asarray(d[f"h{use}"], float), np.asarray(d["base_acc"], float)
            tf, _, _ = fit_direction(X, t, npc=npc)
            do = np.load(P / f"{m}_official.npz", allow_pickle=True)
            s = np.asarray(tf(np.asarray(do[f"h{use}"], float)), float)
            a_, _ = loo_acc(-s, y)
            row.append(a_); acc2.append(a_)
        out(f"   {nm:<24}" + "".join(f"{v:>13.3f}" for v in row) + f"{np.mean(acc2):>10.3f}")
    out()
    out("## 한계 (사전등록에 적은 그대로)")
    out("   * 타깃이 치우쳐 있다(`base_acc`=1.0 이 28/33/39개) → Ridge 수축으로 점수 분산이 작아질 수 있다")
    out("   * 학습은 AIME·MATH L5(단어 중앙 31) · 시험은 공식 올림피아드(57) — 전이 실패를")
    out("     '신호 없음' 과 '문제집합 차이' 로 가를 수 없다")
    out("   * 정답률과 robust 는 다른 양이다 — 상관될 이유는 있지만 같은 양이 아니다")
    out("   * Qwen3.5 는 자체 라벨 0개라 빠졌다(3모델)")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
