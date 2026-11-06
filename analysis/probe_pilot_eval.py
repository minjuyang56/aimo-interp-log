"""프로브 파일럿 평가 — 자체 라벨 88개로 모델별 선형 프로브가 되는가. GPU 0. (지휘 #75-2)

설계(측정 전 고정)
  특징: 문제 텍스트 forward 1회의 hidden state, 문제 토큰 구간 평균 풀링. 층은 **폴드 내부 CV 로만** 고른다.
  모델: 로지스틱 회귀(L2), `class_weight="balanced"` — 생성 라벨은 robust 로 쏠리고(62:26) 검증셋은 균형이라
        쏠린 학습은 바로 손해다(지휘 #71-3).
  평가: **문제 단위 GroupKFold**(여기서는 문제가 행마다 하나뿐이라 StratifiedKFold 와 같다 — 모델이 늘면 의미가 생긴다).
  지표: **AUC 와 balanced accuracy 를 둘 다**(지휘 #71-3). AUC 는 순위만 보고 균형에 둔감하고,
        balanced accuracy 는 실제 bool 판정에 대응한다.
  검문소: 라벨 **셔플 null 200** · **길이 대조군**(log 단어수 단독 로지스틱) · `label` vs `label_fin` 둘 다 ·
        **외부 테스트 = 공식 train-main-v2 33문제**(학습에 절대 쓰지 않는다).

왜 또 프로브인가: H16 에서 같은 구조를 공식 33문제로 돌렸을 때 AUC 0.531(null 안)이었고, 내부 CV 가 최강 규제를
골랐다 = 33문제로 3,584차원은 일반화 불가. 라벨이 88개로 늘었으니 **같은 방법에 데이터만 바꿔** 다시 본다.

사용:  .venv/Scripts/python.exe analysis/probe_pilot_eval.py [--null 200]
"""

import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "experiments" / "results" / "probe"
CS = (0.003, 0.01, 0.03, 0.1, 0.3)


NPC = 24          # PCA 성분 수. 88문제에 3,584차원은 과하고(H16 에서 확인), 적합 속도도 못 맞춘다.


def fit_predict(Xtr, ytr, Xte, C):
    """표준화 → **학습분으로만 적합한 PCA** → 로지스틱(class_weight balanced)."""
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    if Xtr.shape[1] > NPC:
        pca = PCA(n_components=min(NPC, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=0).fit(Ztr)
        Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    m.fit(Ztr, ytr)
    return m.decision_function(Zte), m, sc


def nested_cv(feats, y, layers, seed=0, folds=5):
    """바깥 폴드마다 **안쪽 CV 로 층·C 를 고른다**. 외부 라벨을 보지 않는다."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    score = np.zeros(len(y))
    picks = []
    for tr, te in skf.split(feats[layers[0]], y):
        best, bestkey = -1, None
        inner = StratifiedKFold(min(4, int(min(np.bincount(y[tr])))), shuffle=True, random_state=seed)
        for l in layers:
            for C in CS:
                s = np.zeros(len(tr))
                for itr, ite in inner.split(feats[l][tr], y[tr]):
                    s[ite], _, _ = fit_predict(feats[l][tr][itr], y[tr][itr], feats[l][tr][ite], C)
                a = roc_auc_score(y[tr], s)
                if a > best:
                    best, bestkey = a, (l, C)
        l, C = bestkey
        picks.append(bestkey)
        score[te], _, _ = fit_predict(feats[l][tr], y[tr], feats[l][te], C)
    return score, picks


def metrics(y, s):
    auc = roc_auc_score(y, s)
    pred = (s >= np.median(s)).astype(int)          # 중앙값 분할 — 제출물과 같은 판정 방식
    return auc, balanced_accuracy_score(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="Skywork-OR1-Math-7B")
    ap.add_argument("--null", type=int, default=200)
    a = ap.parse_args()
    own = np.load(PROBE / f"{a.alias}_own.npz", allow_pickle=True)
    off = np.load(PROBE / f"{a.alias}_official.npz", allow_pickle=True)
    layers = [int(x) for x in own["layers"]]
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    F = {l: own[f"h{l}"] for l in layers}
    FO = {l: off[f"h{l}"] for l in layers}
    words = own["words"].astype(float)
    lw = np.log(np.maximum(1, words)).reshape(-1, 1)
    lwo = np.log(np.maximum(1, off["words"].astype(float))).reshape(-1, 1)
    rng = np.random.default_rng(0)

    out(f"# 프로브 파일럿 — {a.alias}")
    out()
    out(f"자체 라벨 {len(own['y'])}문제 · 층 인덱스(hidden_states) {layers} · 차원 {F[layers[0]].shape[1]} · "
        f"C 후보 {CS} · PCA {NPC}성분(폴드 학습분으로만 적합) · null {a.null}회")
    out()
    out("| 라벨 | n | robust/spurious | AUC | balanced acc | null95% AUC | 길이 대조군 AUC | 고른 (층,C) |")
    out("|---|---|---|---|---|---|---|---|")
    results = {}
    for tag, key in (("label (미완주=오답)", "y"), ("label_fin (완주만)", "y_fin")):
        y = own[key].astype(int)
        if len(set(y)) < 2:
            out(f"| {tag} | {len(y)} | 한 클래스 | — | — | — | — | — |")
            continue
        s, picks = nested_cv(F, y, layers)
        auc, bacc = metrics(y, s)
        # 셔플 null: 라벨을 섞고 **같은 nested CV 를 다시** 돌린다(점수만 섞으면 null 이 좁아진다)
        nl = []
        for _ in range(a.null):
            ys = rng.permutation(y)
            try:
                ss, _ = nested_cv(F, ys, [layers[len(layers) // 2]], seed=1, folds=3)
                nl.append(roc_auc_score(ys, ss))
            except Exception:
                pass
        lo, hi = (np.percentile(nl, [2.5, 97.5]) if nl else (np.nan, np.nan))
        sl, _ = nested_cv({0: lw}, y, [0])
        lauc, _ = metrics(y, sl)
        flag = "▲" if auc > hi else ""
        results[key] = dict(y=y, s=s, auc=auc, bacc=bacc, lo=lo, hi=hi, lauc=lauc, picks=picks)
        pk = ", ".join(f"({l},{c})" for l, c in picks[:3]) + ("…" if len(picks) > 3 else "")
        out(f"| {tag} | {len(y)} | {int(y.sum())}/{int((1 - y).sum())} | **{auc:.3f}**{flag} | {bacc:.3f} | "
            f"[{lo:.2f}, {hi:.2f}] | {lauc:.3f} | {pk} |")
    out()
    out("판정은 중앙값 분할(제출물과 같은 방식)이라 balanced acc 가 실제 bool 정확도에 대응한다.")
    out("길이 대조군은 log(단어수) 하나만 쓴 같은 파이프라인이다 — 프로브가 길이를 재는 것 이상인지 보려면 이 열과 비교해야 한다.")
    out()

    # 외부 테스트: 자체 88개 전부로 학습 → 공식 33문제에서 시험 (층·C 는 내부 CV 로)
    out("## 외부 테스트 — 공식 train-main-v2 33문제 (학습에 쓰지 않음)")
    out()
    out("| 학습 라벨 | 고른 (층,C) | 외부 AUC | 외부 balanced acc | 외부 null95% | 외부 길이 대조군 AUC |")
    out("|---|---|---|---|---|---|")
    yo = off["y"].astype(int)
    for tag, key in (("label", "y"), ("label_fin", "y_fin")):
        if key not in results:
            continue
        y = results[key]["y"]
        best, bk = -1, None
        inner = StratifiedKFold(4, shuffle=True, random_state=0)
        for l in layers:
            for C in CS:
                s = np.zeros(len(y))
                for itr, ite in inner.split(F[l], y):
                    s[ite], _, _ = fit_predict(F[l][itr], y[itr], F[l][ite], C)
                aa = roc_auc_score(y, s)
                if aa > best:
                    best, bk = aa, (l, C)
        l, C = bk
        se, _, _ = fit_predict(F[l], y, FO[l], C)
        auc_e, bacc_e = metrics(yo, se)
        sle, _, _ = fit_predict(lw, y, lwo, 1.0)
        auc_l, _ = metrics(yo, sle)
        # 외부 permutation 밴드: 점수는 고정이고 라벨만 섞는다(1000회, 비용 0)
        nl = [roc_auc_score(rng.permutation(yo), se) for _ in range(1000)]
        elo, ehi = np.percentile(nl, [2.5, 97.5])
        fl = "▲" if auc_e > ehi else ("▼" if auc_e < elo else "")
        out(f"| {tag} | ({l}, {C}) | **{auc_e:.3f}**{fl} | {bacc_e:.3f} | [{elo:.2f}, {ehi:.2f}] | {auc_l:.3f} |")
    out()
    out(f"외부 테스트 셋: 공식 문제 단위 라벨 {len(yo)}개 (robust {int(yo.sum())}/spurious {int((1 - yo).sum())}). ")
    out("**Skywork 자기 라벨 행은 공식 학습셋에 3개뿐이라 그것만으로는 시험할 수 없다** → 문제 단위 라벨로 대신한다.")
    (ROOT / "report" / "probe_pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n보고서 저장: report/probe_pilot.md")


if __name__ == "__main__":
    main()
