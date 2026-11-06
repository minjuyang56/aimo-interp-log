"""#SAE-OLMO 평가 — 희소 SAE 특징 vs 현 PCA24+LR. **#GEN-PROBE 프로토콜 그대로.**

사전등록: CLAUDE.md `## #SAE-OLMO` (커밋 `69bb2bf`).

## 같은 폴드·같은 null 경로를 쓴다
`probe_final_eval` 의 `make_plan`·`nested_scores`·`null_band`·`resid` 를 **그대로 import** 한다 —
비교 상대 A(현 PCA24+LR)와 기준선 3종이 그 경로로 돌아야 '특징이 좋은가' 와 '평가경로가 다른가'
가 섞이지 않는다. **SAE 쪽만** 분류기를 L1 로지스틱으로 바꾼다(지휘 지정). 폴드 분할·순열 수열·
건너뛰기 규칙은 동일하다.

## 사전 선별 (라벨 미사용, 사전등록 (2))
163,840 latents 를 31행으로 학습할 수 없으므로 **라벨을 보지 않고** 후보를 K=256 으로 줄인다:
  `fire` = 발화율 상위 (전 토큰에서 활성 > 0 인 비율)
  `var`  = 행별 평균 활성의 분산 상위
두 기준을 **따로** 보고한다. 둘 다 라벨과 무관하지만 **평가에 쓰는 같은 31행**에서 고르므로
transductive 하다 — 사전등록에 적은 그대로이고, 통과하면 폴드 내부 선별판도 같이 돌린다.

사용:  .venv/Scripts/python.exe analysis/sae_olmo_eval.py --perms 1000 --jobs 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from probe_final_eval import (auc_of, bacc_of, make_plan, nested_scores,  # noqa: E402
                              null_band, resid)

P = ROOT / "experiments" / "results" / "probe"
SAE = ROOT / "experiments" / "results" / "sae"
ALIAS = "Olmo-3-7B-Think"
LAYER = 16
K_SEL = 256
CS_L1 = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
L = []


def out(t=""):
    print(t, flush=True)
    L.append(t)


# ---------------------------------------------------------------- SAE 특징
def sae_features(quant):
    """희소 활성 → (평균 활성, 발화빈도) 밀집 행렬 두 개. 163,840 차원."""
    z = np.load(SAE / f"{ALIAS}_sae_sparse_{quant}.npz", allow_pickle=True)
    pidx = z["pidx"].astype(int)
    d_sae = int(json.loads((SAE / f"{ALIAS}_sae_gate_{quant}.json").read_text(encoding="utf-8"))["d_sae"])
    mean = np.zeros((len(pidx), d_sae), dtype=np.float32)
    fire = np.zeros((len(pidx), d_sae), dtype=np.float32)
    for n in range(len(pidx)):
        idx, val = z[f"i{n}"], z[f"v{n}"]
        T = idx.shape[0]
        np.add.at(mean[n], idx.reshape(-1), val.reshape(-1))
        mean[n] /= T
        pos = val.reshape(-1) > 0
        np.add.at(fire[n], idx.reshape(-1)[pos], 1.0)
        fire[n] /= T
    return pidx, mean, fire


def select(mean, fire, k=K_SEL):
    """라벨을 보지 않는 두 선별 기준. 반환: {이름: 인덱스 배열}"""
    return {"fire": np.argsort(-fire.mean(0))[:k],
            "var": np.argsort(-mean.var(0))[:k]}


# ---------------------------------------------------------------- L1 nested CV (같은 폴드 구조)
def plan_l1(X, y, folds=5, seed=0):
    kmin = int(min(np.bincount(y))) if len(set(y.tolist())) > 1 else 0
    if kmin < 2:
        return None
    outer = list(StratifiedKFold(min(folds, kmin), shuffle=True, random_state=seed).split(X, y))
    plan = {"outer": outer, "inner": [], "proj": {}}
    for fi, (tr, te) in enumerate(outer):
        ykmin = int(min(np.bincount(y[tr])))
        plan["inner"].append(list(StratifiedKFold(min(4, max(2, ykmin)), shuffle=True,
                                                  random_state=seed).split(X[tr], y[tr]))
                             if ykmin >= 2 else [])
        sc = StandardScaler().fit(X[tr])
        plan["proj"][fi] = (sc.transform(X[tr]), sc.transform(X[te]))
    return plan


def nested_l1(y, plan, keep_model=False):
    """L1 로지스틱. C 는 폴드 내부 CV 로만 고른다. 반환 (폴드밖 점수, picks, 마지막 전체적합 계수)"""
    score = np.zeros(len(y))
    picks = []
    for fi, (tr, te) in enumerate(plan["outer"]):
        inner = plan["inner"][fi]
        best, bC = -1.0, CS_L1[0]
        if inner and len(set(y[tr].tolist())) > 1:
            Ztr, _ = plan["proj"][fi]
            for C in CS_L1:
                s_in, ok = np.zeros(len(tr)), True
                for itr, ite in inner:
                    if len(set(y[tr][itr].tolist())) < 2:
                        ok = False
                        break
                    m = LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                           class_weight="balanced", max_iter=5000)
                    m.fit(Ztr[itr], y[tr][itr])
                    s_in[ite] = m.decision_function(Ztr[ite])
                if not ok:
                    continue
                aa = roc_auc_score(y[tr], s_in)
                if aa > best:
                    best, bC = aa, C
        picks.append(float(bC))
        Ztr, Zte = plan["proj"][fi]
        if len(set(y[tr].tolist())) < 2:
            continue
        m = LogisticRegression(C=bC, penalty="l1", solver="liblinear",
                               class_weight="balanced", max_iter=5000)
        m.fit(Ztr, y[tr])
        score[te] = m.decision_function(Zte)
    coef = None
    if keep_model:
        sc = StandardScaler().fit(plan["_X"]) if "_X" in plan else None
        if sc is not None:
            m = LogisticRegression(C=max(set(picks), key=picks.count), penalty="l1",
                                   solver="liblinear", class_weight="balanced", max_iter=5000)
            m.fit(sc.transform(plan["_X"]), y)
            coef = m.coef_[0]
    return score, picks, coef


def null_l1(y, plan, logw, perms, jobs):
    from probe_final_eval import _perm_list
    ps = _perm_list(y, perms)
    raw, res = [], []

    def one(yp):
        sp, _, _ = nested_l1(yp, plan)
        return roc_auc_score(yp, sp), roc_auc_score(yp, resid(sp, logw))

    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for a, b in ex.map(_null_worker, [(yp, plan, logw) for yp in ps], chunksize=8):
                raw.append(a)
                res.append(b)
    else:
        for yp in ps:
            a, b = one(yp)
            raw.append(a)
            res.append(b)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return q(raw), q(res), len(ps)


def _null_worker(args):
    yp, plan, logw = args
    sp, _, _ = nested_l1(yp, plan)
    return roc_auc_score(yp, sp), roc_auc_score(yp, resid(sp, logw))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "experiments/results/sae_olmo_report.txt"))
    a = ap.parse_args()

    gate = json.loads((SAE / f"{ALIAS}_sae_gate_{a.quant}.json").read_text(encoding="utf-8"))
    out(f"#SAE-OLMO 평가 — {gate['sae']} · hidden_states[{gate['layer_hs']}] · {a.quant} · null {a.perms}회")
    out(f"품질 게이트: 분산설명률 **{gate['explained_variance']:.4f}** · L0 **{gate['l0']:.2f}** "
        f"(TopK k={gate['k']}) · d_sae {gate['d_sae']}")
    out("")
    if gate["explained_variance"] < 0.7:
        out("!! 게이트 미달 — 사전등록대로 평가하지 않는다")
        Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
        return 2

    d = np.load(P / f"{ALIAS}_gen2048.npz", allow_pickle=True)
    y = d["y"].astype(int)
    words = d["words"].astype(float)
    logw = np.log(np.maximum(1.0, words))
    pidx_ref = d["pidx"].astype(int)
    pidx, mean, fire = sae_features(a.quant)
    assert np.array_equal(pidx, pidx_ref), "행 순서가 특징 npz 와 다르다"
    out(f"행 {len(y)} (robust {int(y.sum())} / spurious {int((1 - y).sum())}) · "
        f"SAE 활성 평균 행렬 {mean.shape}")
    out(f"전체 latent 중 한 번이라도 발화한 것 **{int((fire > 0).any(0).sum()):,}** / {mean.shape[1]:,}")
    out("")

    rows = {}
    # ---- A: 현 설계 (PCA24 + L2 LR), 층 16 고정 · 같은 폴드 경로
    A = {LAYER: d[f"h{LAYER}"].astype(np.float64)}
    plan = make_plan(A, y, [LAYER])
    sA, picksA = nested_scores(y, [LAYER], plan)
    nrA, ndA = null_band(y, [LAYER], plan, words, a.perms, a.jobs)
    rows["A: 현 PCA24+LR (밀집 4096)"] = (auc_of(y, sA), auc_of(y, resid(sA, logw)), bacc_of(y, sA), nrA, ndA)

    # ---- B: SAE 희소 특징 (L1), 선별 기준 둘 × 특징 둘
    sel = select(mean, fire)
    coefs = {}
    for selname, ix in sel.items():
        for featname, M in (("평균활성", mean), ("발화빈도", fire)):
            X = M[:, ix].astype(np.float64)
            pl = plan_l1(X, y)
            pl["_X"] = X
            s, picks, coef = nested_l1(y, pl, keep_model=True)
            nr, nd = null_l1(y, pl, logw, a.perms, a.jobs)
            key = f"B: SAE {featname} · 선별 {selname} (K={len(ix)})"
            rows[key] = (auc_of(y, s), auc_of(y, resid(s, logw)), bacc_of(y, s), nr, nd)
            coefs[key] = (ix, coef, picks)

    # ---- 기준선 3종: 같은 폴드·같은 nested 경로, 1차원
    for lbl, col in (("기준선1 base_acc(k=8)", "base_acc"), ("기준선2 greedy 정답여부", "correct"),
                     ("기준선3 greedy 토큰수", "n_tok")):
        if col not in d:
            continue
        x = d[col].astype(np.float64).reshape(-1, 1)
        if x.std() == 0:
            rows[lbl] = None
            continue
        F = {LAYER: x}
        pb = make_plan(F, y, [LAYER])
        sb, _ = nested_scores(y, [LAYER], pb)
        nrb, ndb = null_band(y, [LAYER], pb, words, a.perms, a.jobs)
        rows[lbl] = (auc_of(y, sb), auc_of(y, resid(sb, logw)), bacc_of(y, sb), nrb, ndb)

    out(f"{'설계':44s} {'AUC':>7s} {'null 95%':>16s} {'길이잔차':>8s} {'bacc':>6s}")
    for k, v in rows.items():
        if v is None:
            out(f"{k:44s} {'분산 0 → 판정 불가':>7s}")
            continue
        auc, res_, bac, nr, nd = v
        mark = " ▲" if auc > nr[1] else ""
        out(f"{k:44s} {auc:>7.3f} [{nr[0]:.2f}, {nr[1]:.2f}]{mark:3s} {res_:>8.3f} {bac:>6.3f}")
    out("")

    # ---- 판정 (사전등록 기준 셋)
    base = [v[0] for k, v in rows.items() if k.startswith("기준선") and v]
    bestB = max(((k, v) for k, v in rows.items() if k.startswith("B:")), key=lambda kv: kv[1][0])
    aucA = rows["A: 현 PCA24+LR (밀집 4096)"][0]
    g1 = bestB[1][0] > bestB[1][3][1]
    g2 = all(bestB[1][0] > b for b in base) if base else False
    g3 = bestB[1][0] > aucA
    out("## 판정 (사전등록: 셋 다 만족해야 'SAE 가 더 낫다')")
    out(f"   가장 높은 SAE 설계: {bestB[0]} · AUC {bestB[1][0]:.3f}")
    out(f"   1 null 95% 상단 {bestB[1][3][1]:.2f} 초과 → {'통과' if g1 else '미달'}")
    out(f"   2 기준선 3종 최대 {max(base) if base else float('nan'):.3f} 초과 → {'통과' if g2 else '미달'}")
    out(f"   3 A(현 설계) {aucA:.3f} 초과 → {'통과' if g3 else '미달'}")
    out(f"   → **{'SAE 가 더 낫다' if (g1 and g2 and g3) else '개선 미입증'}**")
    out("")

    # ---- 부가: 상위 가중 latent (해석용, 판정 미사용)
    out("## 부가 — 상위 가중 latent 5개 (해석용 · 판정에 쓰지 않는다)")
    ix, coef, picks = coefs[bestB[0]]
    if coef is not None:
        order = np.argsort(-np.abs(coef))[:5]
        out(f"   고른 C: {picks} · 비영 계수 {int((coef != 0).sum())}/{len(coef)}")
        for r in order:
            lat = int(ix[r])
            out(f"   latent {lat:>7d} · 계수 {coef[r]:+.4f} · 발화율 {fire[:, lat].mean():.4f} · "
                f"평균활성 분산 {mean[:, lat].var():.2e}")
        out("   (대표 토큰 문맥은 analysis/sae_olmo_context.py 로 따로 뽑는다 — 희소 활성에 토큰 id 가 있다)")
    else:
        out("   계수를 얻지 못했다")
    out("")
    out("## 한계")
    out("   n=31 · spurious 10 이라 null 폭이 넓다(#77 Olmo [0.19, 0.74]). 이 규모에서 AUC 0.75")
    out("   미만은 우연과 구분할 수 없다 — 이 프로젝트의 중심 문장이 그대로 적용된다.")
    out("   선별이 transductive 하다(라벨은 안 보지만 평가에 쓰는 같은 31행에서 골랐다).")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
