"""#LIT-PLAN P1 — S(폴드 안 단변량 선별, **탐색**) + mass-mean 전이(**사전등록 1회**). 비용 $0.

사전등록은 CLAUDE.md `## #LIT-PLAN` 절에 있다. 요점만:

## P1a — S 를 실제 데이터에 돌린다 (탐색 표기)
`power_select.txt` 가 합성 신호(d=4096 · 8σ · n=33)에서 **A 0.455 vs S 0.701** 을 쟀다.
#77·#GEN-PROBE·#CMP4·#ORG-TRAIN 이 전부 A(PCA24+LR) 하나로만 평가됐으므로 **같은 데이터에서
결론이 바뀔 수 있다**. S 의 정의: 폴드 **학습분만** 표준화 → 학습분에서 `|corr(z_j, y)|` 상위
**24차원** → 로지스틱 L2 balanced. PCA 없음.
  * 선별은 y 를 보지만 **적합 분할 안에서만** 본다 — `cmp4_eval.nested` 의 `fit_score` 인터페이스에
    꽂으면 내부 CV 폴드에서도 그 폴드의 학습분으로 다시 고르므로 **완전 중첩**이다.
    (`power_select.py` 는 C 를 학습 AUC 로 골랐지만, 여기서는 A 와 공정하게 맞대려고
     **A 와 똑같은 내부 CV 프로토콜**을 쓴다. 그 차이를 보고서에 적는다.)
  * **셔플 null 에서도 선별을 매번 다시 한다** → 선택 편향이 null 에 들어간다.
두 칸: (i) 자체 라벨 창 2048(#CMP4 와 같은 행·폴드·null) · (ii) 공식 33 LOPO(#ORG-TRAIN 과 같은 경로).

## P1b — mass-mean(평균차) 전이 (사전등록 1회)
자체 라벨 전체에서 `w = mean(z[y=1]) − mean(z[y=0])` 하나만 적합 → **공식 33 에 1회 적용**.
층 L//2 고정 · 부호 '점수 큼 = robust'(뒤집지 않는다) · 임계 둘(학습 중앙값 · 공식 LOO 사전확률
교정) 병기 · 통과 기준 = 모델평균 AUC > null 95% 상단 **그리고** 3모델 중 2 이상 각자 초과.

사용:
  .venv/Scripts/python.exe analysis/p1_lit.py --gate              # 틀 동일성만
  .venv/Scripts/python.exe analysis/p1_lit.py --perms 1000 --jobs 10
  .venv/Scripts/python.exe analysis/p1_lit.py --part b --perms 1000
"""

import argparse
import os
import sys
from pathlib import Path

#  **BLAS 스레드를 1 로 묶는다.** 이 실험의 로지스틱은 (32행 × 24차원) 짜리라 커널이 아니라
#  스레드 생성이 비용이다 — 실측 LR 1회 51 ms(기본) vs **22 ms(1스레드)**, perm 1회 47.7s → 15.0s.
#  게다가 joblib 워커 16개가 각자 20스레드를 띄우면 서로 밀어낸다. 통계량은 바뀌지 않는다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))

import cmp4_eval as C4  # noqa: E402
import org_train_eval as OT  # noqa: E402
from probe_final_eval import CS, auc_of, bacc_of, make_plan, resid, _perm_list  # noqa: E402
from sub2048_plus_eval import acc_of, band as acc_band, pred_loo_prior  # noqa: E402

FEAT = ROOT / "experiments/results/probe"
OUT = ROOT / "experiments/results/p1_lit_report.txt"
NSEL = 24
#  (ii) 와 (i) 의 A 재현 게이트 — 사전등록에 적은 기존 기록
GATE_OWN = {"Skywork": 0.981, "Olmo": 0.762, "DeepSeek": 0.874}
GATE_ORG = {"Skywork-OR1-Math-7B": 0.565, "Olmo-3-7B-Think": 0.638,
            "DeepSeek-R1-0528-Qwen3-8B": 0.515, "pool": 0.573}
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


# ───────────────────────── S 의 정의 (모듈 최상단 = 병렬에서 pickle 가능) ─────────────────────────

def sel_idx(Ztr, ytr, npc=NSEL):
    """학습분에서만 |corr(z_j, y)| 상위 npc 차원. 동점은 인덱스 순서(argsort 안정)."""
    yc = np.asarray(ytr, float)
    yc = yc - yc.mean()
    num = Ztr.T @ yc
    den = np.linalg.norm(Ztr, axis=0) * np.linalg.norm(yc) + 1e-12
    return np.argsort(-np.abs(num / den), kind="stable")[:npc]


def fs_select(Ztr, ytr, Zte, g, _t=None):
    keep = sel_idx(Ztr, ytr)
    m = LogisticRegression(C=g, class_weight="balanced", max_iter=5000)
    m.fit(Ztr[:, keep], ytr)
    return m.decision_function(Zte[:, keep])


# ───────────────────────── (ii) LOPO 틀 — 배열로 받아 병렬 비용을 줄인다 ─────────────────────────

def stack_proj(proj, dtype=np.float64):
    """`[(Ztr, Zte, tr), ...]` → 한 배열씩. 33개 작은 배열을 그대로 넘기면 joquib 이
    메모리맵을 못 걸어 perm 마다 15 MB 를 피클한다."""
    Ztr = np.stack([p[0] for p in proj]).astype(dtype)
    Zte = np.stack([p[1] for p in proj]).astype(dtype)
    tr = np.stack([p[2] for p in proj])
    return Ztr, Zte, tr


def lopo_arrays(Ztr_all, Zte_all, tr_all, y, fit_score, grid):
    """문제 단위 LOPO. (하이퍼)는 **학습 폴드 내부 CV 의 AUC** 로, 임계는 **학습분 두 클래스
    중앙값의 중점**(`OT.mid_thr`)으로 정한다 — `org_train_eval.lopo_probe` 와 같은 절차다."""
    n = len(y)
    sc = np.zeros(n)
    pr = np.zeros(n, dtype=bool)
    picks = []
    for i in range(n):
        Ptr, Pte, tr = Ztr_all[i], Zte_all[i], tr_all[i]
        ytr = y[tr]
        best, bg = -1.0, grid[0]
        k = min(4, int(np.bincount(ytr, minlength=2)[:2].min()))
        if k >= 2 and len(grid) > 1:
            inner = list(StratifiedKFold(k, shuffle=True, random_state=0).split(Ptr, ytr))
            for g in grid:
                s_in, ok = np.zeros(len(tr)), True
                for a, b in inner:
                    if len(set(ytr[a].tolist())) < 2:
                        ok = False
                        break
                    s_in[b] = fit_score(Ptr[a], ytr[a], Ptr[b], g)
                if not ok:
                    continue
                v = auc_of(ytr, s_in)
                if v > best:
                    best, bg = v, g
        picks.append(bg)
        s = fit_score(Ptr, ytr, np.vstack([Ptr, Pte]), bg)
        sc[i] = float(s[len(tr):][0])
        pr[i] = sc[i] >= OT.mid_thr(s[:len(tr)], ytr)
    return sc, pr, picks


def _one_org(yp, A, S):
    """perm 하나에 A·S 를 둘 다 돌린다 → 같은 null 추출을 쓴다."""
    sa, pa, _ = lopo_arrays(A[0], A[1], A[2], yp, C4.fs_logistic, CS)
    ss, ps, _ = lopo_arrays(S[0], S[1], S[2], yp, fs_select, CS)
    return (auc_of(yp, sa), float((pa == (yp == 1)).mean()),
            auc_of(yp, ss), float((ps == (yp == 1)).mean()))


def org_null(y, A, S, perms, jobs):
    yps = _perm_list(y, perms)
    if jobs and jobs != 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(_one_org)(yp, A, S) for yp in yps)
    else:
        res = [_one_org(yp, A, S) for yp in yps]
    return np.array(res)


# ───────────────────────── 게이트 ─────────────────────────

def gate_frame():
    """배열판 LOPO 가 `org_train_eval.lopo_probe` 와 같은 점수를 내는지 계측(합성 난수)."""
    rng = np.random.default_rng(3)
    n, d = 24, 120
    X = rng.standard_normal((n, d))
    y = np.array([1] * 15 + [0] * 9)
    X[y == 1, :4] += 0.7
    proj = OT.prep_proj(X, n)
    s1, p1, c1 = OT.lopo_probe(proj, y)
    A = stack_proj(proj)
    s2, p2, c2 = lopo_arrays(A[0], A[1], A[2], y, C4.fs_logistic, CS)
    dd = float(np.abs(s1 - s2).max())
    ok = dd < 1e-5 and (p1 == p2).all() and list(c1) == list(c2)
    print(f"[게이트] 배열판 LOPO vs org_train_eval.lopo_probe · 점수 최대절대차 {dd:.3e} · "
          f"예측 일치 {int((p1 == p2).sum())}/{n} · 고른 C 일치 {list(c1) == list(c2)} · "
          f"{'동일' if ok else '**다르다**'}", flush=True)
    ok2 = C4.verify_frame()
    return ok and ok2


# ───────────────────────── P1a (i) 자체 라벨 창 2048 ─────────────────────────

def part_a_own(perms, jobs):
    out("=" * 78)
    out("## P1a (i) — 자체 라벨 창 2048 · A(PCA24+LR) vs **S(폴드 안 단변량 상위 24)** · 탐색")
    out(f"   같은 행·같은 폴드(`make_plan` 재사용)·같은 null 추출(행 인덱스 순열) · null {perms}회")
    out()
    res, hi, rr, hr = {}, {}, {}, {}
    gate_ok = True
    for alias, npz, segnpz, _md in C4.MODELS:
        z, k8, y, words, pidx, feats, segs, layers = C4.load_model(npz, segnpz)
        plan = make_plan(feats, y, layers, 5, 0)
        if plan is None:
            out(f"## {alias} — 폴드 생성 불가")
            continue
        out(f"## {alias} · n {len(y)} (robust {int(y.sum())} / spurious {int((1 - y).sum())})")
        logw = np.log(np.maximum(1, words))
        for key, kind, fit in (("A", "pca", C4.fs_logistic), ("S", "raw", fs_select)):
            pl = C4.prep(kind, plan, feats, segs, layers)
            s, picks = C4.nested(y, layers, pl, fit, CS)
            nr, nd = C4.band(y, layers, pl, fit, CS, None, words, perms, jobs)
            au, rs = auc_of(y, s), auc_of(y, resid(s, logw))
            h, h2 = float(np.percentile(nr, 95)), float(np.percentile(nd, 95))
            res.setdefault(key, {})[alias] = au
            hi.setdefault(key, {})[alias] = h
            rr.setdefault(key, {})[alias] = rs
            hr.setdefault(key, {})[alias] = h2
            out(f"   {key}  AUC **{au:.3f}** · null 95% [{np.percentile(nr, 5):.2f}, {h:.2f}]"
                f"{' ▲' if au > h else ''} · 길이잔차 {rs:.3f} (상단 {h2:.2f}) · "
                f"bacc {bacc_of(y, s):.3f}")
            out(f"      고른 (층, C) {picks}")
            if key == "A":
                g = GATE_OWN[alias]
                okg = abs(au - g) < 0.002
                gate_ok &= okg
                out(f"      재현 게이트: 기존 기록 {g:.3f} vs 이번 {au:.3f} → "
                    f"{'통과' if okg else '**불일치**'}")
        out()
    if not gate_ok:
        out("**A 재현 게이트 실패 → S 숫자를 해석하지 않는다.**")
    ma = {k: float(np.mean(list(v.values()))) for k, v in res.items()}
    mh = {k: float(np.mean(list(hi[k].values()))) for k in res}
    mr = {k: float(np.mean(list(rr[k].values()))) for k in res}
    mhr = {k: float(np.mean(list(hr[k].values()))) for k in res}
    out("### (i) 요약 — 모델평균")
    out(f"   {'방법':<6}{'모델평균 AUC':>13}{'null상단평균':>13}{'길이잔차':>10}{'(상단)':>9}"
        f"{'A 대비':>9}")
    for k in ("A", "S"):
        out(f"   {k:<6}{ma[k]:>13.3f}{mh[k]:>13.2f}{mr[k]:>10.3f}{mhr[k]:>9.2f}"
            f"{(ma[k] - ma['A']):>+9.3f}")
    out()
    return ma, mh


# ───────────────────────── P1a (ii) 공식 33 LOPO ─────────────────────────

def part_a_org(perms, jobs):
    out("=" * 78)
    out("## P1a (ii) — 공식 33 LOPO · 주최측 `is_robust` · A vs S · 탐색")
    out(f"   `org33` 특징(창 2048 · 층 L//2 하나) · 임계는 학습 폴드 안에서만 · null {perms}회")
    out()
    from h21_uncertainty import problems
    lab = {r["problem"]: r["y"] for r in problems()}
    pool = {k: {"s": [], "y": [], "p": []} for k in ("A", "S")}
    rows = []
    gate_ok = True
    for mid, alias in OT.MODELS.items():
        d = np.load(FEAT / f"{alias}_org33.npz", allow_pickle=True)
        pt = [str(x) for x in d["problems"]]
        y = np.array([lab[t] for t in pt], dtype=int)
        X = d["X"].astype(np.float64)
        n = len(y)
        A = stack_proj(OT.prep_proj(X, n))
        Sp = []
        for i in range(n):
            tr = np.array([j for j in range(n) if j != i])
            sc = StandardScaler().fit(X[tr])
            Sp.append((sc.transform(X[tr]), sc.transform(X[i:i + 1]), tr))
        S = stack_proj(Sp, np.float32)   # 34 MB -> 17 MB. 선별 통계에 정밀도 무관
        nul = org_null(y, A, S, perms, jobs)
        got = {}
        for key, idx, pack, fit in (("A", 0, A, C4.fs_logistic), ("S", 2, S, fs_select)):
            s, pr, picks = lopo_arrays(pack[0], pack[1], pack[2], y, fit, CS)
            au, ac = auc_of(y, s), float((pr == (y == 1)).mean())
            h, hac = float(np.percentile(nul[:, idx], 95)), float(np.percentile(nul[:, idx + 1], 95))
            got[key] = (au, ac, h, hac)
            pool[key]["s"].append(s)
            pool[key]["y"].append(y)
            pool[key]["p"].append(pr)
            uc = {c: picks.count(c) for c in sorted(set(picks))}
            out(f"   {alias[:26]:<26} {key}  AUC **{au:.3f}** · null 95% 상단 {h:.2f}"
                f"{' ▲' if au > h else ''} · 정확도 {ac:.3f} (상단 {hac:.2f}) · "
                f"True비율 {pr.mean():.3f} · 고른 C {uc}")
        rows.append((alias, got))
        g = GATE_ORG[alias]
        okg = abs(got["A"][0] - g) < 0.002
        gate_ok &= okg
        out(f"   {'':<26} 재현 게이트 A: 기존 {g:.3f} vs 이번 {got['A'][0]:.3f} → "
            f"{'통과' if okg else '**불일치**'}")
    out()
    out("### (ii) 풀링 99행 (주지표, #ORG-TRAIN 과 같은 단위)")
    psum = {}
    for key in ("A", "S"):
        ps = np.concatenate(pool[key]["s"])
        py = np.concatenate(pool[key]["y"])
        pp = np.concatenate(pool[key]["p"])
        psum[key] = (auc_of(py, ps), float((pp == (py == 1)).mean()))
        out(f"   {key}  풀링 AUC **{psum[key][0]:.3f}** · 정확도 {psum[key][1]:.3f} · n {len(py)}")
    g = GATE_ORG["pool"]
    okp = abs(psum["A"][0] - g) < 0.002
    out(f"   재현 게이트 풀링 A: 기존 {g:.3f} vs 이번 {psum['A'][0]:.3f} → "
        f"{'통과' if okp else '**불일치**'}")
    if not (gate_ok and okp):
        out("**A 재현 게이트 실패 → S 숫자를 해석하지 않는다.**")
    out()
    return psum, rows


# ───────────────────────── P1b mass-mean 전이 ─────────────────────────

def auc_band(score, y, perms, seed=0):
    """점수를 고정하고 라벨만 섞는다 → 밴드는 n·클래스비율로 정해진다."""
    rng = np.random.default_rng(seed)
    v = [auc_of(rng.permutation(y), score) for _ in range(perms)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


OWN = [("Skywork-OR1-Math-7B", "Skywork-OR1-Math-7B_gen2048.npz"),
       ("Olmo-3-7B-Think", "Olmo-3-7B-Think_gen2048.npz"),
       ("DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-0528-Qwen3-8B_gen2048_supp.npz")]


def part_b(perms):
    out("=" * 78)
    out("## P1b — mass-mean(평균차) 전이 · **사전등록 1회** (자체 라벨 학습 → 공식 33 시험)")
    out("   층 L//2 고정 · PCA 없음 · 적합 파라미터 = 방향 w 하나 · 부호: 점수 큼 = robust")
    out()
    from h21_uncertainty import problems
    lab = {r["problem"]: r["y"] for r in problems()}
    per, ys = {}, None
    words = None
    for alias, npz in OWN:
        z = np.load(FEAT / npz, allow_pickle=True)
        layers = [int(x) for x in z["layers"]]
        lmid = layers[1]                                   # L//4, **L//2**, 3L//4, L
        k8 = np.asarray(z["gen_k"], int) == 8
        ytr = np.asarray(z["y"], int)[k8]
        Xtr = z[f"h{lmid}"][k8].astype(np.float64)
        d = np.load(FEAT / f"{alias}_org33.npz", allow_pickle=True)
        if int(d["layer"]) != lmid:
            raise SystemExit(f"{alias}: org33 층 {int(d['layer'])} vs 학습 층 {lmid} — 다르다")
        pt = [str(x) for x in d["problems"]]
        y = np.array([lab[t] for t in pt], dtype=int)
        Xte = d["X"].astype(np.float64)
        if ys is None:
            ys = y
            words = np.array([len(t.split()) for t in pt], float)
        elif not np.array_equal(ys, y):
            raise SystemExit(f"{alias}: 공식 33 라벨 순서가 다르다")
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        w = Ztr[ytr == 1].mean(0) - Ztr[ytr == 0].mean(0)
        s_tr, s_te = Ztr @ w, Zte @ w
        thr_tr = float(np.median(s_tr))
        pred_a = s_te >= thr_tr
        pred_b = pred_loo_prior(s_te, y)
        au = auc_of(y, s_te)
        lo, h = auc_band(s_te, y, perms)
        rs = auc_of(y, resid(s_te, np.log(np.maximum(1, words))))
        nb = acc_band(lambda sx, yx: sx >= thr_tr, s_te, y, perms)
        nb2 = acc_band(pred_loo_prior, s_te, y, perms)
        per[alias] = dict(auc=au, hi=h, lo=lo, resid=rs,
                          acc_a=acc_of(pred_a, y), tr_a=float(pred_a.mean()),
                          acc_b=acc_of(pred_b, y), tr_b=float(pred_b.mean()),
                          nb_a=nb[1], nb_b=nb2[1], n_tr=len(ytr),
                          pos=int(ytr.sum()), neg=int((1 - ytr).sum()), layer=lmid)
        p = per[alias]
        out(f"   {alias[:26]:<26} 학습 {p['n_tr']}행({p['pos']}/{p['neg']}) · 층 {lmid}")
        out(f"   {'':<26} AUC **{au:.3f}** · 셔플 null 95% [{lo:.2f}, {h:.2f}]"
            f"{' ▲' if au > h else ''} · 길이잔차 {rs:.3f}")
        out(f"   {'':<26} 임계(ㄱ) 학습중앙값 {thr_tr:+.3f} → 정확도 {p['acc_a']:.3f} "
            f"(null상단 {p['nb_a']:.3f}) · True비율 {p['tr_a']:.3f}")
        out(f"   {'':<26} 임계(ㄴ) LOO 사전확률교정 → 정확도 {p['acc_b']:.3f} "
            f"(null상단 {p['nb_b']:.3f}) · True비율 {p['tr_b']:.3f}")
    out()
    aus = [per[a]["auc"] for a, _ in OWN]
    his = [per[a]["hi"] for a, _ in OWN]
    ma, mh = float(np.mean(aus)), float(np.mean(his))
    npass = sum(1 for a, _ in OWN if per[a]["auc"] > per[a]["hi"])
    out("### P1b 통과 기준 (사전등록 — 둘 다 만족해야 '우연 이상')")
    out(f"   1. 모델평균 AUC {ma:.3f} > null 95% 상단 평균 {mh:.3f} → "
        f"{'통과' if ma > mh else '**미달**'}")
    out(f"   2. 3모델 중 2 이상 각자 null 초과 → {npass}/3 → "
        f"{'통과' if npass >= 2 else '**미달**'}")
    out(f"   → **{'전이 입증' if (ma > mh and npass >= 2) else '전이 미입증'}**")
    out()
    out("### 같은 특징·같은 행에서의 비교 (전부 기록된 값)")
    out("   PCA24+LR 전이(#ORG-LABEL 시험2) 모델평균 AUC **0.610** · 정확도 0.505 · True비율 0.172")
    out("   길이 단독 h15C(LOO 교정) AUC 0.671 · 정확도 0.636 · 상수 '전부 robust' 0.606")
    out()
    best_acc = max(max(per[a]["acc_a"], per[a]["acc_b"]) for a, _ in OWN)
    macc_a = float(np.mean([per[a]["acc_a"] for a, _ in OWN]))
    macc_b = float(np.mean([per[a]["acc_b"] for a, _ in OWN]))
    out(f"### zip 빌드 조건 (지휘: 정확도 > 0.65 **그리고** 자기 null 상단 초과)")
    out(f"   모델평균 정확도 (ㄱ) {macc_a:.3f} · (ㄴ) {macc_b:.3f} · 모델별 최고 {best_acc:.3f}")
    okz = False
    for a, _ in OWN:
        for tag, k, nk in (("ㄱ", "acc_a", "nb_a"), ("ㄴ", "acc_b", "nb_b")):
            if per[a][k] > 0.65 and per[a][k] > per[a][nk]:
                okz = True
                out(f"   **빌드 조건 통과**: {a} 임계{tag} 정확도 {per[a][k]:.3f} > 0.65 · "
                    f"null 상단 {per[a][nk]:.3f}")
    if not okz:
        out("   → 어느 칸도 두 조건을 같이 넘지 못했다 → **zip 을 만들지 않는다**")
    out()
    return per, (ma, mh, npass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--part", default="all", choices=["all", "ai", "aii", "b"])
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    if a.gate:
        raise SystemExit(0 if gate_frame() else 1)

    out("#LIT-PLAN P1 — S(단변량 선별, **탐색**) + mass-mean 전이(**사전등록 1회**) · 비용 $0")
    out(f"   S 선별 차원 {NSEL} · C 격자 {CS} · null {a.perms}회 · jobs {a.jobs}")
    out()
    if not gate_frame():
        raise SystemExit("틀 게이트 실패 — 돌리지 않는다")
    out("틀 게이트 통과 (배열판 LOPO == org_train_eval.lopo_probe · nested == nested_scores)")
    out()
    if a.part in ("all", "ai"):
        part_a_own(a.perms, a.jobs)
    if a.part in ("all", "aii"):
        part_a_org(a.perms, a.jobs)
    if a.part in ("all", "b"):
        part_b(a.perms)
    out("## 읽는 법")
    out("   * **P1a 는 탐색이다.** S 가 A 보다 높아도 #77·#GEN-PROBE·#CMP4·#ORG-TRAIN 의 판정을")
    out("     뒤집지 않는다 — 뒤집으려면 새 데이터에서 S 를 주지표로 사전등록해야 한다.")
    out("   * S 는 셔플 null 에서도 선별을 다시 하므로 **선택 편향이 null 에 포함**된다 →")
    out("     S 의 null 상단이 A 보다 높게 나오는 것이 정상이고, 그 위를 넘어야 의미가 있다.")
    out("   * P1b 는 사전등록 1회이고 부호·임계·기준을 결과 보고 바꾸지 않았다.")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
