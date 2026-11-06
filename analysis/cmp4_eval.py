"""#CMP4 — 분류기 4종 비교. 창 2048 풀이 도중 벡터 · 같은 행·같은 폴드·같은 null. **GPU 0.**

사전등록: CLAUDE.md `## #CMP4` (커밋 `cc9cd43` + 보완 `a96c993`). 평가는 1회.

## 네 방법 (지휘 지정)
| 키 | 방법 | 적합하는 것 | 내부 CV 가 고르는 것 |
|---|---|---|---|
| `A` | StandardScaler → PCA 24 → L2 로지스틱 balanced | 24차원 가중치 | 층 · C |
| `B` | StandardScaler → mass-mean (클래스 평균 차) | 방향 하나 | 층 |
| `C` | StandardScaler → `sign(z@R)` D=10000 → 프로토타입 | **없음** | 층 |
| `D` | 16구간 × `sign(z_i@R)` ⊙ `P_i` 묶기 → 프로토타입 | **없음** | 층 |
보조 두 칸: `A-ridge`·`B-ridge` = 같은 전처리에 **연속 `max_drop` Ridge**, 점수 = −예측.

## 폴드를 베껴 쓰지 않는다 — `make_plan` 의 인덱스를 재사용한다
`probe_final_eval.make_plan` 을 한 번 불러 `outer`·`inner` 를 받고, **그 인덱스 그대로** 방법별
전처리만 다시 만든다. 그래서 '네 방법이 같은 폴드' 가 구성상 보장된다(값 비교로 확인할 필요가 없다).
A 는 `probe_final_eval.nested_scores` 를 **import 해서 그대로** 쓴다 — 기존 숫자 재현 게이트가 그걸 검증한다.

## null 1000 을 로컬에서 도는 구조 (사전등록 보완 절)
(폴드, 층)별 전처리를 한 번만 만들고 null 반복에서는 **분류기만** 다시 적합한다.
`StandardScaler`·`PCA`·`sign(z@R)` 은 전부 **y 를 보지 않으므로** 재사용에 누출이 없다.
순열은 **행 인덱스 순열**이라 네 방법이 **같은 null 추출**을 쓴다(방법 간 비교가 공정해진다).
`_perm_list`(#77, y 를 직접 섞음)와 분포는 같지만 배열이 달라 A 의 null 밴드가 `probe_gen_report`
의 밴드와 소수 셋째 자리에서 다를 수 있다 — **점추정 AUC 는 순열과 무관하므로 재현 게이트는 영향 없다.**

사용:
  .venv/Scripts/python.exe analysis/cmp4_eval.py --perms 1000 --jobs 10
  .venv/Scripts/python.exe analysis/cmp4_eval.py --verify-frame   (틀 검증만 하고 끝)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from probe_final_eval import (CS, NPC, auc_of, bacc_of, make_plan,  # noqa: E402
                              nested_scores, resid)

P = ROOT / "experiments" / "results" / "probe"
LG = ROOT / "experiments" / "results" / "runpod" / "labelgen32k"
OUT = ROOT / "experiments" / "results" / "cmp4_report.txt"
WIN = 2048
DIM = 10000                      # 지휘 지정 (스윕하지 않는다)
NSEG = 16
ALPHAS = (1.0, 10.0, 100.0, 1000.0)

MODELS = [
    # 별칭, 특징 npz, 16구간 npz, 연속 max_drop 원천(없으면 None = ridge 판정 불가)
    ("Skywork", "Skywork-OR1-Math-7B_gen2048.npz", "Skywork-OR1-Math-7B_gen2048_seg.npz",
     ["pj7vh1qllei51i/out/Skywork-OR1-Math-7B_off40_v2_labels.json"]),
    ("Olmo", "Olmo-3-7B-Think_gen2048.npz", "Olmo-3-7B-Think_gen2048_seg.npz", None),
    ("DeepSeek", "DeepSeek-R1-0528-Qwen3-8B_gen2048_supp.npz",
     "DeepSeek-R1-0528-Qwen3-8B_gen2048_seg.npz",
     ["mbpc29h3v7um64/out/DeepSeek-R1-0528-Qwen3-8B_off40_v2_labels.json",
      "9svqsw8dzrtjsv/out/DeepSeek-R1-0528-Qwen3-8B_off80_v2_labels.json"]),
]
BASELINES = [("기준선1 base_acc(k=8)", "base_acc"), ("기준선2 greedy 정답여부", "correct"),
             ("기준선3 greedy 토큰수", "n_tok")]
METHODS = ["A", "B", "C", "D"]
RIDGE = ["A-ridge", "B-ridge"]
lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


# ---------------------------------------------------------------- 전처리 (전부 y 를 보지 않는다)

_R = {}


def proj_mat(d):
    """고정 난수 투영. 같은 d 면 항상 같은 R (`default_rng(0)`)."""
    if d not in _R:
        _R[d] = np.random.default_rng(0).standard_normal((d, DIM)).astype(np.float32)
    return _R[d]


def pos_vecs():
    """구간 위치 벡터 P_i ∈ {−1,+1}^D (`default_rng(1)`)."""
    return np.where(np.random.default_rng(1).random((NSEG, DIM)) < 0.5, -1, 1).astype(np.int8)


def sign01(X):
    """sign, 0 은 +1 (사전등록대로)."""
    return np.where(X >= 0, 1, -1).astype(np.int8)


def prep(kind, plan, feats, segs, layers):
    """`plan` 의 폴드 인덱스를 그대로 쓰고 전처리만 새로 만든다."""
    pr = {}
    for fi, (tr, te) in enumerate(plan["outer"]):
        for l in layers:
            if kind == "pca":
                X = feats[l]
                sc = StandardScaler().fit(X[tr])
                Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
                if X.shape[1] > NPC:
                    pc = PCA(n_components=min(NPC, len(tr) - 1, X.shape[1]), random_state=0).fit(Ztr)
                    Ztr, Zte = pc.transform(Ztr), pc.transform(Zte)
                pr[(fi, l)] = (Ztr, Zte)
            elif kind == "raw":
                X = feats[l]
                sc = StandardScaler().fit(X[tr])
                pr[(fi, l)] = (sc.transform(X[tr]), sc.transform(X[te]))
            elif kind == "hdc":
                X = feats[l]
                sc = StandardScaler().fit(X[tr])
                R = proj_mat(X.shape[1])
                pr[(fi, l)] = (sign01(sc.transform(X[tr]).astype(np.float32) @ R),
                               sign01(sc.transform(X[te]).astype(np.float32) @ R))
            elif kind == "hdcseg":
                S = segs[l]                                  # (n, NSEG, d)
                d = S.shape[2]
                sc = StandardScaler().fit(S[tr].reshape(-1, d))
                R, Pv = proj_mat(d), pos_vecs()
                both = []
                for idx in (tr, te):
                    Z = sc.transform(S[idx].reshape(-1, d)).astype(np.float32).reshape(len(idx), NSEG, d)
                    acc = np.zeros((len(idx), DIM), np.int32)
                    for i in range(NSEG):
                        acc += sign01(Z[:, i, :] @ R).astype(np.int32) * Pv[i]
                    both.append(sign01(acc))
                pr[(fi, l)] = tuple(both)
            else:
                raise ValueError(kind)
    return {**plan, "proj": pr}


# ---------------------------------------------------------------- 분류기 (fit_score 인터페이스)

def fs_logistic(Ztr, ytr, Zte, g, _t=None):
    m = LogisticRegression(C=g, class_weight="balanced", max_iter=5000).fit(Ztr, ytr)
    return m.decision_function(Zte)


def fs_massmean(Ztr, ytr, Zte, _g, _t=None):
    w = Ztr[ytr == 1].mean(0) - Ztr[ytr == 0].mean(0)
    return Zte @ w


def fs_hdc(Htr, ytr, Hte, _g, _t=None):
    """묶기로 프로토타입 → cos 차. ±1 벡터라 cos = dot/D."""
    pr = sign01(Htr[ytr == 1].astype(np.int32).sum(0)).astype(np.float32)
    ps = sign01(Htr[ytr == 0].astype(np.int32).sum(0)).astype(np.float32)
    return (Hte.astype(np.float32) @ (pr - ps)) / DIM


def fs_ridge(Ztr, _ytr, Zte, g, t=None):
    """연속 max_drop 회귀 → 점수 = −예측(max_drop 큼 = spurious)."""
    return -Ridge(alpha=g).fit(Ztr, t).predict(Zte)


SPEC = {"A": ("pca", fs_logistic, CS), "B": ("raw", fs_massmean, (None,)),
        "C": ("hdc", fs_hdc, (None,)), "D": ("hdcseg", fs_hdc, (None,)),
        "A-ridge": ("pca", fs_ridge, ALPHAS), "B-ridge": ("raw", fs_ridge, ALPHAS)}


def nested(y, layers, plan, fit_score, grid, tgt=None):
    """`nested_scores` 와 같은 구조. (층, grid) 를 폴드 내부 CV 의 AUC 로 고른다."""
    score, picks = np.zeros(len(y)), []
    for fi, (tr, te) in enumerate(plan["outer"]):
        inner = plan["inner"][fi]
        best, bk = -1.0, (layers[0], grid[0])
        if inner and len(set(y[tr].tolist())) > 1:
            for l in layers:
                Ztr, _ = plan["proj"][(fi, l)]
                for g in grid:
                    si, ok = np.zeros(len(tr)), True
                    for itr, ite in inner:
                        if len(set(y[tr][itr].tolist())) < 2:
                            ok = False
                            break
                        si[ite] = fit_score(Ztr[itr], y[tr][itr], Ztr[ite], g,
                                            None if tgt is None else tgt[tr][itr])
                    if not ok:
                        continue
                    aa = roc_auc_score(y[tr], si)
                    if aa > best:
                        best, bk = aa, (l, g)
        l, g = bk
        picks.append((int(l), g))
        Ztr, Zte = plan["proj"][(fi, l)]
        if len(set(y[tr].tolist())) < 2:
            score[te] = 0.0
            continue
        score[te] = fit_score(Ztr, y[tr], Zte, g, None if tgt is None else tgt[tr])
    return score, picks


# ---------------------------------------------------------------- null

def perm_idx(y, perms):
    """**행 인덱스 순열.** 네 방법이 같은 추출을 쓰고, ridge 는 (y, max_drop) 쌍이 함께 섞인다.
    소수 클래스 2 미만이 되는 순열은 #77 과 같은 규칙으로 건너뛴다."""
    rng = np.random.default_rng(0)
    keep = []
    for _ in range(perms):
        pi = rng.permutation(len(y))
        if int(min(np.bincount(y[pi]))) < 2:
            continue
        keep.append(pi)
    return keep


def _one(pi, y, layers, plan, fit_score, grid, tgt, logw):
    yp = y[pi]
    tp = None if tgt is None else tgt[pi]
    sp, _ = nested(yp, layers, plan, fit_score, grid, tp)
    return roc_auc_score(yp, sp), roc_auc_score(yp, resid(sp, logw))


def band(y, layers, plan, fit_score, grid, tgt, words, perms, jobs):
    pis = perm_idx(y, perms)
    logw = np.log(np.maximum(1, words))
    if jobs and jobs != 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(_one)(pi, y, layers, plan, fit_score, grid, tgt, logw) for pi in pis)
    else:
        res = [_one(pi, y, layers, plan, fit_score, grid, tgt, logw) for pi in pis]
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


# ---------------------------------------------------------------- 데이터

def maxdrop(files, pidx):
    """pidx → 연속 max_drop (gen_k==8 만). 두 파일에 같은 pidx 가 있으면 멈춘다."""
    m = {}
    for f in files:
        for r in json.loads((LG / f).read_text(encoding="utf-8")):
            if int(r.get("gen_k", 0)) != 8 or r.get("max_drop") is None:
                continue
            q = int(r["pidx"])
            if q in m:
                raise SystemExit(f"pidx {q} 가 두 라벨 파일에 있다 — 묶음을 확인하라")
            m[q] = float(r["max_drop"])
    miss = [q for q in pidx if q not in m]
    return (None, miss) if miss else (np.array([m[q] for q in pidx]), [])


def load_model(npz, segnpz):
    z = np.load(P / npz, allow_pickle=True)
    layers = [int(x) for x in z["layers"]]
    k8 = np.asarray(z["gen_k"], int) == 8
    y = np.asarray(z["y"], int)[k8]
    words = np.asarray(z["words"], float)[k8]
    pidx = [int(x) for x in np.asarray(z["pidx"])[k8]]
    #  **dtype 을 올리지 않는다.** `probe_gen_eval.py` 는 npz 의 float32 를 그대로 넘기므로
    #  float64 로 바꾸면 PCA·로지스틱이 소수 넷째 자리에서 달라져 A 의 재현 게이트가 걸린다.
    feats = {l: z[f"h{l}"][k8] for l in layers}
    segs = None
    sp = P / segnpz
    if sp.is_file():
        s = np.load(sp, allow_pickle=True)
        if [int(x) for x in s["pidx"]] != [int(x) for x in np.asarray(z["pidx"])]:
            raise SystemExit(f"{segnpz} 의 pidx 순서가 {npz} 와 다르다")
        segs = {l: s[f"seg{l}"][k8] for l in layers}
    return z, k8, y, words, pidx, feats, segs, layers


# ---------------------------------------------------------------- 검증

def verify_frame():
    """일반화한 nested 루프가 `probe_final_eval.nested_scores` 와 같은 점수를 내는지 계측.
    **실제 라벨·특징을 쓰지 않는다** — 합성 난수로 두 구현을 맞댄다."""
    rng = np.random.default_rng(1)
    n, d, layers = 30, 256, [1, 2, 3, 4]
    fs = {l: rng.standard_normal((n, d)) for l in layers}
    y = np.array([1] * 20 + [0] * 10)
    pl = make_plan(fs, y, layers, 5, 0)
    s1, p1 = nested_scores(y, layers, pl)
    s2, p2 = nested(y, layers, pl, fs_logistic, CS)
    ok = np.allclose(s1, s2) and p1 == p2
    print(f"틀 검증 · 최대 절대차 {np.max(np.abs(s1 - s2)):.3e} · 고른 (층,C) 일치 {p1 == p2}")
    print("**동일**" if ok else "**다르다 — 이 틀을 쓰지 말 것**")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--verify-frame", action="store_true")
    a = ap.parse_args()
    if a.verify_frame:
        raise SystemExit(0 if verify_frame() else 1)

    #  재현 게이트 (사전등록): A 는 probe_gen_report_supp_w2048 의 값과 같아야 한다.
    GATE = {"Skywork": 0.981, "Olmo": 0.762, "DeepSeek": 0.874}

    out(f"#CMP4 — 분류기 4종 비교 · 창 {WIN} · null {a.perms}회 · HDC D={DIM} · 구간 {NSEG}")
    out("같은 행·같은 폴드(`make_plan` 인덱스 재사용)·같은 null 추출(행 인덱스 순열)")
    out()
    res = {k: {} for k in METHODS + RIDGE}
    hi = {k: {} for k in METHODS + RIDGE}
    res_r, hi_r = {k: {} for k in METHODS + RIDGE}, {k: {} for k in METHODS + RIDGE}
    base = {k: {} for _, k in BASELINES}
    gate_ok = True

    for alias, npz, segnpz, mdfiles in MODELS:
        z, k8, y, words, pidx, feats, segs, layers = load_model(npz, segnpz)
        out(f"## {alias} · {npz} · n {len(y)} (robust {int(y.sum())} / spurious {int((1 - y).sum())})"
            f"{'' if segs is not None else '  ← 16구간 npz 없음 → D 건너뜀'}")
        plan = make_plan(feats, y, layers, 5, 0)
        if plan is None:
            out("   폴드 생성 불가 → 판정 불가")
            out()
            continue
        tgt, miss = (None, ["원천 없음"]) if mdfiles is None else maxdrop(mdfiles, pidx)
        if tgt is None:
            out(f"   연속 max_drop **원천 없음** → ridge 두 칸 판정 불가 ({miss[:5]})")

        plans = {}
        for key in METHODS + RIDGE:
            kind, fit, grid = SPEC[key]
            if key == "D" and segs is None:
                continue
            if key in RIDGE and tgt is None:
                continue
            if kind not in plans:
                plans[kind] = prep(kind, plan, feats, segs, layers)
            s, picks = nested(y, layers, plans[kind], fit, grid, tgt if key in RIDGE else None)
            nr, nd = band(y, layers, plans[kind], fit, grid, tgt if key in RIDGE else None,
                          words, a.perms, a.jobs)
            au, rs = auc_of(y, s), auc_of(y, resid(s, np.log(np.maximum(1, words))))
            h, hr = float(np.percentile(nr, 95)), float(np.percentile(nd, 95))
            res[key][alias], hi[key][alias] = au, h
            res_r[key][alias], hi_r[key][alias] = rs, hr
            mark = " ▲" if au > h else ""
            out(f"   {key:<8} AUC **{au:.3f}** · null 95% [{np.percentile(nr, 5):.2f}, {h:.2f}]{mark}"
                f" · 길이잔차 {rs:.3f} (상단 {hr:.2f}) · bacc {bacc_of(y, s):.3f}")
            out(f"            고른 (층, 하이퍼) {picks}")
            if key == "A":
                g = GATE[alias]
                same = abs(au - g) < 0.0015
                gate_ok &= same
                out(f"            재현 게이트: 기존 기록 {g:.3f} vs 지금 {au:.3f} → "
                    f"{'일치' if same else '**어긋남**'}")
        for nm, kk in BASELINES:
            X = np.asarray(z[kk], float)[k8].reshape(-1, 1)
            if np.std(X) == 0:
                out(f"   {nm}: 분산 0 → 판정 불가")
                base[kk][alias] = float("nan")
                continue
            pb = make_plan({0: X}, y, [0], 5, 0)
            sb, _ = nested_scores(y, [0], pb)
            base[kk][alias] = auc_of(y, sb)
            out(f"   {nm}: AUC {base[kk][alias]:.3f}")
        out()

    if not gate_ok:
        out("## 재현 게이트 **실패** — A 가 기존 숫자를 재현하지 않는다.")
        out("   사전등록대로 B·C·D 를 판정에 쓰지 않고 멈춘다(위 숫자는 진단용).")
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise SystemExit(f"재현 게이트 실패 — {OUT}")

    def mean_over(key, who):
        v = [res[key][m] for m in who if m in res[key]]
        return float(np.mean(v)) if len(v) == len(who) else float("nan")

    all3 = [m for m, *_ in MODELS]
    two = ["Skywork", "DeepSeek"]
    out("## 주지표 — 모델평균 AUC · 채택 기준 (지휘: A 대비 +0.05 이상 **그리고** 자기 null 상단 초과)")
    out()
    out("| 방법 | 모델 | 모델평균 AUC | A 대비 | null 상단 평균 | 길이잔차(상단) | 채택 |")
    out("|---|---|---:|---:|---:|---:|---|")
    verdict = {}
    for key in METHODS + RIDGE:
        who = two if key in RIDGE else all3
        if any(m not in res[key] for m in who):
            out(f"| {key} | — | — | — | — | — | **판정 불가**(원천/특징 없음) |")
            verdict[key] = "판정 불가"
            continue
        mv = mean_over(key, who)
        ref = mean_over("A", who)
        hv = float(np.mean([hi[key][m] for m in who]))
        rv = float(np.mean([res_r[key][m] for m in who]))
        rh = float(np.mean([hi_r[key][m] for m in who]))
        c1, c2 = (mv - ref) >= 0.05, mv > hv
        ok = "**채택**" if (c1 and c2) else f"미채택 (기준1 {'O' if c1 else 'X'} · 기준2 {'O' if c2 else 'X'})"
        if key == "A":
            ok = "기준"
        verdict[key] = ok
        out(f"| {key} | {'2모델' if key in RIDGE else '3모델'} | **{mv:.3f}** | "
            f"{mv - ref:+.3f} | {hv:.2f} | {rv:.3f} ({rh:.2f}) | {ok} |")
    out()
    out("모델별 AUC:")
    for key in METHODS + RIDGE:
        if res[key]:
            out(f"   {key:<8} " + " · ".join(f"{m} {res[key][m]:.3f}" for m in all3 if m in res[key]))
    out()
    out("기준선 모델평균 (판정 미사용 · 병기):")
    for nm, kk in BASELINES:
        v = [x for x in base[kk].values() if not np.isnan(x)]
        out(f"   {nm}: {np.mean(v):.3f}" if v else f"   {nm}: 전부 판정 불가")
    out()
    out("## 판정")
    adopted = [k for k in METHODS[1:] + RIDGE if verdict.get(k, "").startswith("**채택**")]
    out(f"   채택된 방법: {adopted if adopted else '**없음 — 네 방법 모두 A 를 기준만큼 넘지 못했다**'}")
    out(f"   (ridge 두 칸의 A 대비 상대값은 **같은 2모델 A 이진판** {mean_over('A', two):.3f} 이다 "
        f"— Olmo 는 연속 max_drop 원천이 없다)")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
