"""#ORG-TRAIN — **주최측 라벨로 직접 학습**. 공식 33문제 × 3모델 · 문제단위 LOPO · 비용 $0.

지금까지 모든 프로브는 **우리 자체 라벨**로 학습했다. #ORG-LABEL 시험1 이 그 라벨의 한계를
정량화했고(주최측 max_drop 격자 0.01 ≈ 100샘플 vs 우리 k=8 의 0.125 — 12~25배 거칠다),
#TEMP-SENS 가 재현성 약 2/3 을 쟀다. 이 스크립트가 가리는 것은 하나다:
**'거친 자체 라벨' 이 원인이었나, 아니면 이 특징·이 규모로는 애초에 안 되는가.**

설계는 사전등록(CLAUDE.md `#ORG-TRAIN`)에 전부 박혀 있다. 요점:
  * 문제 단위 **leave-one-problem-out 33폴드** — 같은 문제가 train·test 에 동시에 안 들어간다.
    풀링에서도 **그 문제의 3모델 행을 한꺼번에** 뺀다.
  * 층 **L//2 고정**(org33 추출이 그 층만 떴다 → 층 다중비교 0) · PCA **min(24, n_train−1)**
  * C 는 **폴드 학습분 내부 CV** · **임계도 학습 폴드 안에서만**(두 클래스 중앙값의 중점)
  * null 1000 은 라벨을 섞고 **LOPO 파이프라인 전체**를 다시 돈다
  * 주지표 = **풀링 AUC**(점수 수준 풀링 — 특징 차원이 3584/4096 로 달라 못 이어 붙인다)
  * 채택 = **길이 기준선 +0.05 이상 AND 자기 null 95% 상단 초과**

사용:
  .venv/Scripts/python.exe analysis/org_train_eval.py --verify-parallel   # 직렬==병렬 확인
  .venv/Scripts/python.exe analysis/org_train_eval.py --perms 1000 --jobs 10
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
FEAT = ROOT / "experiments/results/probe"
OUT = ROOT / "experiments/results/org_train_report.txt"
MODELS = {"Skywork/Skywork-OR1-Math-7B": "Skywork-OR1-Math-7B",
          "allenai/Olmo-3-7B-Think": "Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B"}
CS = (0.003, 0.01, 0.03, 0.1, 0.3)
NPC_MAX = 24
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


# ───────────────────────── 지표 ─────────────────────────

def auc_of(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


def mid_thr(s, y):
    """학습 폴드 안에서만 정하는 임계 = 두 클래스 중앙값의 중점 (h15 관행)."""
    a, b = np.median(s[y == 1]), np.median(s[y == 0])
    return float((a + b) / 2.0)


# ───────────────────────── 프로브 A: LOPO ─────────────────────────

def prep_proj(X, n):
    """폴드별 (학습 투영, 테스트 투영) 을 **한 번만** 만든다.

    StandardScaler·PCA 는 **y 를 보지 않으므로** null 1000 반복에서 다시 만들 필요가 없다 —
    #77·#CMP4 가 쓴 그 구조다. 이렇게 하면 null 반복에서 다시 도는 것은 **로지스틱 적합과
    임계**뿐이고, 통계량은 글자 그대로 같다(`--verify-prep` 가 합성 난수로 확인한다).
    """
    pr = []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        npc = min(NPC_MAX, len(tr) - 1)
        st = StandardScaler().fit(X[tr])
        Ztr, Zte = st.transform(X[tr]), st.transform(X[i:i + 1])
        pc = PCA(n_components=npc, random_state=0).fit(Ztr)
        pr.append((pc.transform(Ztr), pc.transform(Zte), tr))
    return pr


def lopo_probe(proj, y):
    """반환 (폴드밖 점수, 폴드밖 예측bool, 고른 C 목록). C·임계 전부 학습분에서만."""
    n = len(y)
    sc = np.zeros(n)
    pr = np.zeros(n, dtype=bool)
    picked = []
    for i in range(n):
        Ptr, Pte, tr = proj[i]
        ytr = y[tr]
        # 내부 CV 로 C 선택 (AUC 기준 — #77·#CMP4 와 같은 규약)
        best, bc = -1.0, CS[0]
        k = min(4, int(min(np.bincount(ytr, minlength=2)[:2])))
        if k >= 2:
            inner = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
            for C in CS:
                s_in = np.zeros(len(tr))
                for a, b in inner.split(Ptr, ytr):
                    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(
                        Ptr[a], ytr[a])
                    s_in[b] = m.decision_function(Ptr[b])
                v = auc_of(ytr, s_in)
                if v > best:
                    best, bc = v, C
        picked.append(bc)
        m = LogisticRegression(C=bc, max_iter=2000, class_weight="balanced").fit(Ptr, ytr)
        s_tr = m.decision_function(Ptr)
        sc[i] = float(m.decision_function(Pte)[0])
        pr[i] = sc[i] >= mid_thr(s_tr, ytr)
    return sc, pr, picked


def lopo_len(w, y):
    """B 기준선: 점수는 −log(단어수) 로 고정(적합 없음) · **임계만** 학습 폴드에서 다시 낸다."""
    n = len(y)
    s = -w
    pr = np.zeros(n, dtype=bool)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        pr[i] = s[i] >= mid_thr(s[tr], y[tr])
    return s, pr


# ───────────────────────── 한 번의 LOPO 전체 (null 반복 단위) ─────────────────────────

def one_run(Xs, w, y):
    """세 모델 프로브 + 길이 기준선을 한 번 돌려 풀링·모델별 지표를 돌려준다.
    `Xs` 값은 `prep_proj` 산출(폴드별 투영)이다."""
    per, oof_s, oof_y, oof_p = {}, [], [], []
    for name, X in Xs.items():
        s, p, picked = lopo_probe(X, y)
        per[name] = dict(auc=auc_of(y, s), acc=float((p == (y == 1)).mean()),
                         tr=float(p.mean()), picked=picked, s=s)
        oof_s.append(s)
        oof_y.append(y)
        oof_p.append(p)
    ps, py, pp = np.concatenate(oof_s), np.concatenate(oof_y), np.concatenate(oof_p)
    avg = np.mean(np.stack([per[k]["s"] for k in Xs]), axis=0)
    ls, lp = lopo_len(w, y)
    return dict(
        pool_auc=auc_of(py, ps), pool_acc=float((pp == (py == 1)).mean()),
        pool_tr=float(pp.mean()),
        avg_auc=auc_of(y, avg), avg_acc=float(((avg >= mid_thr(avg, y)) == (y == 1)).mean()),
        len_auc=auc_of(y, ls), len_acc=float((lp == (y == 1)).mean()), len_tr=float(lp.mean()),
        per=per)


def _worker(args):
    Xs, w, y, seeds = args
    res = []
    for sd in seeds:
        yp = np.random.default_rng(sd).permutation(y)
        r = one_run(Xs, w, yp)
        res.append((r["pool_auc"], r["pool_acc"], r["avg_auc"], r["len_auc"], r["len_acc"],
                    [r["per"][k]["auc"] for k in Xs], [r["per"][k]["acc"] for k in Xs]))
    return res


def null_runs(Xs, w, y, perms, jobs):
    """순열 수열을 **미리 같은 순서로** 만들어 나눈다 → 직렬·병렬이 구성상 같다."""
    seeds = list(range(1, perms + 1))
    if jobs <= 1:
        return _worker((Xs, w, y, seeds))
    import multiprocessing as mp
    chunks = [seeds[i::jobs] for i in range(jobs)]
    order = {sd: i for i, sd in enumerate(seeds)}
    with mp.Pool(jobs) as pool:
        got = pool.map(_worker, [(Xs, w, y, c) for c in chunks])
    flat = [(sd, r) for c, rs in zip(chunks, got) for sd, r in zip(c, rs)]
    flat.sort(key=lambda x: order[x[0]])
    return [r for _, r in flat]


def _naive_probe(X, y):
    """`prep_proj` 도입 **전**의 경로. 폴드마다 scaler·PCA 를 그 자리에서 적합한다.
    `--verify-prep` 가 이것과 새 경로를 맞댄다(합성 난수, 실제 라벨·특징 미사용)."""
    n = len(y)
    sc, pr, picked = np.zeros(n), np.zeros(n, dtype=bool), []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        ytr = y[tr]
        npc = min(NPC_MAX, len(tr) - 1)
        st = StandardScaler().fit(X[tr])
        Ztr, Zte = st.transform(X[tr]), st.transform(X[i:i + 1])
        pc = PCA(n_components=npc, random_state=0).fit(Ztr)
        Ptr, Pte = pc.transform(Ztr), pc.transform(Zte)
        best, bc = -1.0, CS[0]
        k = min(4, int(min(np.bincount(ytr, minlength=2)[:2])))
        if k >= 2:
            inner = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
            for C in CS:
                s_in = np.zeros(len(tr))
                for a, b in inner.split(Ptr, ytr):
                    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(
                        Ptr[a], ytr[a])
                    s_in[b] = m.decision_function(Ptr[b])
                v = auc_of(ytr, s_in)
                if v > best:
                    best, bc = v, C
        picked.append(bc)
        m = LogisticRegression(C=bc, max_iter=2000, class_weight="balanced").fit(Ptr, ytr)
        sc[i] = float(m.decision_function(Pte)[0])
        pr[i] = sc[i] >= mid_thr(m.decision_function(Ptr), ytr)
    return sc, pr, picked


def verify_prep():
    """전처리 재사용이 통계량을 바꾸지 않는지 확인한다 — 주장하지 않고 계측한다."""
    rng = np.random.default_rng(11)
    y = np.array([1] * 13 + [0] * 9)
    ok = True
    for d in (40, 60):
        X = rng.standard_normal((22, d))
        X[y == 1, :5] += 0.8
        s1, p1, c1 = _naive_probe(X, y)
        s2, p2, c2 = lopo_probe(prep_proj(X, len(y)), y)
        dd = float(np.abs(s1 - s2).max())
        same = dd == 0.0 and (p1 == p2).all() and c1 == c2
        ok &= same
        print(f"[검증] d={d} 점수 최대절대차 {dd:.3e} · 예측 일치 {int((p1 == p2).sum())}/{len(y)} · "
              f"고른 C 일치 {c1 == c2} · {'동일' if same else '**다르다**'}", flush=True)
    return ok


def verify_parallel():
    """합성 난수로 직렬 null 과 병렬 null 이 같은지 확인한다(실제 라벨·특징 미사용)."""
    rng = np.random.default_rng(7)
    y = np.array([1] * 12 + [0] * 8)
    Xs = {"m1": prep_proj(rng.standard_normal((20, 40)), 20),
          "m2": prep_proj(rng.standard_normal((20, 50)), 20)}
    w = rng.standard_normal(20) + 4
    a = null_runs(Xs, w, y, 6, 1)
    b = null_runs(Xs, w, y, 6, 3)
    fa = np.array([[x[0], x[1], x[2], x[3], x[4]] for x in a])
    fb = np.array([[x[0], x[1], x[2], x[3], x[4]] for x in b])
    d = float(np.abs(fa - fb).max())
    print(f"[검증] 직렬 vs 병렬 최대 절대차 {d:.3e} · {'동일' if d == 0 else '**다르다**'}", flush=True)
    return d == 0


# ───────────────────────── 본 실행 ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--verify-parallel", action="store_true")
    ap.add_argument("--verify-prep", action="store_true")
    a = ap.parse_args()
    if a.verify_prep:
        raise SystemExit(0 if verify_prep() else 1)
    if a.verify_parallel:
        raise SystemExit(0 if verify_parallel() else 1)

    from h21_uncertainty import problems
    lab = {r["problem"]: r["y"] for r in problems()}

    Xs, ref = {}, None
    for mid, alias in MODELS.items():
        d = np.load(FEAT / f"{alias}_org33.npz", allow_pickle=True)
        pt = [str(x) for x in d["problems"]]
        if ref is None:
            ref = pt
        elif pt != ref:
            raise SystemExit(f"{alias} 의 문제 순서가 다르다 — 풀링이 어긋난다")
        Xs[alias] = prep_proj(d["X"].astype(np.float64), len(pt))
    y = np.array([lab[t] for t in ref])
    w = np.array([math.log(max(1, len(t.split()))) for t in ref])

    out("#ORG-TRAIN — 주최측 라벨로 직접 학습 (사전등록 1회 · 비용 $0)")
    out(f"   특징 창 2048 · 층 L//2 고정 · 라벨 **주최측 is_robust** 문제단위 {len(y)} "
        f"(robust {int(y.sum())} / spurious {int((1 - y).sum())})")
    out(f"   문제 단위 LOPO {len(y)}폴드 · PCA min({NPC_MAX}, n_train−1) = "
        f"{min(NPC_MAX, len(y) - 2)} · C {CS} 내부 CV · 임계는 학습 폴드 안에서만")
    out(f"   셔플 null {a.perms}회 (LOPO 파이프라인 전체 재실행) · jobs {a.jobs}")
    out()

    r = one_run(Xs, w, y)
    nul = null_runs(Xs, w, y, a.perms, a.jobs)
    pa = np.array([x[0] for x in nul])
    pc = np.array([x[1] for x in nul])
    av = np.array([x[2] for x in nul])
    la = np.array([x[3] for x in nul])
    lc = np.array([x[4] for x in nul])
    mper_a = np.array([x[5] for x in nul])
    mper_c = np.array([x[6] for x in nul])

    def hi(v):
        return float(np.percentile(v, 97.5))

    def lo(v):
        return float(np.percentile(v, 2.5))

    names = list(Xs)
    out("## 표 — 주지표는 **풀링 AUC** 1개")
    out(f"   {'항목':<30}{'AUC':>8}{'null 95%':>18}{'우연이상':>9}{'정확도':>8}{'True비율':>9}")

    def row(label, auc_v, nv, acc, accnul, tr):
        mark = "예" if auc_v > hi(nv) else "아니다"
        out(f"   {label:<30}{auc_v:>8.3f}   [{lo(nv):.2f}, {hi(nv):.2f}]{mark:>9}"
            f"{acc:>8.3f}{tr:>9.3f}"
            + (f"   (정확도 null 상단 {hi(accnul):.3f})" if accnul is not None else ""))

    row("A 프로브 · 풀링 99행", r["pool_auc"], pa, r["pool_acc"], pc, r["pool_tr"])
    row("A 프로브 · 모델평균 33행", r["avg_auc"], av, r["avg_acc"], None, float("nan"))
    for i, nm in enumerate(names):
        row(f"A 프로브 · {nm[:22]}", r["per"][nm]["auc"], mper_a[:, i],
            r["per"][nm]["acc"], mper_c[:, i], r["per"][nm]["tr"])
    row("B 길이 기준선 h15C", r["len_auc"], la, r["len_acc"], lc, r["len_tr"])
    out(f"   {'C 상수 전부True':<30}{0.5:>8.3f}{'정의상 0.5':>18}{'—':>9}"
        f"{float((y == 1).mean()):>8.3f}{1.0:>9.3f}")
    out()

    out("## 채택 판정 (사전등록: 둘 다 만족해야 채택)")
    c1 = r["pool_auc"] >= r["len_auc"] + 0.05
    c2 = r["pool_auc"] > hi(pa)
    out(f"   1. 풀링 AUC {r['pool_auc']:.3f} ≥ 길이 기준선 {r['len_auc']:.3f} + 0.05 = "
        f"{r['len_auc'] + 0.05:.3f} → {'통과' if c1 else '**미달**'}")
    out(f"   2. 풀링 AUC {r['pool_auc']:.3f} > 자기 null 95% 상단 {hi(pa):.3f} → "
        f"{'통과' if c2 else '**미달**'}")
    out(f"   → **{'채택' if (c1 and c2) else '미채택'}**")
    out()
    out("## 고른 C (폴드별, 과적합 진단)")
    for nm in names:
        pk = r["per"][nm]["picked"]
        u = {c: pk.count(c) for c in sorted(set(pk))}
        out(f"   {nm:<26} {u}")
    out()
    out("## 읽는 법")
    out("   `풀링 99행` 은 (모델,문제) 행이고 **같은 문제단위 라벨이 3번 반복**된다 — 유효 표본은")
    out("   33 에 가깝다. null 도 같은 구조로 섞으므로 그 반복은 밴드 폭에 반영된다.")
    out("   `정확도` 는 불균형(robust 20/33 = 0.606)에서 상수 예측에 유리하므로 **게이트가 아니다**")
    out("   — 사전등록대로 주지표는 AUC 하나다.")
    out("   B 기준선은 점수(−log 단어수)가 적합 없이 고정이므로 AUC 가 LOPO 와 무관하고, **임계만**")
    out("   폴드 학습분에서 다시 낸다. 출하된 h15C 임계는 바로 이 33문제에서 고정한 in-sample")
    out("   값이라 쓰지 않았다.")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
