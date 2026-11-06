"""#77 최종 프로브 평가 — 다모델. **1회만 돌린다.** GPU 0.

사전등록: 지휘 #77(2026-10-08) + **Amendment 1**(지휘 #RULE-B, 철회) + **Amendment 2**(지휘 #AMEND2).
이 파일은 **라벨이 다 도착하기 전에** 쓴 것이다 — 평가 코드를 데이터 본 뒤에 쓰면 측정이 아니라 선택이다.

### 고정된 설계 (바꾸면 기록에 '탐색' 으로 적는다)
* 특징 = 문제 텍스트 forward 1회의 hidden state, **문제 본문 토큰 평균 풀링**(`experiments/probe_pilot.py`).
* 층 후보 = `hidden_states` 인덱스 L//4 · L//2 · 3L//4 · L — **폴드 내부 CV 로만** 고른다.
* PCA **24성분 고정**, **각 폴드 학습분으로만** 적합. 분류기 = 로지스틱 L2,
  `class_weight="balanced"`, C ∈ {0.003, 0.01, 0.03, 0.1, 0.3} — 폴드 내부 선택.
* **주라벨 = `label`**(max_drop 임계) · **주분석은 `gen_k == 8` 라벨만** (Amendment 2).
  민감도 = `gen_k == 4` 라벨 · `label_fin` · `label_major`(과반; 양성 0 이면 **'적용 불가'** 로 적는다).
* 주지표 = **주라벨 모델별 AUC 의 모델평균**. 보조 = balanced accuracy · 풀링 · 외부 테스트 · 분리력.

### 통과 기준 (셋 다 만족해야 주장한다)
1. 주지표가 **셔플 null 95% 상단**을 넘는다 (null = 라벨을 섞고 **같은 nested CV 재실행**).
2. **log(단어수) 잔차 AUC** 도 null 95% 상단을 넘는다.
3. 대조 라벨(`label_fin`)의 AUC 가 **0.5 미만으로 뒤집히지 않는다**.
모델×문제 성분을 주장하려면 추가로 **모델 간 분리력 > 0.5, 이항검정 p < 0.05**(쌍 20개 미만이면 판정 불가).

### null 을 어떻게 1000회 도는가 (구현 메모)
"라벨을 섞고 같은 nested CV 재실행" 을 글자 그대로 하면 층·C 탐색까지 1000번 돌아 로컬에서 불가능하다.
그런데 **표준화와 PCA 는 y 를 보지 않는다** — 그래서 (층, 폴드)별 투영 행렬을 **한 번만** 만들어 두고
null 반복에서는 **로지스틱만 다시 적합**한다. 결과는 글자 그대로의 재실행과 **같다**(층·C 선택도
null 라벨로 다시 하므로 선택 편향까지 포함된다). 24차원 로지스틱이라 1000회가 분 단위로 끝난다.

사용:
  .venv/Scripts/python.exe analysis/probe_final_eval.py --sets Skywork=...k8.npz Olmo=... DeepSeek=...
  (--sets 는 `별칭=npz경로` 꼴. --perms 로 null 횟수, --k 로 주분석 gen_k 를 바꿀 수 있다.)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "probe_final_report.txt"
CS = (0.003, 0.01, 0.03, 0.1, 0.3)
NPC = 24


# ---------------------------------------------------------------- 폴드·투영 사전계산 (y 를 보지 않는다)

def make_plan(feats, y, layers, folds=5, seed=0):
    """**폴드를 한 번 정하고 (층, 폴드)별 투영을 한 번 적합한다.**

    왜 폴드를 고정하나: 순열검정에서 폴드를 매번 라벨로 다시 나누면 투영도 매번 다시 적합해야 해서
    1000회가 로컬에서 불가능하다. 폴드 구조를 고정한 채 라벨만 섞는 것이 순열검정의 표준 설계이고
    ("같은 CV 구조에서 우연한 라벨 배치로 이 AUC 가 나올 수 있나"), 표준화·PCA 는 y 를 보지 않으므로
    투영 재사용에 누출이 없다. 층·C 선택은 null 라벨로도 매번 다시 하므로 **선택 편향은 포함된다.**"""
    kmin = int(min(np.bincount(y))) if len(set(y.tolist())) > 1 else 0
    if kmin < 2:
        return None
    outer = list(StratifiedKFold(min(folds, kmin), shuffle=True, random_state=seed).split(
        feats[layers[0]], y))
    plan = {"outer": outer, "inner": [], "proj": {}, "iproj": {}}
    for fi, (tr, te) in enumerate(outer):
        ykmin = int(min(np.bincount(y[tr])))
        inner = list(StratifiedKFold(min(4, max(2, ykmin)), shuffle=True, random_state=seed).split(
            feats[layers[0]][tr], y[tr])) if ykmin >= 2 else []
        plan["inner"].append(inner)
        for l in layers:
            X = feats[l]
            sc = StandardScaler().fit(X[tr])
            Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
            if X.shape[1] > NPC:
                pca = PCA(n_components=min(NPC, len(tr) - 1, X.shape[1]), random_state=0).fit(Ztr)
                Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
            plan["proj"][(fi, l)] = (Ztr, Zte)
    return plan


def nested_scores(y, layers, plan):
    """고정된 폴드·투영 위에서 (층, C)를 폴드 내부 CV 로 고르고 폴드 밖 점수를 모은다."""
    score = np.zeros(len(y))
    picks = []
    for fi, (tr, te) in enumerate(plan["outer"]):
        inner = plan["inner"][fi]
        best, bestkey = -1.0, (layers[0], CS[0])
        if inner and len(set(y[tr].tolist())) > 1:
            for l in layers:
                Ztr, _ = plan["proj"][(fi, l)]
                for C in CS:
                    sc_in, ok = np.zeros(len(tr)), True
                    for itr, ite in inner:
                        if len(set(y[tr][itr].tolist())) < 2:
                            ok = False
                            break
                        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
                        m.fit(Ztr[itr], y[tr][itr])
                        sc_in[ite] = m.decision_function(Ztr[ite])
                    if not ok:
                        continue
                    aa = roc_auc_score(y[tr], sc_in)
                    if aa > best:
                        best, bestkey = aa, (l, C)
        l, C = bestkey
        picks.append((int(l), float(C)))
        Ztr, Zte = plan["proj"][(fi, l)]
        if len(set(y[tr].tolist())) < 2:
            score[te] = 0.0
            continue
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        m.fit(Ztr, y[tr])
        score[te] = m.decision_function(Zte)
    return score, picks


def _perm_list(y, perms):
    """순열을 **미리 같은 순서로** 만든다. 직렬·병렬이 똑같은 수열을 쓰게 하는 유일한 방법이다.
    소수 클래스가 2 미만이 되는 순열은 직렬판과 **같은 규칙으로** 건너뛴다(그 자리를 비운다)."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(perms):
        yp = rng.permutation(y)
        if int(min(np.bincount(yp))) < 2:
            continue
        out.append(yp)
    return out


def _one_perm(yp, layers, plan, logw):
    sp, _ = nested_scores(yp, layers, plan)
    return roc_auc_score(yp, sp), roc_auc_score(yp, resid(sp, logw))


def null_band(y, layers, plan, words, perms, jobs=1):
    """셔플 null. **`jobs` 는 구현 세부이고 통계량을 바꾸지 않는다** — 순열 수열이 시드로 고정돼
    있고 perm 끼리 독립이므로 직렬과 병렬의 결과 배열이 글자까지 같다(`--verify-parallel` 로 확인).
    왜 필요했나: perm 하나에 로지스틱 **400회**(층 4 × C 5 × 내부폴드 4 × 외부폴드 5)가 들어가
    실측 perm 당 벽시계 7.5초였고, 3모델 × 1000회면 **6.3시간**이었다."""
    yps = _perm_list(y, perms)
    logw = np.log(np.maximum(1, words))
    if jobs and jobs != 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(_one_perm)(yp, layers, plan, logw) for yp in yps)
    else:
        res = [_one_perm(yp, layers, plan, logw) for yp in yps]
    return [r[0] for r in res], [r[1] for r in res]


# ---------------------------------------------------------------- 지표

def auc_of(y, s):
    return roc_auc_score(y, s) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def bacc_of(y, s):
    return balanced_accuracy_score(y, (np.asarray(s) >= np.median(s)).astype(int))


def resid(x, basis):
    X = np.column_stack([np.ones(len(x)), basis])
    b, *_ = np.linalg.lstsq(X, np.asarray(x, float), rcond=None)
    return np.asarray(x, float) - X @ b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, metavar="ALIAS=NPZ")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--k", type=int, default=8, help="주분석에 쓸 gen_k (Amendment 2: 8)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1,
                    help="null 순열을 돌릴 프로세스 수. **구현 세부이고 통계량을 바꾸지 않는다** "
                         "(순열 수열이 시드 고정). 1 = 종전 직렬 경로.")
    ap.add_argument("--external", nargs="*", default=None, metavar="ALIAS=NPZ",
                    help="외부 테스트(공식 train-main-v2 33문제) 특징. #77 이 **보조**로 요구한 항목 — "
                         "학습에는 절대 쓰지 않고, 자체 라벨 전체로 적합한 프로브를 여기에 한 번 적용한다.")
    ap.add_argument("--verify-parallel", action="store_true",
                    help="합성 데이터로 직렬 vs 병렬 null 배열이 같은지만 확인하고 끝낸다.")
    a = ap.parse_args()

    if a.verify_parallel:
        #  **실제 라벨·특징을 쓰지 않는다** — 같은 코드 경로에 난수를 넣어 두 구현이 같은 배열을
        #  내는지만 본다. 평가를 두 번 돌리는 것도, 결과를 미리 보는 것도 아니다.
        rng = np.random.default_rng(1)
        n, h, layers = 28, 512, [1, 2, 3, 4]
        fs = {l: rng.standard_normal((n, h)) for l in layers}
        yv = np.array([1] * 15 + [0] * 13)
        words = rng.integers(20, 90, n)
        pl = make_plan(fs, yv, layers, a.folds, 0)
        s1 = null_band(yv, layers, pl, words, 12, 1)
        s2 = null_band(yv, layers, pl, words, 12, max(2, a.jobs))
        same = (np.allclose(s1[0], s2[0]) and np.allclose(s1[1], s2[1])
                and len(s1[0]) == len(s2[0]))
        print(f"직렬 n={len(s1[0])} · 병렬 n={len(s2[0])} · 최대 절대차 "
              f"{max(np.max(np.abs(np.array(s1[0]) - np.array(s2[0]))), np.max(np.abs(np.array(s1[1]) - np.array(s2[1])))):.3e}")
        print("**동일**" if same else "**다르다 — 병렬 경로를 쓰지 말 것**")
        raise SystemExit(0 if same else 1)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("#77 최종 프로브 평가 — 다모델 · 1회 실행")
    out()
    out(f"주라벨 `label`(max_drop) · **주분석 gen_k == {a.k}** (Amendment 2) · null {a.perms}회 "
        f"(라벨 섞고 같은 nested CV 재실행; 표준화·PCA 는 y 무관이라 투영을 재사용한다)")
    out("민감도: gen_k != 주분석 · `label_fin` · `label_major`(과반; 양성 0 이면 '적용 불가')")
    out()

    per_model = {}
    ext_src = {}
    for spec in a.sets:
        alias, _, path = spec.partition("=")
        p = Path(path)
        if not p.is_file():
            out(f"!! {alias}: 파일 없음 {path}")
            continue
        z = np.load(p, allow_pickle=True)
        layers = [int(x) for x in z["layers"]]
        feats = {l: z[f"h{l}"] for l in layers}
        gk = z["gen_k"] if "gen_k" in z.files else np.zeros(len(z["y"]), int)
        sel = (gk == a.k) if (gk == a.k).any() else np.ones(len(z["y"]), bool)
        note = f"gen_k=={a.k} {int(sel.sum())}/{len(sel)}" if (gk == a.k).any() else "gen_k 메타 없음 → 전체"
        y = np.asarray(z["y"], int)[sel]
        words = np.asarray(z["words"], float)[sel]
        fs = {l: feats[l][sel] for l in layers}
        out(f"## {alias} · {p.name} · {note} · robust {int(y.sum())} / spurious {int((1 - y).sum())}")
        if len(set(y.tolist())) < 2 or int(min(np.bincount(y))) < 2:
            out("   **한 클래스뿐이거나 소수 클래스가 2 미만 → 평가 불가**")
            out()
            continue
        plan = make_plan(fs, y, layers, a.folds, 0)
        if plan is None:
            out("   **폴드를 만들 수 없다 → 평가 불가**")
            out()
            continue
        s, picks = nested_scores(y, layers, plan)
        auc, bacc = auc_of(y, s), bacc_of(y, s)
        rauc = auc_of(y, resid(s, np.log(np.maximum(1, words))))
        # null: 라벨을 섞고 같은 nested CV 를 다시 (투영 재사용 — y 무관)
        nl, nlr = null_band(y, layers, plan, words, a.perms, a.jobs)
        lo, hi = (np.percentile(nl, [2.5, 97.5]) if nl else (np.nan, np.nan))
        rlo, rhi = (np.percentile(nlr, [2.5, 97.5]) if nlr else (np.nan, np.nan))
        out(f"   AUC **{auc:.3f}** · null 95% [{lo:.2f}, {hi:.2f}] {'▲' if auc > hi else ''}")
        out(f"   길이잔차 AUC {rauc:.3f} · null [{rlo:.2f}, {rhi:.2f}] {'▲' if rauc > rhi else ''}")
        out(f"   balanced acc {bacc:.3f} · 고른 (층, C) {picks}")
        # 민감도 라벨
        for tag, key in (("label_fin", "y_fin"), ("label_major", "y_major")):
            if key not in z.files:
                continue
            yy = np.asarray(z[key], int)[sel]
            #  -1 = 그 규칙으로는 라벨이 없는 행(중간대역). **그 행만 빼고** 나머지로 센다 —
            #  하나라도 -1 이면 열 전체를 '적용 불가' 로 버리던 옛 동작은 쓸 수 있는 대조를
            #  날린다(#77 통과기준 3 이 label_fin 을 요구하므로 특히 나쁘다). 주분석 실행 전 수정.
            keep = yy >= 0
            if keep.sum() < 2 or len(set(yy[keep].tolist())) < 2:
                out(f"   민감도 {tag}: **적용 불가**(양성 0 또는 값 없음 · "
                    f"robust {int((yy == 1).sum())} / spurious {int((yy == 0).sum())})")
                continue
            out(f"   민감도 {tag}: 같은 점수로 AUC {auc_of(yy[keep], s[keep]):.3f} "
                f"(robust {int((yy == 1).sum())} / spurious {int((yy == 0).sum())}"
                f"{f' · 라벨 없음 {int((~keep).sum())} 제외' if (~keep).any() else ''})")
        ext_src[alias] = ({l: fs[l] for l in layers}, y, layers)
        per_model[alias] = dict(y=y, s=s, auc=auc, rauc=rauc, hi=hi, rhi=rhi,
                                pidx=np.asarray(z["pidx"])[sel])
        out()

    if not per_model:
        out("평가된 모델이 없다.")
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # ---- 주지표: 모델별 AUC 의 모델평균
    out("## 주지표 — 주라벨 모델별 AUC 의 **모델평균**")
    out()
    mean_auc = float(np.mean([v["auc"] for v in per_model.values()]))
    mean_rauc = float(np.mean([v["rauc"] for v in per_model.values()]))
    mean_hi = float(np.mean([v["hi"] for v in per_model.values()]))
    mean_rhi = float(np.mean([v["rhi"] for v in per_model.values()]))
    out(f"   모델평균 AUC **{mean_auc:.3f}** · null 상단 평균 {mean_hi:.2f} "
        f"{'▲ 통과' if mean_auc > mean_hi else '— null 안'}")
    out(f"   모델평균 길이잔차 {mean_rauc:.3f} · null 상단 평균 {mean_rhi:.2f} "
        f"{'▲ 통과' if mean_rauc > mean_rhi else '— null 안'}")
    out(f"   → 통과 기준 1·2: {'**둘 다 통과**' if (mean_auc > mean_hi and mean_rauc > mean_rhi) else '**미달**'}")
    out()

    # ---- 풀링: 모델별 폴드 밖 점수를 모아 (모델, 문제) 전체에서
    out("## 보조 — 풀링 (모델별 폴드 밖 점수를 모아 (모델, 문제) 행 전체)")
    ys = np.concatenate([v["y"] for v in per_model.values()])
    # 모델마다 점수 스케일이 달라 **모델별 z 표준화** 후 합친다(가중치 학습 없음)
    ss = np.concatenate([(v["s"] - v["s"].mean()) / (v["s"].std() or 1) for v in per_model.values()])
    out(f"   AUC {auc_of(ys, ss):.3f} · balanced acc {bacc_of(ys, ss):.3f} · n {len(ys)}")
    out()

    # ---- 모델 간 분리력
    out("## 보조 — 모델 간 분리력 (#77 정의)")
    by_pidx = {}
    for alias, v in per_model.items():
        for pi, yy, sc in zip(v["pidx"], v["y"], v["s"]):
            by_pidx.setdefault(int(pi), {})[alias] = (int(yy), float(sc))
    pairs = ok = 0
    for pi, d in by_pidx.items():
        al = list(d)
        for i in range(len(al)):
            for j in range(i + 1, len(al)):
                (y1, s1), (y2, s2) = d[al[i]], d[al[j]]
                if y1 == y2:
                    continue
                pairs += 1
                rob, spu = (s1, s2) if y1 == 1 else (s2, s1)
                ok += int(rob > spu)
    if pairs < 20:
        out(f"   라벨이 갈리는 쌍 **{pairs}개 < 20 → 판정 불가**(사전등록 규칙). 숫자를 주장하지 않는다.")
    else:
        from math import comb
        p = sum(comb(pairs, t) for t in range(ok, pairs + 1)) / 2 ** pairs
        out(f"   쌍 {pairs}개 · robust 쪽 점수가 더 높은 비율 **{ok / pairs:.3f}** · 이항검정 p {p:.4f} "
            f"{'(유의)' if p < 0.05 else '(유의하지 않음)'}")
    out()

    # ---- 보조: 외부 테스트 (학습 금지 — 자체 라벨 전체로 적합 후 공식 33 에 1회 적용)
    #  #77 사전등록이 보조로 요구했는데 평가기에 빠져 있었다(2026-10-11 감사에서 발견).
    #  주지표 판정은 이 블록과 무관하다 — 여기서 뭐가 나와도 기준 1·2 는 위에서 이미 끝났다.
    if a.external:
        out("## 보조 — 외부 테스트 (공식 train-main-v2 문제단위 라벨 33 · 학습 금지)")
        for item in a.external:
            alias, path = item.split("=", 1)
            ze = np.load(path, allow_pickle=True)
            src = ext_src.get(alias)
            if src is None:
                out(f"   {alias}: 자체 라벨 특징이 없어 건너뜀")
                continue
            fs_tr, y_tr, lay = src
            ye = np.asarray(ze["y"], int)
            we = np.asarray(ze["words"], float)
            best = None
            #  층·C 는 **자체 라벨 안에서만** 교차검증으로 고른다(외부 라벨을 보지 않는다).
            for l in lay:
                plan_l = make_plan({l: fs_tr[l]}, y_tr, [l], a.folds, 0)
                if plan_l is None:
                    continue
                sc, pk = nested_scores(y_tr, [l], plan_l)
                aa = auc_of(y_tr, sc)
                if best is None or aa > best[0]:
                    best = (aa, l, pk[0][1])
            if best is None:
                out(f"   {alias}: 폴드 생성 불가")
                continue
            _, l_sel, c_sel = best
            sc_ = StandardScaler().fit(fs_tr[l_sel])
            Z = sc_.transform(fs_tr[l_sel])
            pca = PCA(n_components=min(NPC, len(y_tr) - 1, Z.shape[1]), random_state=0).fit(Z)
            m = LogisticRegression(C=c_sel, class_weight="balanced", max_iter=5000)
            m.fit(pca.transform(Z), y_tr)
            Xe = pca.transform(sc_.transform(ze[f"h{l_sel}"]))
            se = m.decision_function(Xe)
            # 외부 null: 외부 라벨만 섞는다(프로브는 고정) — 적합을 다시 하지 않으므로 1000회가 싸다
            rng = np.random.default_rng(0)
            nn = [auc_of(rng.permutation(ye), se) for _ in range(a.perms)]
            lo_e, hi_e = np.percentile(nn, [2.5, 97.5])
            ae = auc_of(ye, se)
            re_ = auc_of(ye, resid(se, np.log(np.maximum(1, we))))
            out(f"   {alias}: 고른 층 {l_sel} · C {c_sel} → AUC **{ae:.3f}** "
                f"[null {lo_e:.2f}, {hi_e:.2f}]{' ▲' if ae > hi_e else ''} · 길이잔차 {re_:.3f} · "
                f"n {len(ye)} (robust {int(ye.sum())} / spurious {int((1 - ye).sum())})")
        out()

    out("## 판정")
    out()
    passed = mean_auc > mean_hi and mean_rauc > mean_rhi
    out(f"주지표·길이잔차 {'둘 다 null 을 넘었다' if passed else '중 하나 이상이 null 안이다'} → "
        f"**{'신호 있음(대조 라벨 확인 필요)' if passed else '신호 미입증'}**")
    out()
    out("검정력 메모(Amendment 2 에 미리 적음): k=8 라벨은 모델당 약 30문제로 공식 33 과 같은 규모다.")
    out("그 규모에서는 AUC 0.75 미만을 우연과 구분할 수 없다(`analysis/power_33.py`) — 따라서 이 표는")
    out("'신호 있음' 을 주장하기보다 **구간을 보고하는** 쪽으로 읽는다.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
