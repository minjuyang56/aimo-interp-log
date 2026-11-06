"""#TT-PERTURB (루프 R1) 평가 — 공식 33문제 주최측 라벨. 비용 $0.

사전등록(CLAUDE.md #TT-PERTURB) 그대로:
  * 점수 score_spurious = (답이 원본과 갈린 섭동 수)/(쓸 수 있는 섭동 수), **높을수록 spurious**
  * 임계는 **사전확률 교정** + **문제 단위 leave-one-out 안에서만** (테스트 문제를 보고 정하지 않는다)
  * **주지표 = 모델별 정확도의 모델평균** · 주 체크포인트 **CoT 512**
  * 검문소: 문제단위 셔플 null 1000(정확도·AUC 둘 다) · 길이 대조군 h15C(같은 LOO 절차) ·
    '전부 robust' 상수 0.606 · 사전지정 2차 규칙 `score <= 0.10`
  * 사전지정 민감도: 체크포인트 0·256·1024 · 섭동 유형 부분집합 · 모델평균 문제단위 예측

사용: .venv/Scripts/python.exe analysis/ttp_eval.py --dir experiments/results/runpod/ttperturb/<pod>/out
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "solutions/ttp"))
REPORT = ROOT / "experiments/results/ttp_report.txt"
PRIMARY_CKPT = "512"          # 사전등록 보완(2026-10-18)으로 결과 보기 전에 고정
K_MIN = 4                     # ttp_lib.K_MIN 과 같아야 한다 (아래에서 수치 확인)
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


# ---------------------------------------------------------------- 지표

def auc(spur, y):
    """robust=1 을 양성으로 본 AUC. 점수는 **−spur**(spur 높을수록 spurious)."""
    s = -np.asarray(spur, float)
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    # 동점 보정 (평균 순위)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv), float)
    i = 0
    sv = allv[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def thr_prior(spur_tr, y_tr):
    """학습분에서 **예측 True 비율 = 그 라벨의 robust 비율** 이 되게 하는 임계.
    spur 오름차순으로 k개를 True 로 찍는다. 경계가 동점이면 비율이 목표에서 벗어나고,
    사전등록대로 **벗어난 값을 그대로 쓴다**(임계를 손대지 않는다)."""
    n = len(y_tr)
    k = int(round(float(np.mean(y_tr)) * n))
    sv = np.sort(np.asarray(spur_tr, float))
    if k <= 0:
        return sv[0] - 1e-9
    if k >= n:
        return sv[-1] + 1e-9
    return (sv[k - 1] + sv[k]) / 2.0


def loo_acc(spur, y):
    """문제 단위 leave-one-out: i 를 빼고 임계를 교정해 i 를 맞힌다. 33번."""
    spur = np.asarray(spur, float)
    y = np.asarray(y, int)
    n = len(y)
    pred = np.zeros(n, bool)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        t = thr_prior(spur[m], y[m])
        pred[i] = spur[i] <= t
    return float((pred == (y == 1)).mean()), pred


def null_band(spur, y, reps=1000, seed=0):
    """라벨만 섞고 같은 LOO 절차·같은 AUC 를 다시 낸다 → 정확도·AUC 의 95% 밴드."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    accs, aucs = [], []
    for _ in range(reps):
        ys = rng.permutation(y)
        accs.append(loo_acc(spur, ys)[0])
        aucs.append(auc(spur, ys))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(accs), q(aucs)


# ---------------------------------------------------------------- 점수 만들기

def score_rows(rows, ckpt, kinds=None):
    """(pidx → (score_spurious, n_usable, orig_ans)). kinds 가 주어지면 그 유형만 쓴다."""
    by = {}
    for r in rows:
        by.setdefault(r["pidx"], {"orig": None, "pert": []})
        v = r["val"].get(ckpt)
        if r["kind"] == "orig":
            by[r["pidx"]]["orig"] = v
        elif kinds is None or r["kind"] in kinds:
            by[r["pidx"]]["pert"].append((r["kind"], v))
    res = {}
    for p, d in by.items():
        o = d["orig"]
        usable = [v for _, v in d["pert"] if v is not None]
        if o is None or len(usable) < K_MIN:
            res[p] = (None, len(usable), o)
        else:
            res[p] = (sum(1 for v in usable if v != o) / len(usable), len(usable), o)
    return res


def pidx_map(txt):
    """`org33_problems.json` 의 `pidx` 는 train-main-v2 **36문제** 기준(0~35)이고 라벨 배열은 33개
    순서다. 둘을 그대로 맞대면 15문제가 어긋나고 pidx 33~35 는 조용히 버려진다(2026-10-18 발견).
    **문제 텍스트로 매핑한다** — 33/33 완전 일치를 확인했다."""
    org = list(json.loads((ROOT / "experiments/runpod/org33_problems.json")
                          .read_text(encoding="utf-8")).values())[0]
    t2i = {t: i for i, t in enumerate(txt)}
    m = {x["pidx"]: t2i[x["problem"]] for x in org if x["problem"] in t2i}
    if len(m) != len(txt):
        raise SystemExit(f"pidx 매핑 실패 {len(m)}/{len(txt)} — 정렬이 깨진 채로 평가하지 않는다")
    return m


def reindex(sc, p2i):
    """score_rows 의 pidx 키를 라벨 인덱스 키로 바꾼다."""
    return {p2i[k]: v for k, v in sc.items() if k in p2i}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    d = Path(a.dir)

    import ttp_lib as T
    assert T.K_MIN == K_MIN, f"K_MIN 불일치 {T.K_MIN} vs {K_MIN}"
    from h21_uncertainty import problems
    rows_lab = problems()
    y_all = np.array([r["y"] for r in rows_lab], int)
    txt = [r["problem"] for r in rows_lab]
    n = len(y_all)
    base_rate = float(y_all.mean())

    p2i = pidx_map(txt)
    out("#TT-PERTURB (루프 R1) 결과 — 공식 train-main-v2 문제단위 라벨 33문제")
    out(f"   라벨 robust {int(y_all.sum())} / spurious {int(n - y_all.sum())} · "
        f"'전부 robust' 상수 정확도 **{base_rate:.4f}**")
    out(f"   주 체크포인트 CoT **{PRIMARY_CKPT}** (사전등록 보완으로 결과 보기 전 고정) · "
        f"임계는 사전확률 교정 + LOO · 셔플 null {a.reps}")
    out()

    files = sorted(d.glob("*_ttp_rows.json"))
    if not files:
        out(f"!! {d} 에 *_ttp_rows.json 이 없다")
        REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
        return
    models = {}
    for f in files:
        models[f.name.replace("_ttp_rows.json", "")] = json.loads(f.read_text(encoding="utf-8"))

    # ---- 길이 대조군 (같은 LOO 절차, spur = +log 단어수 = 길수록 spurious)
    lw = np.array([math.log(max(1, len(t.split()))) for t in txt], float)
    acc_len, _ = loo_acc(lw, y_all)
    auc_len = auc(lw, y_all)
    (lo_a, hi_a), (lo_u, hi_u) = null_band(lw, y_all, a.reps, 7)
    out("## 대조군")
    out(f"   길이 h15C (LOO 임계) : 정확도 **{acc_len:.4f}** [null {lo_a:.3f},{hi_a:.3f}] · "
        f"AUC {auc_len:.4f} [null {lo_u:.3f},{hi_u:.3f}]")
    out(f"   '전부 robust' 상수   : 정확도 **{base_rate:.4f}** · AUC 0.500 (정의상)")
    out()

    ckpts = sorted({c for r in next(iter(models.values())) for c in r["val"]}, key=int)
    out("## 주지표 — 체크포인트별 · 모델별 (정확도 / AUC)")
    hdr = f"   {'ckpt':>5} " + " ".join(f"{m[:14]:>16}" for m in models) + f"{'모델평균':>12}{'null상단':>10}"
    out(hdr)
    summary = {}
    for ck in ckpts:
        accs, aucs, nulls_a, nulls_u, cells = [], [], [], [], []
        for mname, rows in models.items():
            sc = reindex(score_rows(rows, ck), p2i)
            spur = np.array([sc.get(i, (None, 0, None))[0] for i in range(n)], dtype=object)
            # 판정 불가(None) 는 길이 폴백으로 메운다 — 제출물이 그렇게 하기 때문이다
            nfb = int(sum(1 for v in spur if v is None))
            sv = np.array([lw[i] / 100.0 + 0.5 if spur[i] is None else float(spur[i]) for i in range(n)], float)
            acc, _ = loo_acc(sv, y_all)
            au = auc(sv, y_all)
            (la, ha), (lu, hu) = null_band(sv, y_all, a.reps, 11)
            accs.append(acc); aucs.append(au); nulls_a.append(ha); nulls_u.append(hu)
            cells.append(f"{acc:.3f}/{au:.3f}" + (f"(fb{nfb})" if nfb else ""))
        ma, mu = float(np.mean(accs)), float(np.mean(aucs))
        summary[ck] = dict(acc=ma, auc=mu, null_acc=float(np.mean(nulls_a)),
                           null_auc=float(np.mean(nulls_u)),
                           per_model_acc=dict(zip(models, accs)), per_model_auc=dict(zip(models, aucs)))
        star = " <= 주지표" if ck == PRIMARY_CKPT else ""
        out(f"   {ck:>5} " + " ".join(f"{c:>16}" for c in cells) +
            f"{ma:>7.3f}/{mu:.3f}{np.mean(nulls_a):>10.3f}{star}")
    out()

    p = summary.get(PRIMARY_CKPT)
    if p:
        out("## 사전등록 판정 (주 체크포인트 " + PRIMARY_CKPT + ")")
        out(f"   모델평균 정확도 **{p['acc']:.4f}** · 셔플 null 95% 상단 평균 {p['null_acc']:.3f}")
        out(f"   모델평균 AUC    **{p['auc']:.4f}** · null 상단 평균 {p['null_auc']:.3f}")
        out(f"   #TT-PERTURB 채택 기준 0.606+0.10 = **0.706** → "
            f"{'통과' if p['acc'] >= 0.706 else '**미달**'}")
        out(f"   #LOOP 목표 0.90 → {'달성' if p['acc'] >= 0.90 else '**미달**'}")
        out(f"   길이 대조 {acc_len:.4f} 대비 {p['acc'] - acc_len:+.4f} · "
            f"상수 {base_rate:.4f} 대비 {p['acc'] - base_rate:+.4f}")
        out()

    # ---- 사전지정 2차 규칙 (주최측 문자 그대로)
    out("## 사전지정 2차 규칙 — score_spurious <= 0.10 (주최측 임계 문자 그대로)")
    for ck in ckpts:
        accs = []
        for mname, rows in models.items():
            sc = reindex(score_rows(rows, ck), p2i)
            pred = np.array([(sc.get(i, (None,))[0] is not None and sc[i][0] <= 0.10) for i in range(n)])
            accs.append(float((pred == (y_all == 1)).mean()))
        out(f"   ckpt {ck:>5}: " + " ".join(f"{m[:10]} {v:.3f}" for m, v in zip(models, accs)) +
            f" · 모델평균 {np.mean(accs):.4f}")
    out()

    # ---- 섭동 유형 부분집합 (사전지정 민감도)
    out("## 민감도 — 섭동 유형 부분집합 (주 체크포인트, 모델평균 정확도)")
    subsets = {"전체(8종)": None, "rename만": {"rename", "rename2"},
               "typos만": {"typos0", "typos1"}, "distract만": {"distract0", "distract1"},
               "공식3유형": {"rename", "rename2", "typos0", "typos1", "distract0", "distract1"},
               "재서술만": {"reorder", "notation"}}
    for nm, ks in subsets.items():
        accs, nus = [], []
        for mname, rows in models.items():
            sc = reindex(score_rows(rows, PRIMARY_CKPT, ks), p2i)
            nus.append(float(np.mean([sc.get(i, (None, 0))[1] for i in range(n)])))
            sv = np.array([lw[i] / 100.0 + 0.5 if sc.get(i, (None,))[0] is None else float(sc[i][0])
                           for i in range(n)], float)
            accs.append(loo_acc(sv, y_all)[0])
        out(f"   {nm:<12} 모델평균 정확도 {np.mean(accs):.4f} · 쓸 수 있는 섭동 평균 {np.mean(nus):.2f}")
    out()

    # ---- 보조: 모델평균 점수로 만든 문제단위 예측
    out("## 보조 — 모델평균 점수의 문제단위 예측 (테스트타임에 4모델을 다 열면 가능)")
    for ck in ckpts:
        mat = []
        for mname, rows in models.items():
            sc = reindex(score_rows(rows, ck), p2i)
            mat.append([lw[i] / 100.0 + 0.5 if sc.get(i, (None,))[0] is None else float(sc[i][0])
                        for i in range(n)])
        sv = np.asarray(mat, float).mean(0)
        acc, _ = loo_acc(sv, y_all)
        au = auc(sv, y_all)
        (la, ha), (lu, hu) = null_band(sv, y_all, a.reps, 13)
        out(f"   ckpt {ck:>5}: 정확도 {acc:.4f} [null {la:.3f},{ha:.3f}] · AUC {au:.4f} [null {lu:.3f},{hu:.3f}]")
    out()

    # ---- 진단: 점수 분포·추출률
    out("## 진단 (판정이 아니다)")
    for mname, rows in models.items():
        meta = d / f"{mname}_ttp_meta.json"
        mt = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        sc = reindex(score_rows(rows, PRIMARY_CKPT), p2i)
        vals = [v for v, _, _ in sc.values() if v is not None]
        nfb = sum(1 for v, _, _ in sc.values() if v is None)
        oks = sum(1 for _, _, o in sc.values() if o is not None)
        out(f"   {mname:<30} 원본답 추출 {oks}/{n} · 판정불가 {nfb} · "
            f"score 중앙 {np.median(vals) if vals else float('nan'):.3f} · "
            f"서로 다른 값 {len(set(vals))} · 마커 {mt.get('marks', '?')} · {mt.get('seconds', '?')}s")
    out()
    out("(점수가 몇 칸뿐이면 분해능이 낮다는 뜻이다 — K 중앙값 6 이면 0, 1/6, …, 1 의 7칸이다.)")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")
    json.dump(summary, open(ROOT / "experiments/results/ttp_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
