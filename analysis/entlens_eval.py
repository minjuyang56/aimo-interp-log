"""#LIT-PLAN P2·P3 평가기 — 엔트로피 궤적 3특징 + logit lens 결정 층. 비용 $0.

사전등록은 CLAUDE.md `### P2 — 토큰 엔트로피 궤적` · `### P3 — logit lens` 절이다.
**이 파일은 커널 결과를 보기 전에 썼다**(커밋 순서로 확인된다).

특징 (부호 고정 — 결과 보고 뒤집지 않는다)
  `ent_front` = 앞 25%(`t < ceil(T/4)`) H 평균            큼 = spurious
  `ent_slope` = H_t 를 t/(T−1) 에 OLS 회귀한 기울기        큼 = spurious
  `ent_back`  = 뒤 25%(`t >= T − floor(T/4)`) H 평균       큼 = spurious
  `lens_frac` = min{ℓ : p_ℓ > 0.5} / L (넘는 층 없으면 1.0)  큼 = robust

주지표 = **3특징 로지스틱의 문제단위 LOPO** 모델평균 (정확도·AUC).
보조 = 세 특징 단독(학습 없음) · 앞 300토큰판 · 전체평균 `mean_entropy`(H21a 와 맞대기) ·
P3 의 임계 0.3·0.7 판 · 최종층 확률.
검문소 = 셔플 null 1000 · 길이 h15C(LOO) · 상수 '전부 robust' · log(단어수) 잔차.

**내가 고정한 칸**(사전등록에 없던 것): 3특징 로지스틱 앞에 **폴드 학습분 StandardScaler** 를 둔다
(y 를 보지 않고, 선형 모델이라 AUC 를 거의 바꾸지 않지만 C 격자의 의미를 모델 간에 맞춘다).
`T < 8` 인 행은 사분면에 2개가 안 들어가 **판정 불가**로 적고 그 행만 뺀다.

사용:
  .venv/Scripts/python.exe analysis/entlens_eval.py --scan            # 받은 파일 목록만
  .venv/Scripts/python.exe analysis/entlens_eval.py --perms 1000 --jobs 16
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))

import cmp4_eval as C4  # noqa: E402
import org_train_eval as OT  # noqa: E402
from p1_lit import lopo_arrays, stack_proj  # noqa: E402
from probe_final_eval import CS, auc_of, resid, _perm_list  # noqa: E402

RES = ROOT / "experiments/results"
OUT = RES / "entlens_report.txt"
SMALL = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


# ───────────────────────── 특징 ─────────────────────────

def ent_feats(ent, cap=None):
    """(front, slope, back, mean) 또는 None(T<8)."""
    a = np.asarray(ent, float)
    if cap:
        a = a[:cap]
    T = len(a)
    if T < 8:
        return None
    nf = -(-T // 4)                      # ceil(T/4)
    nb = T // 4                          # floor(T/4)
    x = np.arange(T, dtype=float) / (T - 1)
    slope = float(np.polyfit(x, a, 1)[0])
    return (float(a[:nf].mean()), slope, float(a[T - nb:].mean()), float(a.mean()))


def lens_frac(pc, thr=0.5):
    """`min{ℓ : p_ℓ > thr} / L`. 넘는 층이 없으면 1.0 (끝까지 안 확정)."""
    if not pc:
        return None
    L = len(pc) - 1
    for li, p in enumerate(pc):
        if p > thr:
            return li / L
    return 1.0


# ───────────────────────── LOPO (고정 점수 · 학습 점수) ─────────────────────────

def lopo_fixed(score, y):
    """점수는 고정(적합 없음) · **임계만** 학습 폴드에서 다시 낸다 — `org_train_eval.lopo_len` 과 같다."""
    n = len(y)
    s = np.asarray(score, float)
    pr = np.zeros(n, dtype=bool)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        pr[i] = s[i] >= OT.mid_thr(s[tr], y[tr])
    return s, pr


def prep_lopo_scale(X, n):
    pr = []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        sc = StandardScaler().fit(X[tr])
        pr.append((sc.transform(X[tr]), sc.transform(X[i:i + 1]), tr))
    return stack_proj(pr)


def _one(yp, pack, fixed):
    """perm 하나: 3특징 로지스틱 + 고정점수 지표들을 **같은 추출**에서 돌린다."""
    sa, pa, _ = lopo_arrays(pack[0], pack[1], pack[2], yp, C4.fs_logistic, CS)
    r = [auc_of(yp, sa), float((pa == (yp == 1)).mean())]
    for s in fixed:
        _, pf = lopo_fixed(s, yp)
        r += [auc_of(yp, s), float((pf == (yp == 1)).mean())]
    return r


def band(y, pack, fixed, perms, jobs):
    yps = _perm_list(y, perms)
    #  **병렬은 구현 세부이고 통계량을 바꾸지 않는다** — 순열 수열이 시드로 고정되어 있고
    #  perm 끼리 독립이므로 직렬과 병렬이 같은 배열을 낸다(p1_lit 의 그 구조와 같다).
    #  윈도에서 loky 가 BrokenProcessPool 로 죽는 경우가 있어(부모가 파이프로 감싸인 중첩
    #  subprocess 일 때 관측) **직렬로 떨어뜨린다** — 느려지지만 숫자는 같다.
    if jobs and jobs != 1:
        try:
            from joblib import Parallel, delayed
            return np.array(Parallel(n_jobs=jobs, prefer="processes")(
                delayed(_one)(yp, pack, fixed) for yp in yps))
        except Exception as e:                                    # noqa: BLE001
            print(f"[band] 병렬 실패({type(e).__name__}) → 직렬로 돈다", flush=True)
    return np.array([_one(yp, pack, fixed) for yp in yps])


# ───────────────────────── 데이터 ─────────────────────────

def find_files():
    """받은 커널 산출물을 긁어모은다 (`*_p2.json` 과 같은 폴더의 `*_p3.json`).

    **`RES/entlens` 아래만 본다** — #FLIP-ENT 가 같은 커널(`entlens.py`)로 **다른 문제집합**을
    `RES/flipent` 에 받으므로, RES 전체를 rglob 하면 **같은 별칭이 겹쳐 조용히 덮인다**(정렬 순서로
    늦은 경로가 이긴다). 통계량은 바뀌지 않는다(P2 산출물은 원래 그 폴더에만 있었다).
    """
    got = {}
    for f in sorted((RES / "entlens").rglob("*_p2.json")):
        alias = f.name[:-len("_p2.json")]
        p3 = f.with_name(f"{alias}_p3.json")
        got[alias] = dict(p2=f, p3=p3 if p3.is_file() else None)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()

    got = find_files()
    if a.scan or not got:
        print(f"찾은 묶음 {len(got)}")
        for k, v in got.items():
            n2 = len(json.loads(v["p2"].read_text(encoding="utf-8")))
            n3 = len(json.loads(v["p3"].read_text(encoding="utf-8"))) if v["p3"] else 0
            print(f"  {k:<30} p2 {n2}행 · p3 {n3}행 · {v['p2'].parent}")
        if not got:
            print("받은 것이 없다 — 커널 산출물을 먼저 내려받아라")
        raise SystemExit(0)

    from h21_uncertainty import problems
    lab = {r["problem"]: r["y"] for r in problems()}

    out("#LIT-PLAN P2·P3 — 엔트로피 궤적 3특징 + logit lens 결정 층 (사전등록 1회) · 비용 $0")
    out(f"   공식 33문제 · Small 4모델 · 셔플 null {a.perms}회 · jobs {a.jobs}")
    out("   부호 고정: ent_* 큼 = spurious · lens_frac 큼 = robust (뒤집지 않는다)")
    out()
    agg = {}
    for alias in sorted(got):
        rows = json.loads(got[alias]["p2"].read_text(encoding="utf-8"))
        lens = {int(r["pidx"]): r for r in
                json.loads(got[alias]["p3"].read_text(encoding="utf-8"))} if got[alias]["p3"] else {}
        keep, bad = [], []
        for r in rows:
            f = ent_feats(r["ent"])
            if f is None:
                bad.append((r["pidx"], len(r["ent"])))
                continue
            if r["problem"] not in lab:
                bad.append((r["pidx"], "라벨 없음"))
                continue
            f3 = ent_feats(r["ent"], cap=300)
            lr = lens.get(int(r["pidx"]), {})
            keep.append(dict(pidx=int(r["pidx"]), y=lab[r["problem"]],
                             words=len(r["problem"].split()), f=f, f300=f3,
                             pc=lr.get("pcurve"), ans=lr.get("ans_int"),
                             n_tok=r["n_tok"], fin=r["finished"], marks=r["n_marks"]))
        if len(keep) < 10:
            out(f"## {alias} — 쓸 수 있는 행 {len(keep)} < 10 → **판정 불가** (버린 행 {bad[:5]})")
            out()
            continue
        y = np.array([r["y"] for r in keep], int)
        words = np.array([r["words"] for r in keep], float)
        X = np.array([r["f"][:3] for r in keep], float)
        out(f"## {alias} · 행 {len(y)} (robust {int(y.sum())} / spurious {int((1 - y).sum())})"
            f" · 완주 {sum(r['fin'] for r in keep)} · 마커 {sum(r['marks'] for r in keep)}"
            f" · 토큰 중앙 {int(np.median([r['n_tok'] for r in keep]))}"
            + (f" · 버린 행 {len(bad)}" if bad else ""))

        #  고정 점수들 — 부호를 여기서 적용한다(ent 는 −, lens 는 +)
        fixed, names = [], []
        for j, nm in enumerate(("ent_front", "ent_slope", "ent_back")):
            fixed.append(-X[:, j])
            names.append(nm)
        fixed.append(-np.array([r["f"][3] for r in keep], float))
        names.append("mean_entropy(H21a)")
        lf = np.array([lens_frac(r["pc"]) if r["pc"] else np.nan for r in keep], float)
        n_lens = int(np.isfinite(lf).sum())
        if n_lens == len(lf):
            fixed.append(lf)
            names.append("lens_frac(P3 주스칼라)")
        for t in (0.3, 0.7):
            v = np.array([lens_frac(r["pc"], t) if r["pc"] else np.nan for r in keep], float)
            if np.isfinite(v).all():
                fixed.append(v)
                names.append(f"lens_frac thr={t}")
        plast = np.array([(r["pc"][-1] if r["pc"] else np.nan) for r in keep], float)
        if np.isfinite(plast).all():
            fixed.append(plast)
            names.append("최종층 확률(탐색)")
        fixed.append(-np.log(np.maximum(1, words)))
        names.append("길이 h15C")

        pack = prep_lopo_scale(X, len(y))
        nul = band(y, pack, fixed, a.perms, a.jobs)
        s3, p3, picks = lopo_arrays(pack[0], pack[1], pack[2], y, C4.fs_logistic, CS)
        au, ac = auc_of(y, s3), float((p3 == (y == 1)).mean())
        hA, hAc = float(np.percentile(nul[:, 0], 95)), float(np.percentile(nul[:, 1], 95))
        out(f"   **주지표 3특징 LOPO**  AUC {au:.3f} (null상단 {hA:.2f}"
            f"{' ▲' if au > hA else ''}) · 정확도 **{ac:.3f}** (null상단 {hAc:.2f}"
            f"{' ▲' if ac > hAc else ''}) · True비율 {p3.mean():.3f}")
        uc = {c: picks.count(c) for c in sorted(set(picks))}
        out(f"   {'':<6} 고른 C {uc} · 길이잔차 AUC {auc_of(y, resid(s3, np.log(np.maximum(1, words)))):.3f}")
        agg[alias] = dict(auc=au, acc=ac, hi=hA, hi_acc=hAc, n=len(y), n_lens=n_lens)
        for k, nm in enumerate(names):
            sv = fixed[k]
            _, pf = lopo_fixed(sv, y)
            aa, cc = auc_of(y, sv), float((pf == (y == 1)).mean())
            h1 = float(np.percentile(nul[:, 2 + 2 * k], 95))
            h2 = float(np.percentile(nul[:, 3 + 2 * k], 95))
            out(f"   {nm:<24} AUC {aa:.3f} (상단 {h1:.2f}{' ▲' if aa > h1 else ''}) · "
                f"정확도 {cc:.3f} (상단 {h2:.2f}{' ▲' if cc > h2 else ''})")
            agg[alias].setdefault("fx", {})[nm] = (aa, cc, h1, h2)
        #  민감도: 앞 300토큰판 3특징
        X3 = np.array([r["f300"][:3] for r in keep if r["f300"]], float)
        if len(X3) == len(y):
            pk = prep_lopo_scale(X3, len(y))
            s, p, _ = lopo_arrays(pk[0], pk[1], pk[2], y, C4.fs_logistic, CS)
            out(f"   {'앞300토큰판 3특징(민감도)':<24} AUC {auc_of(y, s):.3f} · "
                f"정확도 {float((p == (y == 1)).mean()):.3f}")
        out(f"   상수 '전부 robust' 정확도 {float((y == 1).mean()):.3f}"
            f"{'' if n_lens == len(lf) else f' · P3 렌즈 결측 {len(lf) - n_lens}행 → 그 칸 생략'}")
        out()

    if not agg:
        out("판정할 모델이 없다")
        OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        raise SystemExit(1)

    out("## 주지표 요약 — 모델평균 (사전등록: 정확도 > 0.65 **그리고** 자기 null 상단 초과)")
    ma = float(np.mean([v["auc"] for v in agg.values()]))
    mc = float(np.mean([v["acc"] for v in agg.values()]))
    mh = float(np.mean([v["hi"] for v in agg.values()]))
    mhc = float(np.mean([v["hi_acc"] for v in agg.values()]))
    out(f"   모델 {len(agg)}종 · 모델평균 AUC **{ma:.3f}** (null상단평균 {mh:.2f}) · "
        f"모델평균 정확도 **{mc:.3f}** (null상단평균 {mhc:.2f})")
    c1, c2 = mc > 0.65, mc > mhc
    out(f"   1. 정확도 {mc:.3f} > 0.65 → {'통과' if c1 else '**미달**'}")
    out(f"   2. 정확도 {mc:.3f} > null 상단 평균 {mhc:.3f} → {'통과' if c2 else '**미달**'}")
    out(f"   → **{'통과 · zip 빌드 대상' if (c1 and c2) else '미달 · zip 만들지 않는다'}**")
    out()
    out("## 보조 특징 모델평균")
    keys = sorted({k for v in agg.values() for k in v.get("fx", {})})
    for nm in keys:
        vs = [v["fx"][nm] for v in agg.values() if nm in v.get("fx", {})]
        out(f"   {nm:<24} AUC {np.mean([x[0] for x in vs]):.3f} (상단 "
            f"{np.mean([x[2] for x in vs]):.2f}) · 정확도 {np.mean([x[1] for x in vs]):.3f} "
            f"(상단 {np.mean([x[3] for x in vs]):.2f}) · 모델 {len(vs)}")
    out()
    out("## 읽는 법")
    out("   * 기준선: 상수 '전부 robust' 0.606 · 길이 h15C(LOO) 0.636 — **이 두 줄을 못 넘으면**")
    out("     숫자가 null 안인지와 무관하게 쓸 물건이 아니다(#SUB2048-PLUS 에서 같은 일이 있었다).")
    out("   * H21a 기록(`mean_entropy` 1024토큰 전체 평균) 4모델 0.608~0.658 · 전부 null 안 —")
    out("     이번 `mean_entropy(H21a)` 줄이 그 값을 재현하면 측정 경로가 맞다는 뜻이다.")
    out("   * P3 가 실패해도 '암기 신호 없음' 이 아니라 '강제 즉답이 불안정하다' 일 수 있다")
    out("     (R1: CoT 길이만 바꿔도 즉답이 64~82% 문제에서 달라졌다).")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
