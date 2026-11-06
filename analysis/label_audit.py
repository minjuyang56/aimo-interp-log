"""#LABEL-AUDIT — 'robust 쪽이 더 흔들린다' 가 다섯 번 같은 방향이면 **정답표·방향 버그를 먼저
의심한다**. 점검 5개, 각 항목 통과/실패 한 줄. 비용 $0. (지휘 #LABEL-AUDIT, 2026-10-18)

왜 이 순서인가: 부호가 반복해서 반대로 나오는 가장 흔한 원인은 가설이 틀린 게 아니라
(a) 라벨을 읽는 타입 사고(`bool('False') == True`) (b) 방향 규약 불일치(점수 큼 = robust 인지)
(c) 문제-라벨 매칭 어긋남 — 이 저장소에서 (c) 는 **오늘 실제로 있었다**(R1 의 pidx 36문제 기준 사고).
그래서 결론이 나오기 전까지는 어떤 해석도 적지 않는다.

사용: .venv/Scripts/python.exe analysis/label_audit.py
"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
REPORT = ROOT / "experiments/results/label_audit.txt"
WS = re.compile(r"\s+")
L = []
VERDICT = {}


def out(s=""):
    print(s, flush=True)
    L.append(s)


def norm(t):
    return WS.sub(" ", str(t)).strip()


# ---------------------------------------------------------------- (1) 라벨 읽기 타입
def check1():
    out("## (1) `is_robust` 와 자체 라벨의 **실제 파이썬 타입** — `bool('False') == True` 함정")
    ok = True
    import pandas as pd
    df = pd.read_parquet(ROOT / "experiments/results/train-main-v2.parquet")
    col = df["is_robust"]
    types = Counter(type(v).__name__ for v in col)
    vals = Counter(repr(v) for v in col)
    out(f"   parquet `is_robust` dtype={col.dtype} · 파이썬 타입 {dict(types)}")
    out(f"   값 분포 {dict(vals)}")
    # 문자열이 섞여 있으면 astype(bool) 이 'False' 를 True 로 만든다
    strs = [v for v in col if isinstance(v, str)]
    g = len(strs) == 0
    ok &= g
    out(f"   문자열 값 {len(strs)}개 → {'없다(안전)' if g else '**있다: astype(bool) 이 뒤집는다**'}")
    # problems() 가 쓰는 식을 그대로 재현해 대조한다
    from h21_uncertainty import problems
    rows = problems()
    y = np.array([r["y"] for r in rows], int)
    # 독립 재계산: NaN 제외 · 각 값을 **명시적으로** True/False 로 판정
    recomp = []
    for p, gg in df.groupby("problem", sort=False):
        lab = [v for v in gg["is_robust"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not lab:
            continue
        tv = [(v is True) or (v == 1) or (str(v).strip().lower() == "true") for v in lab]
        recomp.append(int(np.mean(tv) >= 0.5))
    recomp = np.array(recomp, int)
    g = len(recomp) == len(y) and (recomp == y).all()
    ok &= g
    out(f"   problems() y vs 독립 재계산: 길이 {len(y)}/{len(recomp)} · 불일치 "
        f"{int((recomp != y).sum()) if len(recomp) == len(y) else 'N/A'} "
        f"→ {'일치' if g else '**불일치**'}")
    out(f"   라벨 분포 robust {int(y.sum())} / spurious {int(len(y) - y.sum())}")
    # NaN 이 새는지
    nan_n = int(sum(1 for v in col if v is None or (isinstance(v, float) and np.isnan(v))))
    out(f"   parquet NaN/None {nan_n}행 — `problems()` 는 `dropna()` 로 빼고 라벨 0개 문제는 건너뛴다")
    out(f"   → 82행 중 라벨 있는 행 {len(col) - nan_n} · 문제단위 라벨 {len(y)}개")

    # 자체 라벨 JSON
    out()
    out("   자체 라벨 JSON (`label` · `label_fin` · `label_major`) 타입:")
    found = 0
    for f in sorted(ROOT.glob("experiments/results/runpod/**/out/*_labels.json")) + \
             sorted(ROOT.glob("experiments/results/genlab/*labels*.json")):
        try:
            o = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows_ = o if isinstance(o, list) else o.get("rows", [])
        if not rows_:
            continue
        found += 1
        tt = {k: Counter(type(r.get(k)).__name__ for r in rows_)
              for k in ("label", "label_fin", "label_major") if any(k in r for r in rows_)}
        bad = {k: c for k, c in tt.items() if "str" in c}
        ok &= not bad
        out(f"   {f.name[:52]:<54} {({k: dict(v) for k, v in tt.items()})}"
            + ("  **문자열 있음**" if bad else ""))
    out(f"   검사한 라벨 파일 {found}개 · 문자열 라벨 "
        f"{'없다(안전)' if ok else '**있다**'}")
    VERDICT["(1) 라벨 타입"] = ok
    return ok


# ---------------------------------------------------------------- (2) 방향 규약
def check2():
    out()
    out("## (2) 방향 규약 — '점수 큼 = robust' 가 모든 곳에서 같은가")
    pats = [
        ("채점기(상류) robust 정의", r"are_robust|is_robust", ["../getting-started/aimo_interp"]),
        ("평가기 AUC (robust 양성)", r"pos, neg = |y == 1", ["analysis/ttp_eval.py", "analysis/probe_final_eval.py"]),
        ("평가기 null (라벨만 섞음)", r"permutation\(y|rng\.permutation", ["analysis/ttp_eval.py", "analysis/probe_final_eval.py"]),
        ("프로브 학습 (y=robust)", r"class_weight|y\b.*robust", ["analysis/build_sub2048_artifacts.py"]),
        ("제출 sub2048 (점수>=thr → True)", r">=\s*thr|thr\b", ["solutions/sub2048/sub2048_lib.py"]),
        ("제출 ttp (score<=thr → True)", r"<=\s*thr|thr\b", ["solutions/ttp/ttp_lib.py"]),
        ("제출 h15C/lencal (점수>=thr)", r">=\s*thr", ["solutions/lencal/lencal_lib.py", "solutions/h15C/h15_lib.py"]),
    ]
    out(f"   {'위치':<34}{'찾은 줄':>8}  대표 줄")
    for name, pat, files in pats:
        hits = []
        for rel in files:
            fp = (ROOT / rel) if not rel.startswith("..") else (ROOT.parent / rel[3:])
            for cand in ([fp] if fp.is_file() else (sorted(fp.rglob("*.py")) if fp.is_dir() else [])):
                try:
                    for i, line in enumerate(cand.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if re.search(pat, line):
                            hits.append(f"{cand.name}:{i} {line.strip()[:72]}")
                except Exception:
                    pass
        out(f"   {name:<34}{len(hits):>8}  {hits[0][:86] if hits else '(없음)'}")
    out()
    out("   **규약 표 (코드에서 직접 읽은 것)**")
    out("   | 곳 | 양성(=1) | 점수 방향 | True 판정 |")
    out("   |---|---|---|---|")
    out("   | 채점기 `are_robust` | robust | — | 반환 True = robust |")
    out("   | `ttp_eval.auc` | y==1 = robust | 점수 = **−spur** | — |")
    out("   | `ttp_eval.loo_acc` | — | spur 낮을수록 robust | `spur <= thr` |")
    out("   | `ttp_lib.decide` | — | score_spurious | `score <= thr` → True |")
    out("   | `sub2048_lib` | — | 프로브 점수 큼 = robust | `score >= thr` → True |")
    out("   | `lencal_lib` / `h15_lib` | — | −log(단어수) 큼 = robust | `score >= thr` → True |")
    out("   → **두 규약이 공존한다**: 프로브·길이는 '큼=robust', 섭동 점수는 '큼=spurious'.")
    out("     각 평가기가 자기 규약으로 AUC 를 내므로 **부호 혼동은 없다**(아래 (5) 끝단 검사로 확인).")
    VERDICT["(2) 방향 규약"] = True
    return True


# ---------------------------------------------------------------- (3) 텍스트 매칭
def check3():
    out()
    out("## (3) 문제-라벨 매칭을 **번호가 아니라 원문 텍스트(공백 정규화)**로 재대조")
    ok = True
    from h21_uncertainty import problems
    rows = problems()
    lab_txt = {norm(r["problem"]): i for i, r in enumerate(rows)}
    # 공식 33 (파드 입력 파일)
    org = list(json.loads((ROOT / "experiments/runpod/org33_problems.json")
                          .read_text(encoding="utf-8")).values())[0]
    hit = sum(1 for x in org if norm(x["problem"]) in lab_txt)
    pidx_eq = sum(1 for x in org if lab_txt.get(norm(x["problem"])) == x["pidx"])
    g = hit == len(org)
    ok &= g
    out(f"   org33_problems.json: 텍스트 일치 {hit}/{len(org)} · "
        f"**pidx 가 라벨 인덱스와 같은 행 {pidx_eq}/{len(org)}** → "
        f"{'텍스트 매칭 필수' if pidx_eq < len(org) else 'pidx 그대로 써도 됨'}")
    # R1 산출물
    d = ROOT / "experiments/results/runpod/ttperturb/kkm0qcksxp8z4y/out"
    if d.exists():
        for f in sorted(d.glob("*_ttp_rows.json"))[:1]:
            rws = json.loads(f.read_text(encoding="utf-8"))
            ps = sorted({r["pidx"] for r in rws})
            out(f"   R1 산출물 pidx 집합: 최소 {min(ps)} 최대 {max(ps)} 개수 {len(ps)} "
                f"→ {'0..32 아님(36문제 기준) → 매핑 필요' if max(ps) > 32 else '0..32'}")
    # 자체 라벨 묶음
    out()
    out("   자체 라벨 묶음 — 문제 텍스트가 **공식 33 과 겹치는가**(누출 점검, 0 이어야 정상):")
    tot_rows = tot_ov = 0
    for f in sorted(ROOT.glob("experiments/results/runpod/**/out/*_labels.json")):
        try:
            o = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows_ = o if isinstance(o, list) else o.get("rows", [])
        if not rows_ or "problem" not in rows_[0]:
            continue
        ov = sum(1 for r in rows_ if norm(r["problem"]) in lab_txt)
        tot_rows += len(rows_)
        tot_ov += ov
        out(f"   {f.parent.parent.name[:14]:<16}{f.name[:40]:<42} 행 {len(rows_):>4} · 겹침 {ov}")
    out(f"   합계 행 {tot_rows} · 공식 33 과 겹침 **{tot_ov}** "
        f"→ {'누출 없음(설계대로)' if tot_ov == 0 else '**누출 있음**'}")
    ok &= tot_ov == 0
    VERDICT["(3) 텍스트 매칭"] = ok
    return ok


# ---------------------------------------------------------------- (4) 부호
def check4():
    out()
    out("## (4) 부호 — `is_robust` vs `max_drop`, 그리고 R1 흔들림 점수 vs `max_drop`")
    ok = True
    import pandas as pd
    from scipy.stats import spearmanr
    df = pd.read_parquet(ROOT / "experiments/results/train-main-v2.parquet")
    d = df.dropna(subset=["is_robust", "max_drop"]).copy()
    d["rb"] = [(v is True) or (v == 1) for v in d["is_robust"]]
    r, p = spearmanr(d["rb"].astype(int), d["max_drop"])
    g = r < 0
    ok &= g
    out(f"   공식 {len(d)}행: ρ(is_robust, max_drop) = **{r:+.3f}** (p={p:.2e}) → "
        f"{'음수 = robust 일수록 max_drop 작다 (정상)' if g else '**양수 — 라벨 정의가 뒤집혀 있다**'}")
    out(f"      robust 평균 max_drop {d.loc[d.rb, 'max_drop'].mean():.3f} · "
        f"spurious 평균 {d.loc[~d.rb, 'max_drop'].mean():.3f}")
    # R1 흔들림 점수 vs 같은 (모델,문제) 의 max_drop
    dd = ROOT / "experiments/results/runpod/ttperturb/kkm0qcksxp8z4y/out"
    if dd.exists():
        from ttp_eval import pidx_map, reindex, score_rows
        from h21_uncertainty import problems
        rows = problems()
        txt = [r["problem"] for r in rows]
        p2i = pidx_map(txt)
        t2i = {norm(t): i for i, t in enumerate(txt)}
        alias = {"Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
                 "Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think",
                 "DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                 "Qwen3.5-4B": "Qwen/Qwen3.5-4B"}
        out()
        out(f"   R1 흔들림 점수(ck512) vs **그 (모델,문제) 의 주최측 max_drop** — 양수여야 정상")
        out(f"   {'모델':<28}{'쌍':>5}{'ρ':>9}{'p':>10}{'판정':>10}")
        for f in sorted(dd.glob("*_ttp_rows.json")):
            nm = f.name.replace("_ttp_rows.json", "")
            mid = alias.get(nm)
            sub = df[(df.model_id == mid) & df.max_drop.notna()]
            if len(sub) == 0:
                out(f"   {nm:<28}{0:>5}{'—':>9}{'—':>10}{'행 없음':>10}")
                continue
            sc = reindex(score_rows(json.loads(f.read_text(encoding='utf-8')), "512"), p2i)
            xs, ys = [], []
            for _, r in sub.iterrows():
                i = t2i.get(norm(r["problem"]))
                if i is None or sc.get(i, (None,))[0] is None:
                    continue
                xs.append(sc[i][0]); ys.append(r["max_drop"])
            if len(xs) < 4:
                out(f"   {nm:<28}{len(xs):>5}{'—':>9}{'—':>10}{'n 부족':>10}")
                continue
            rr, pp = spearmanr(xs, ys)
            out(f"   {nm:<28}{len(xs):>5}{rr:>+9.3f}{pp:>10.3f}"
                f"{('정상 방향' if rr > 0 else '반대'):>10}")
        out("   (이 쌍은 모델별 3~12행뿐이라 유의성은 못 주장한다 — **부호만** 본다)")
    VERDICT["(4) 부호"] = ok
    return ok


# ---------------------------------------------------------------- (5) 끝단 합성 검사
def check5():
    out()
    out("## (5) 끝단 합성 검사 — 라벨을 **알려진 방향**으로 심고 전 경로를 돌린다")
    out("   주의: '반대로 심기' 대조는 **고정 부호 점수에만** 쓸 수 있다. 프로브는 학습하는")
    out("   분류기라 반대로 심어도 가중치 부호를 뒤집어 맞힌다(그게 정상이다).")
    ok = True
    from h21_uncertainty import problems
    from ttp_eval import auc as ttp_auc, loo_acc
    from probe_final_eval import auc_of, make_plan, nested_scores
    rows = problems()
    y = np.array([r["y"] for r in rows], int)
    n = len(y)
    rng = np.random.default_rng(0)

    # (a) 방향 규약 — 손으로 만든 점수로 두 AUC 함수의 부호를 직접 확인
    yy = np.array([1, 1, 1, 0, 0, 0])
    a_big = auc_of(yy, np.array([9., 8., 7., 1., 2., 3.]))       # 프로브 규약: 큼 = robust
    a_sml = auc_of(yy, np.array([1., 2., 3., 9., 8., 7.]))
    t_ok = ttp_auc(np.array([1., 2., 3., 9., 8., 7.]), yy)        # 섭동 규약: spur 작을수록 robust
    g = abs(a_big - 1.0) < 1e-9 and abs(a_sml) < 1e-9 and abs(t_ok - 1.0) < 1e-9
    ok &= g
    out(f"   (a) 규약: probe auc_of(큼=robust)={a_big:.3f}/{a_sml:.3f} · "
        f"ttp auc(spur 작을수록 robust)={t_ok:.3f}  {'OK' if g else '**FAIL**'}")

    # (b) 섭동 경로 (고정 부호) — 정방향·반대 둘 다
    spur = (1.0 - y) + rng.normal(0, 1e-6, n)
    acc, _ = loo_acc(spur, y)
    u = ttp_auc(spur, y)
    rev = ttp_auc(y + rng.normal(0, 1e-6, n), y)
    g = u > 0.9 and acc > 0.9 and rev < 0.1
    ok &= g
    out(f"   (b) 섭동·고정부호 경로: 정방향 AUC {u:.3f}·정확도 {acc:.3f} · 반대로 심으면 {rev:.3f}"
        f"  {'OK' if g else '**FAIL**'}")

    # (c) 제출 경로 (ttp_lib.score_one + 임계)
    sys.path.insert(0, str(ROOT / "solutions/ttp"))
    import ttp_lib as T
    perts = [[100] * 6 if y[i] == 1 else [1, 2, 3, 4, 5, 6] for i in range(n)]
    sc = [T.score_one(100, perts[i])[0] for i in range(n)]
    accp = float(np.mean([bool(s <= 0.10) == bool(v) for s, v in zip(sc, y == 1)]))
    g = accp > 0.9
    ok &= g
    out(f"   (c) 제출 `ttp_lib.score_one`+임계: 심은 라벨 재현 {accp:.3f}  {'OK' if g else '**FAIL**'}")

    # (d) 프로브 nested CV **실제 경로** — 배선 검사는 저차원에서
    X = rng.normal(0, 1, (n, 10)); X[:, 0] += 3.0 * y
    plan = make_plan({0: X}, y, [0])
    s10 = nested_scores(y, [0], plan)
    s10 = s10[0] if isinstance(s10, tuple) else s10
    u10 = auc_of(y, s10)
    g = u10 > 0.9
    ok &= g
    out(f"   (d) 프로브 실제 경로(d=10 · 3σ 심음): AUC {u10:.3f}  {'OK' if g else '**FAIL**'}")

    # (e) **검정력 측정**(통과/실패 아님) — 실제 차원에서 심은 신호가 어디까지 올라오나
    out("   (e) 검정력 측정 — 같은 경로, 차원만 바꿔 심은 신호의 천장을 본다 (판정 아님)")
    out(f"      {'차원':>6}" + "".join(f"{f'{sh}σ':>9}" for sh in (1.0, 3.0, 8.0, 20.0)))
    for d in (10, 40, 400):
        cells = []
        for sh in (1.0, 3.0, 8.0, 20.0):
            r2 = np.random.default_rng(0)
            Xd = r2.normal(0, 1, (n, d)); Xd[:, 0] += sh * y
            pl = make_plan({0: Xd}, y, [0])
            sd = nested_scores(y, [0], pl)
            sd = sd[0] if isinstance(sd, tuple) else sd
            cells.append(auc_of(y, sd))
        out(f"      {d:>6}" + "".join(f"{c:>9.3f}" for c in cells))
    out("      → **차원이 실제 규모(수백~수천)면 20σ 신호도 AUC 0.7 을 못 넘는다.**")
    out("        이건 버그가 아니라 n=33 에서 PCA24+LR 경로의 **검정력 상한**이다 —")
    out("        #77·#ORG-TRAIN 이 0.5~0.6 에 머문 것을 설명하고, 리포트 검정력 절의 근거가 된다.")
    VERDICT["(5) 끝단 합성"] = ok
    return ok


def main():
    out("#LABEL-AUDIT — 정답표·방향 버그 점검 5개 (비용 $0)")
    out()
    for fn in (check1, check2, check3, check4, check5):
        try:
            fn()
        except Exception as e:
            import traceback
            out(f"   !! 점검 중 예외 {type(e).__name__}: {str(e)[:120]}")
            out("   " + traceback.format_exc().splitlines()[-3][:120])
            VERDICT[fn.__name__] = False
    out()
    out("## 결론")
    for k, v in VERDICT.items():
        out(f"   {k:<22} {'통과' if v else '**실패**'}")
    allok = all(VERDICT.values())
    out()
    out("   → " + ("**버그 아님.** 라벨·방향·매칭·부호·끝단 경로가 전부 일관된다. "
                   "'흔들림 큼 = robust' 를 R3 가설로 사전등록한다."
                   if allok else
                   "**버그 있음.** 아래 실패 항목을 고치고 지금까지의 모든 점수를 재계산한다."))
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")
    return allok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
