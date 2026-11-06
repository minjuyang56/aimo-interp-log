"""#SUB2048-PLUS — sub2048 개선 후보 평가. 저장된 공식 33 특징만 쓴다. $0 · GPU 0.

사전등록은 CLAUDE.md `## #SUB2048-PLUS` 절(커밋 277086f, **평가 전**)이다. 요약:
  원판  sub2048 고정 임계 (#ORG-LABEL 시험2 에서 교정 불량 — True 비율 0.15~0.21)
  **A**  모델별 순위 50% 컷 — 자유 파라미터 0. True 비율이 정의상 0.5
  **B**  rank(프로브) + rank(−log 단어수) 1:1 고정 평균 → 임계는 **LOO 사전확률 교정**
  (C)   조건 미충족(공식 33 특징은 창 2048 하나 · 학습 특징에 1024 창 없음) → 뺐다
기준선: 상수 '전부 robust' 0.6061 · 길이 h15C **같은 LOO 절차** · 원판 고정 임계.
검문소: 문제단위 셔플 null 1000 (라벨만 섞고 **같은 판정 절차**를 다시 돌린다).
빌드 조건: 정확도 > 0.65 **그리고** 자기 null 95% 상단 초과. 둘 다.

사용: .venv/Scripts/python.exe analysis/sub2048_plus_eval.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))

from org_sub2048_transfer import ART, FEAT, MODELS, auc as auc_rob, probe_score, resid
from ttp_eval import thr_prior

REPORT = ROOT / "experiments/results/sub2048_plus_report.txt"
PERMS = 1000
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


# ─────────────────────── 판정 규칙 셋 (라벨을 쓰는 곳은 LOO 안으로 묶인다) ───────────────────────

def pred_tophalf(score):
    """(A) 호출 안에서 점수 상위 ceil(n/2) = True. **라벨을 보지 않는다.**"""
    n = len(score)
    k = -(-n // 2)
    ordr = np.argsort(-np.asarray(score, float), kind="stable")
    p = np.zeros(n, bool)
    p[ordr[:k]] = True
    return p


def pred_loo_prior(score, y):
    """점수 큼 = robust 규약에서 LOO 사전확률 교정. `ttp_eval.thr_prior` 를 그대로 쓴다
    (그쪽은 spur 규약이라 **부호를 뒤집어** 넘긴다 — 같은 함수여야 R1~R5 와 절차가 같다)."""
    spur = -np.asarray(score, float)
    y = np.asarray(y, int)
    n = len(y)
    p = np.zeros(n, bool)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        p[i] = spur[i] <= thr_prior(spur[m], y[m])
    return p


def rankz(x):
    """모델 안 순위를 [0,1] 로. 동점은 평균 순위."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x), float)
    i = 0
    sv = x[o]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r / max(1, len(x) - 1)


def acc_of(pred, y):
    return float((np.asarray(pred, bool) == (np.asarray(y, int) == 1)).mean())


def band(rule, score, y, perms=PERMS, seed=0):
    """라벨만 섞고 **같은 판정 절차**를 다시 돌린 정확도의 95% 밴드."""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(perms):
        ys = rng.permutation(np.asarray(y, int))
        accs.append(acc_of(rule(score, ys), ys))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    from h21_uncertainty import problems
    probs = problems()
    ptext = {r["problem"]: r for r in probs}
    art = json.loads(ART.read_text(encoding="utf-8"))

    out("#SUB2048-PLUS — sub2048 개선 후보 (사전등록 1회 · $0 · 저장된 공식 33 특징만)")
    out(f"   공식 train-main-v2 문제단위 라벨 {len(probs)}문제 "
        f"(robust {sum(r['y'] for r in probs)} / spurious {len(probs) - sum(r['y'] for r in probs)})")
    out(f"   가중치 {ART.relative_to(ROOT).as_posix()} 그대로 · 창 {art['win']} · PCA {art['npc']} · 재적합 없음")
    out("   빌드 조건: 정확도 > 0.65 **그리고** 자기 셔플 null 95% 상단 초과 (둘 다)")
    out()

    acc_tab, auc_tab, tr_tab = {}, {}, {}
    bands = {}

    for mid in MODELS:
        alias = mid.split("/")[-1].replace(".", "_")
        p = FEAT / f"{alias}_org33.npz"
        if not p.is_file():
            out(f"## {alias} — 특징 파일 없음 → 건너뜀")
            continue
        d = np.load(p, allow_pickle=True)
        am = art["per_model"][mid]
        if int(d["layer"]) != int(am["layer"]):
            raise SystemExit(f"층 불일치 {int(d['layer'])} vs {am['layer']}")
        sc = probe_score(am, d["X"])                       # 큼 = robust
        pt = [str(x) for x in d["problems"]]
        y = np.array([ptext[t]["y"] for t in pt], int)
        lw = np.array([math.log(max(1, len(t.split()))) for t in pt])
        slen = -lw                                          # 짧을수록 robust (h15C 점수식)
        scb = 0.5 * (rankz(sc) + rankz(slen))               # B: 1:1 고정 순위 평균

        out(f"## {alias} · 층 {int(d['layer'])} · n {len(y)}")
        for name, score, rule in (
            ("원판 고정임계", sc, lambda s, yy, t=float(am["thr"]): np.asarray(s) >= t),
            ("A 순위 50% 컷", sc, lambda s, yy: pred_tophalf(s)),
            ("B 프로브+길이 순위평균", scb, pred_loo_prior),
            ("기준선 프로브 LOO교정", sc, pred_loo_prior),
            ("기준선 길이 h15C LOO교정", slen, pred_loo_prior),
        ):
            pred = rule(score, y)
            a, A = acc_of(pred, y), auc_rob(y, score)
            lo, hi = band(rule, score, y)
            Ar = auc_rob(y, resid(np.asarray(score, float), lw))
            flag = " ▲" if a > hi else ""
            out(f"   {name:<24} 정확도 {a:.3f}{flag} · null95 [{lo:.3f}, {hi:.3f}] · "
                f"AUC {A:.3f} · 길이잔차 {Ar:.3f} · True비율 {pred.mean():.3f}")
            acc_tab.setdefault(name, {})[alias] = a
            auc_tab.setdefault(name, {})[alias] = A
            tr_tab.setdefault(name, {})[alias] = float(pred.mean())
            bands.setdefault(name, {})[alias] = (lo, hi)
        out()

    const = float(np.mean([r["y"] for r in probs]))
    out("## 모델평균 (주지표는 A·B 두 칸)")
    out(f"   {'방법':<24}{'정확도':>8}{'null상단':>9}{'AUC':>8}{'True비율':>9}  판정")
    for name in acc_tab:
        a = float(np.mean(list(acc_tab[name].values())))
        hi = float(np.mean([bands[name][k][1] for k in bands[name]]))
        A = float(np.mean(list(auc_tab[name].values())))
        tr = float(np.mean(list(tr_tab[name].values())))
        verdict = "빌드 조건 통과" if (a > 0.65 and a > hi) else (
            "null 넘지만 0.65 미달" if a > hi else ("0.65 넘지만 null 안" if a > 0.65 else "미달"))
        out(f"   {name:<24}{a:>8.3f}{hi:>9.3f}{A:>8.3f}{tr:>9.3f}  {verdict}")
    out(f"   {'상수 전부 robust':<24}{const:>8.3f}{'—':>9}{0.5:>8.3f}{1.0:>9.3f}  기준선")
    out()
    out("## 읽는 법 (사전등록에 적어 둔 그대로)")
    out("   * A 는 점수 **순위만** 쓰므로 AUC 가 원판과 같다 — 바뀌는 것은 정확도·True 비율뿐이다.")
    out("     즉 A 는 '신호를 늘리는 개선' 이 아니라 **'교정을 고치는 개선'** 이다.")
    out("   * null 밴드는 **라벨만 섞고 같은 판정 절차를 다시 돌린** 정확도 분포다. A 는 라벨을")
    out("     보지 않으므로 그 밴드가 '상수·순위컷이 우연히 맞을 범위' 를 그대로 보여준다.")
    out("   * 같은 테스트셋(공식 33)을 여러 번 본 오염이 있다 → 통과해도 '이 셋에서의 값' 으로만 적는다.")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
