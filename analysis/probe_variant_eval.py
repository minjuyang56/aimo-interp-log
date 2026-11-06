"""프로브 **벡터 설계** 탐색 평가 — 같은 k=8 라벨 · 같은 nested CV · 같은 null. (지휘 #PROBE-AUDIT)

**전부 탐색이다. 확증이 아니다.** #77 주지표는 1회 실행으로 확정됐고(모델평균 AUC 0.616,
길이잔차 0.559 → 미입증) 여기서 나오는 어떤 숫자로도 그 판정을 고치지 않는다. 묻는 것은 하나다:
**신호가 없는 것인가, 아니면 우리 벡터가 신호를 담지 못한 것인가.**

비교하는 설계 다섯 (모두 `probe_final_eval` 의 make_plan/nested_scores/null_band 를 그대로 쓴다):
  `mean`        원본, 문제 본문 평균 풀링 (= #77 이 쓴 벡터)
  `last`        원본, **마지막 입력 토큰** (= 주최측 probing 베이스라인의 위치)
  `diff_mean`   h(원본) − h(rename) — **라벨이 '원본 vs 변형 차이' 이므로 가장 직접적인 대조 벡터**
  `diff_last`   같은 차이를 마지막 토큰에서
  `cat_mean`    [h(원본), h(rename)] 결합

**부분집합을 맞추는 것이 이 비교의 핵심이다.** rename 을 못 만드는 문제가 20~30% 있어
(Skywork 6/30 · Olmo 8/31 · DeepSeek 7/23) 차이 벡터가 0 이 된다. 0 으로 채우면 분류기가
'rename 가능 여부' 를 읽어 버리므로 **`has_var==1` 행만** 쓰고, 비교 대상인 `mean`·`last` 도
**같은 행으로 다시** 계산한다. 그러지 않으면 '설계 차이' 와 '표본 차이' 가 섞인다.

사용: python analysis/probe_variant_eval.py --perms 1000 --jobs 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from probe_final_eval import (auc_of, bacc_of, make_plan, nested_scores,  # noqa: E402
                              null_band, resid)

OUT = ROOT / "experiments" / "results" / "probe_variant_report.txt"
P = ROOT / "experiments" / "results" / "probe"
SETS = {
    "Skywork": "Skywork-OR1-Math-7B_own_k8_var.npz",
    "Olmo": "Olmo-3-7B-Think_own_k8_var.npz",
    "DeepSeek": "DeepSeek-R1-0528-Qwen3-8B_own_k8_var.npz",
}

lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def build(z, layers, kind, sel):
    """설계별 특징 사전 {층: (n, d)}. `sel` 로 행을 먼저 고른다."""
    f = {}
    for l in layers:
        h, hl = z[f"h{l}"][sel], z[f"hl{l}"][sel]
        v, vl = z[f"v{l}"][sel], z[f"vl{l}"][sel]
        f[l] = {"mean": h, "last": hl, "diff_mean": h - v, "diff_last": hl - vl,
                "cat_mean": np.hstack([h, v])}[kind]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    out("프로브 벡터 설계 탐색 — **탐색 표기. #77 판정(미입증)을 고치지 않는다.**")
    out()
    out(f"같은 k=8 라벨 · 같은 nested CV(층·C 폴드 내부 선택, PCA 24) · null {a.perms}회 · "
        f"`has_var==1` 행만(차이 벡터가 정의되는 행)")
    out("참조: #77 주지표(전체 행, mean) 모델평균 AUC **0.616** · 길이잔차 0.559 → 미입증")
    out()

    kinds = ["mean", "last", "diff_mean", "diff_last", "cat_mean"]
    table = {k: {} for k in kinds}
    for alias, fn in SETS.items():
        z = np.load(P / fn, allow_pickle=True)
        layers = [int(x) for x in z["layers"]]
        sel = np.asarray(z["has_var"], int) == 1
        y = np.asarray(z["y"], int)[sel]
        words = np.asarray(z["words"], float)[sel]
        out(f"## {alias} · rename 가능 {int(sel.sum())}/{len(sel)}행 · "
            f"robust {int(y.sum())} / spurious {int((1 - y).sum())}")
        if len(set(y.tolist())) < 2 or int(min(np.bincount(y))) < 2:
            out("   소수 클래스가 2 미만 → 이 모델은 평가 불가")
            out()
            continue
        for kind in kinds:
            fs = build(z, layers, kind, sel)
            plan = make_plan(fs, y, layers, a.folds, 0)
            if plan is None:
                out(f"   {kind:<10} 폴드 생성 불가")
                continue
            s, picks = nested_scores(y, layers, plan)
            auc = auc_of(y, s)
            rauc = auc_of(y, resid(s, np.log(np.maximum(1, words))))
            nl, nlr = null_band(y, layers, plan, words, a.perms, a.jobs)
            lo, hi = np.percentile(nl, [2.5, 97.5])
            rlo, rhi = np.percentile(nlr, [2.5, 97.5])
            table[kind][alias] = (auc, hi, rauc, rhi)
            out(f"   {kind:<10} AUC **{auc:.3f}** [null {lo:.2f},{hi:.2f}]{' ▲' if auc > hi else '  '}"
                f" · 길이잔차 {rauc:.3f} [{rlo:.2f},{rhi:.2f}]{' ▲' if rauc > rhi else ''}"
                f" · bacc {bacc_of(y, s):.3f} · 층 {[p[0] for p in picks]}")
        out()

    out("## 설계별 모델평균 (주지표와 같은 방식으로 집계)")
    out(f"   {'설계':<12}{'AUC 평균':>9}{'null상단 평균':>14}{'길이잔차 평균':>14}{'상단':>8}  판정")
    for kind in kinds:
        v = table[kind]
        if not v:
            continue
        ma = float(np.mean([x[0] for x in v.values()]))
        mh = float(np.mean([x[1] for x in v.values()]))
        mr = float(np.mean([x[2] for x in v.values()]))
        mrh = float(np.mean([x[3] for x in v.values()]))
        verdict = "둘 다 초과" if (ma > mh and mr > mrh) else ("AUC만 초과" if ma > mh else "null 안")
        out(f"   {kind:<12}{ma:>9.3f}{mh:>14.2f}{mr:>14.3f}{mrh:>8.2f}  {verdict}")
    out()
    out("## 읽는 법")
    out("   `mean` 이 이 부분집합에서의 기준선이다 — #77 의 0.616 과 다르면 그건 설계 차이가 아니라")
    out("   **행이 20~30% 줄어든 효과**다. 설계 비교는 반드시 같은 열끼리 한다.")
    out("   어느 설계든 '둘 다 초과' 가 아니면 사전등록 기준으로는 신호가 아니고, 여기서는 애초에")
    out("   탐색이므로 **통과해도 확증으로 쓰지 않는다** — 다음 단계는 독립 라벨에서의 확증이다.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
