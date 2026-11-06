"""부분집합을 바꾸면 AUC 가 얼마나 움직이나 — **설계 비교의 분해능을 측정한다.** GPU 0. (지휘 #VAR-CHK)

## 왜 이 측정이 필요한가
벡터 설계 탐색(`probe_variant_eval.py`)에서 같은 설계·같은 라벨인데 **행을 20~30% 줄이자**
Olmo `mean` 이 **0.638 → 0.867** 로 움직였다(#77 전체 31행 → rename 가능 23행). 설계 차이로
해석한 숫자들이 그보다 작으면(예: diff_mean 이 mean 보다 ±0.1) 그 비교는 **분해능 밖**이다.

그래서 '행을 무작위로 같은 수만큼 떨어뜨리면 AUC 가 어떻게 흩어지는가' 를 직접 센다.
`has_var==1` 부분집합은 무작위가 아니지만(rename 가능 문제), **무작위 추출의 폭 안에 들어가면
'설계 때문' 도 'rename 가능 여부 때문' 도 주장할 수 없다** — 그게 이 측정의 결론 형식이다.

평가 경로는 `probe_final_eval` 의 것을 그대로 쓴다(층·C 폴드 내부 선택, PCA 24, GroupKFold 대리).
셔플 null 은 돌리지 않는다 — 여기서 묻는 것은 '우연과 구분되나' 가 아니라 '표본을 바꾸면 얼마나
움직이나' 이고, 그건 라벨을 섞지 않고 **행을 다시 뽑아서** 센다.

사용: python analysis/probe_subset_shift.py --draws 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from probe_final_eval import auc_of, make_plan, nested_scores, resid  # noqa: E402

OUT = ROOT / "experiments" / "results" / "probe_subset_shift.txt"
P = ROOT / "experiments" / "results" / "probe"
SETS = {
    "Skywork": ("Skywork-OR1-Math-7B_own_k8.npz", "Skywork-OR1-Math-7B_own_k8_var.npz"),
    "Olmo": ("Olmo-3-7B-Think_own_k8.npz", "Olmo-3-7B-Think_own_k8_var.npz"),
    "DeepSeek": ("DeepSeek-R1-0528-Qwen3-8B_own_k8.npz", "DeepSeek-R1-0528-Qwen3-8B_own_k8_var.npz"),
}

lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def one(feats, y, words, layers, folds=5):
    """한 부분집합의 AUC·길이잔차 AUC. 폴드 생성 불가면 None."""
    plan = make_plan(feats, y, layers, folds, 0)
    if plan is None:
        return None
    s, _ = nested_scores(y, layers, plan)
    return auc_of(y, s), auc_of(y, resid(s, np.log(np.maximum(1, words))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=10,
                    help="추출은 서로 독립이고 시드로 고정돼 있어 나눠도 같은 집합이 나온다")
    a = ap.parse_args()

    out("부분집합 교체가 AUC 를 얼마나 움직이나 — **설계 비교의 분해능**. 탐색 표기.")
    out()
    out("설계·라벨·평가경로를 고정하고 **행만** 무작위로 줄인다(rename 가능 행 수와 같은 개수).")
    out(f"무작위 추출 {a.draws}회 · 특징은 #77 주설계(원본 본문 평균 풀링) 그대로")
    out()
    for alias, (fn, fnv) in SETS.items():
        z = np.load(P / fn, allow_pickle=True)
        zv = np.load(P / fnv, allow_pickle=True)
        layers = [int(x) for x in z["layers"]]
        y = np.asarray(z["y"], int)
        words = np.asarray(z["words"], float)
        feats = {l: z[f"h{l}"] for l in layers}
        n = len(y)
        #  rename 가능 행을 pidx 로 맞춘다 — 두 파일의 행 순서가 같다는 가정을 쓰지 않는다.
        pid, pidv = list(z["pidx"]), list(zv["pidx"])
        hv = dict(zip(pidv, np.asarray(zv["has_var"], int)))
        sel = np.array([hv.get(p, 0) == 1 for p in pid])
        m = int(sel.sum())
        full = one(feats, y, words, layers, a.folds)
        sub = one({l: v[sel] for l, v in feats.items()}, y[sel], words[sel], layers, a.folds)
        out(f"## {alias} · 전체 {n}행 → 부분 {m}행 (rename 가능)")
        out(f"   전체 {n}행      AUC {full[0]:.3f} · 길이잔차 {full[1]:.3f}   ← #77 주분석의 그 값")
        if sub is None:
            out("   rename 가능 행에서 폴드 생성 불가")
            out()
            continue
        out(f"   rename 가능 {m}행 AUC {sub[0]:.3f} · 길이잔차 {sub[1]:.3f}   ← 설계 탐색의 기준선")
        #  추출 집합을 **먼저 같은 순서로 만들고** 나눈다 — 병렬이 결과를 바꾸지 않게.
        rng = np.random.default_rng(0)
        draws = [rng.permutation(n)[:m] for _ in range(a.draws)]

        def run(idx):
            return one({l: v[idx] for l, v in feats.items()}, y[idx], words[idx], layers, a.folds)

        if a.jobs and a.jobs != 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=a.jobs, prefer="processes")(delayed(run)(i) for i in draws)
        else:
            res = [run(i) for i in draws]
        v = np.array([r[0] for r in res if r is not None])
        lo, hi = np.percentile(v, [2.5, 97.5])
        #  부분집합 값이 무작위 추출 분포의 어디에 있나 (양쪽 꼬리 비율)
        pr = float(np.mean(v >= sub[0]))
        out(f"   무작위 {m}행 {len(v)}회: 중앙 {np.median(v):.3f} · 95% [{lo:.2f}, {hi:.2f}] · "
            f"폭 {hi - lo:.2f} · SD {v.std():.3f}")
        out(f"   → rename 가능 행의 {sub[0]:.3f} 보다 높은 무작위 추출이 **{pr:.0%}** "
            f"{'(분포 안 — 부분집합 효과와 구분 불가)' if 0.025 < pr < 0.975 else '(분포 밖 — 행 선택이 특별하다)'}")
        out()
    out("## 읽는 법")
    out("   95% 폭이 설계 간 AUC 차이보다 크면 그 설계 비교는 **분해능 밖**이다 —")
    out("   같은 설계로도 행만 바꿔서 그만큼 움직인다는 뜻이므로.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
