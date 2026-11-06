"""#GEN-PROBE 평가 — '풀이 도중 벡터' 프로브. **#77 프로토콜 그대로** + 지휘 지정 기준선 셋.

평가 경로는 `probe_final_eval` 의 함수를 그대로 import 한다(폴드·투영·nested 선택·null 생성).
여기서 다시 구현하면 #77 숫자와 비교가 안 된다.

## 기준선 셋도 **같은 폴드·같은 경로**로 돈다
1차원 특징(base_acc · greedy 정답여부 · greedy 토큰수)을 `make_plan` 에 그대로 넣으면
PCA 는 건너뛰어지고(`X.shape[1] > NPC` 가 False) 같은 nested 로지스틱이 돈다.
기준선을 다른 코드로 재면 '특징이 좋은가' 와 '평가경로가 다른가' 가 섞인다.

## 판정 (사전등록 #GEN-PROBE, 결과 보기 전 고정)
둘 다 만족해야 신호 있음: (1) 모델평균 AUC > 셔플 null 95% 상단 · (2) 기준선 셋 전부보다 높음.
길이잔차는 같이 적지만 **게이트가 아니다**(기준선 3 이 길이를 직접 통제한다).

사용: python analysis/probe_gen_eval.py --win 512 --perms 1000 --jobs 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from probe_final_eval import (auc_of, bacc_of, make_plan, nested_scores,  # noqa: E402
                              null_band, resid)

P = ROOT / "experiments" / "results" / "probe"
OUT = ROOT / "experiments" / "results" / "probe_gen_report.txt"
MODELS = [("Skywork", "Skywork-OR1-Math-7B"), ("Olmo", "Olmo-3-7B-Think"),
          ("DeepSeek", "DeepSeek-R1-0528-Qwen3-8B")]
BASELINES = [("기준선1 base_acc(k=8)", "base_acc"), ("기준선2 greedy 정답여부", "correct"),
             ("기준선3 greedy 토큰수", "n_tok")]
lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def one(feats, y, layers, words, perms, jobs):
    """nested 점수 → (AUC, 길이잔차 AUC, bacc, 고른 (층,C), null 상단 둘)"""
    plan = make_plan(feats, y, layers, 5, 0)
    if plan is None:
        return None
    s, picks = nested_scores(y, layers, plan)
    lw = np.log(np.maximum(1, words))
    nr, nd = null_band(y, layers, plan, words, perms, jobs)
    return dict(auc=auc_of(y, s), res=auc_of(y, resid(s, lw)), bacc=bacc_of(y, s), picks=picks,
                hi=float(np.percentile(nr, 95)), lo=float(np.percentile(nr, 5)),
                hi_r=float(np.percentile(nd, 95)), lo_r=float(np.percentile(nd, 5)), s=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out(f"#GEN-PROBE — 풀이 도중 벡터 프로브 · **창 앞 {a.win} 토큰** · null {a.perms}회")
    out("주라벨 `label`(max_drop) · gen_k==8 · #77 프로토콜 과 동일한 폴드·PCA24·C격자·층 내부CV")
    out()
    aucs, ress, his, his_r, base = [], [], [], [], {k: [] for _, k in BASELINES}
    for alias, fn in MODELS:
        f = P / f"{fn}_gen{a.win}{a.suffix}.npz"
        if not f.is_file():
            out(f"## {alias} — 특징 파일 없음 ({f.name}) → 건너뜀")
            out()
            continue
        z = np.load(f, allow_pickle=True)
        layers = [int(x) for x in z["layers"]]
        y = np.asarray(z["y"], int)
        k8 = np.asarray(z["gen_k"], int) == 8
        if not k8.all():
            out(f"   (gen_k!=8 {int((~k8).sum())}행 제외 — Amendment 2)")
        y, words = y[k8], np.asarray(z["words"], float)[k8]
        feats = {l: z[f"h{l}"][k8] for l in layers}
        out(f"## {alias} · {f.name} · n {len(y)} (robust {int(y.sum())} / spurious {int((1 - y).sum())}) "
            f"· 쓴 풀이토큰 중앙 {int(np.median(np.asarray(z['n_sol_used'])[k8]))}")
        r = one(feats, y, layers, words, a.perms, a.jobs)
        if r is None:
            out("   폴드 생성 불가 (소수 클래스 2 미만) → 판정 불가")
            out()
            continue
        mark = " ▲" if r["auc"] > r["hi"] else ""
        out(f"   AUC **{r['auc']:.3f}** · null 95% [{r['lo']:.2f}, {r['hi']:.2f}]{mark}")
        out(f"   길이잔차 AUC {r['res']:.3f} · null [{r['lo_r']:.2f}, {r['hi_r']:.2f}]  (게이트 아님)")
        out(f"   balanced acc {r['bacc']:.3f} · 고른 (층, C) {r['picks']}")
        aucs.append(r["auc"]); ress.append(r["res"]); his.append(r["hi"]); his_r.append(r["hi_r"])
        # 민감도 라벨
        for nm, key in (("label_fin", "y_fin"), ("label_major", "y_major")):
            yy = np.asarray(z[key], int)[k8]
            ok = yy >= 0
            if ok.sum() < 4 or len(set(yy[ok].tolist())) < 2 or int(min(np.bincount(yy[ok]))) < 2:
                out(f"   민감도 {nm}: **적용 불가** (라벨 수·클래스 부족)")
                continue
            out(f"   민감도 {nm}: 같은 점수로 AUC {auc_of(yy[ok], r['s'][ok]):.3f} "
                f"(robust {int(yy[ok].sum())} / spurious {int((1 - yy[ok]).sum())})")
        # 기준선 — 같은 폴드·같은 nested 경로, 1차원
        for nm, key in BASELINES:
            X = np.asarray(z[key], float)[k8].reshape(-1, 1)
            if np.std(X) == 0:
                out(f"   {nm}: 분산 0 → 판정 불가")
                base[key].append(float("nan"))
                continue
            pb = make_plan({0: X}, y, [0], 5, 0)
            sb, _ = nested_scores(y, [0], pb)
            v = auc_of(y, sb)
            base[key].append(v)
            out(f"   {nm}: AUC {v:.3f}  (단독 순위 AUC {auc_of(y, X[:, 0]):.3f})")
        out()

    if not aucs:
        out("모델이 하나도 없다 → 판정 불가")
    else:
        ma, mr = float(np.mean(aucs)), float(np.mean(ress))
        out("## 주지표 — 주라벨 모델별 AUC 의 **모델평균**")
        out(f"   모델평균 AUC **{ma:.3f}** · null 상단 평균 {np.mean(his):.2f} "
            f"→ 기준1 {'통과' if ma > np.mean(his) else '**미달**'}")
        out(f"   모델평균 길이잔차 {mr:.3f} · null 상단 평균 {np.mean(his_r):.2f} (게이트 아님)")
        out()
        out("## 기준선 모델평균 (지휘 지정)")
        bm = {}
        for nm, key in BASELINES:
            v = [x for x in base[key] if not np.isnan(x)]
            bm[nm] = float(np.mean(v)) if v else float("nan")
            out(f"   {nm}: 모델평균 AUC {bm[nm]:.3f}")
        worse = [nm for nm, v in bm.items() if not (ma > v)]
        out(f"   → 기준2 {'통과' if not worse else '**미달** (' + ', '.join(worse) + ' 이상)'}")
        out()
        out("## 판정")
        ok1, ok2 = ma > np.mean(his), not worse
        out(f"   {'**신호 있음**' if (ok1 and ok2) else '**신호 미입증**'} "
            f"(기준1 {'O' if ok1 else 'X'} · 기준2 {'O' if ok2 else 'X'})")
    p = Path(a.out) if a.out else OUT
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {p}")


if __name__ == "__main__":
    main()
