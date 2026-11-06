"""외부 테스트의 **층 선택 민감도** — 0.792 가 특별한 층 하나인가, 전 층에서 나오는가. GPU 0. 탐색.

## 왜 이걸 따로 재나
#77 외부 테스트(보조)에서 **DeepSeek 가 자체 라벨 교차검증 0.326(우연 아래)인데 공식 33 전이는
0.792 ▲**(길이잔차 0.758)로 나왔다. 평가기는 층·C 를 **자체 라벨 AUC 로** 고르는데 그 AUC 자체가
0.326 이라 선택 기준이 사실상 무정보다 → **네 층 중 하나를 거의 무작위로 고른 셈**이고, 그러면
0.792 는 '선택 운' 일 수 있다. 전 층을 다 적어 보면 그게 갈린다:
  * 전 층이 비슷하게 높으면 → 선택 운이 아니라 그 모델 특징에 공식 라벨용 신호가 있다.
  * 고른 층만 높으면 → 네 번 뽑아 한 번 맞은 것. 주장하지 않는다.

같이 재는 것 둘:
  * **부호** — 자체 라벨에서 0.5 아래(0.326)인 프로브가 외부에서 0.5 위로 가는 것은 **라벨-길이
    방향이 두 셋에서 반대**(공식 33 짧을수록 robust 0.669 · 자체 31단어 셋 길수록 robust 0.372,
    H23 길이대리 표)일 때 자연히 생긴다. 그래서 외부 점수와 log(단어수) 상관을 같이 적는다.
  * **길이잔차** — 순수 길이 대리면 잔차에서 무너져야 한다. 안 무너지면 길이만은 아니다.

사용: python analysis/probe_external_layers.py
      python analysis/probe_external_layers.py --sets 별칭=자체npz,외부npz --out <파일>
        (#DS-REP 반복본을 같은 코드로 재기 위한 인자다 — 계산은 한 줄도 바뀌지 않는다.)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from probe_final_eval import NPC, auc_of, make_plan, nested_scores, resid  # noqa: E402

OUT = ROOT / "experiments" / "results" / "probe_external_layers.txt"
P = ROOT / "experiments" / "results" / "probe"
SETS = {
    "Skywork": ("Skywork-OR1-Math-7B_own_k8.npz", "Skywork-OR1-Math-7B_official_v2.npz"),
    "Olmo": ("Olmo-3-7B-Think_own_k8.npz", "Olmo-3-7B-Think_official.npz"),
    "DeepSeek": ("DeepSeek-R1-0528-Qwen3-8B_own_k8.npz", "DeepSeek-R1-0528-Qwen3-8B_official.npz"),
}
CS = (0.003, 0.01, 0.03, 0.1, 0.3)

lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=None, metavar="ALIAS=OWN.npz,EXT.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sets = SETS
    if a.sets:
        sets = {}
        for item in a.sets:
            alias, pair = item.split("=", 1)
            fo, fe = pair.split(",", 1)
            sets[alias] = (fo, fe)
    outfile = Path(a.out) if a.out else OUT

    out("외부 테스트 층 선택 민감도 — **탐색**(사전등록은 층 하나를 자체 라벨로 고르는 것이다).")
    out("#77 판정(미입증)을 고치지 않는다. 묻는 것: 0.792 가 층 선택 운인가.")
    out()
    for alias, (fo, fe) in sets.items():
        zo = np.load(P / fo, allow_pickle=True)
        ze = np.load(P / fe, allow_pickle=True)
        layers = [int(x) for x in zo["layers"]]
        yo = np.asarray(zo["y"], int)
        wo = np.asarray(zo["words"], float)
        ye = np.asarray(ze["y"], int)
        we = np.asarray(ze["words"], float)
        lw = np.log(np.maximum(1, we))
        out(f"## {alias} · 자체 {len(yo)}행 학습 → 공식 33 적용 "
            f"(공식: robust {int(ye.sum())} / spurious {int((1 - ye).sum())})")
        out(f"   공식 33 에서 log(단어수) 단독 → robust AUC {auc_of(ye, -lw):.3f} "
            f"(짧을수록 robust 방향) · 자체 셋 {auc_of(yo, -np.log(np.maximum(1, wo))):.3f}")
        out(f"   {'층':>4}{'자체 nested':>13}{'외부 AUC':>11}{'길이잔차':>10}{'ρ(점수,log단어)':>16}  C")
        for l in layers:
            fs = {l: zo[f"h{l}"]}
            plan = make_plan(fs, yo, [l], 5, 0)
            if plan is None:
                out(f"   {l:>4}  폴드 생성 불가")
                continue
            so, pk = nested_scores(yo, [l], plan)
            c = pk[0][1]
            sc = StandardScaler().fit(zo[f"h{l}"])
            Z = sc.transform(zo[f"h{l}"])
            pca = PCA(n_components=min(NPC, len(yo) - 1, Z.shape[1]), random_state=0).fit(Z)
            m = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(pca.transform(Z), yo)
            se = m.decision_function(pca.transform(sc.transform(ze[f"h{l}"])))
            out(f"   {l:>4}{auc_of(yo, so):>13.3f}{auc_of(ye, se):>11.3f}"
                f"{auc_of(ye, resid(se, lw)):>10.3f}{np.corrcoef(se, lw)[0, 1]:>16.3f}  {c}")
        out()
    out("## 읽는 법")
    out("   '자체 nested' 가 0.5 근처거나 아래면 층 선택 기준이 무정보다 → 외부 값은 네 번 중 한 번이다.")
    out("   외부가 전 층에서 비슷하게 높으면 선택 운이 아니고, 한 층만 높으면 주장할 수 없다.")
    out("   ρ 가 크면(±0.6 이상) 점수가 길이를 읽는 것이고, 그때 두 셋의 길이-라벨 방향이 반대라서")
    out("   **자체에서 0.5 아래 · 외부에서 0.5 위**가 동시에 나올 수 있다(H23 길이대리와 같은 기제).")
    outfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {outfile}")


if __name__ == "__main__":
    main()
