"""같은 문제 구간(40–79)을 3모델이 k=8 로 봤다 — **모델 간 라벨 차이**를 처음 센다. GPU 0.

## 왜 이게 이 프로젝트에서 가장 중요한 free 측정인가
공식 `train-main-v2` 는 2개 이상 모델에 걸친 문제 **11/11 이 라벨 동일**이었다(H2). 그래서
"라벨에 모델×문제 성분이 있는가" 를 **학습셋으로는 물을 수 없었다** — H21·H10-B 에서 '모델 간
분리력' 이 매번 '쌍 0개 → 판정 불가' 로 끝난 이유다. 자체 라벨은 (모델, 문제) 단위로 생기므로
그 질문에 처음 답할 수 있다. 세 가지를 센다:

1. **라벨 불일치율** — 두 모델 이상에 라벨이 있는 문제 중 서로 갈리는 비율.
   0 에 가까우면 문제가 라벨을 정한다(= 문제 특징으로 충분, 모델별 프로브가 불필요).
   크면 모델×문제 성분이 실재한다(= #77 분리력을 계산할 쌍이 생긴다).
2. **#77 분리력용 쌍 수** — (robust 모델, spurious 모델) 쌍이 20개 이상이어야 사전등록이
   숫자를 주장하게 허용한다.
3. **결정 변형 유형의 모델 간 일치** — 같은 문제가 여러 모델에서 spurious 일 때 같은 유형이
   무너뜨리는가. 같으면 '그 문제의 그 자리가 약하다'(문제 성질), 다르면 모델마다 다른 편집에
   걸린다(모델 성질). k=8 세 셋에서 결정 유형이 Skywork·Olmo 는 rename 쪽, DeepSeek 는
   typos 쪽으로 갈렸기 때문에 이 구분이 필요하다.

사용: python analysis/label_cross_model.py
"""

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "label_cross_model.txt"

#  **같은 문제 구간 · 같은 생성 예산(cap 32768) · 같은 k(8)** 만 비교한다.
#  섞으면 '모델 차이' 와 '예산 차이' 가 구별되지 않는다(지휘 #78 의 비교가능성 논거).
SETS = {
    "Skywork": "experiments/results/runpod/labelgen32k/pj7vh1qllei51i/out/"
               "Skywork-OR1-Math-7B_off40_v2_labels.json",
    "Olmo": "experiments/results/runpod/labelgen32k/ofclt769pgvn6u/out/"
            "Olmo-3-7B-Think_v3_labels.json",
    "DeepSeek": "experiments/results/runpod/labelgen32k/luvw8yu3mkxsza/out/"
                "DeepSeek-R1-0528-Qwen3-8B_v3_labels.json",
}

lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def main():
    # **DeepSeek 라벨 교체용 인자 (지휘 #DS-RELABEL).** 옛(손상) 보고서를 덮지 않으려고
    # 출력 파일에 접미사를 붙인다 — 나란히 비교하는 것이 이번 재평가의 목적이다.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=None, help="DeepSeek 라벨 json (기본: 옛 손상 파일)")
    ap.add_argument("--suffix", default="", help="출력 파일 접미사")
    a = ap.parse_args()
    global OUT
    if a.ds:
        SETS["DeepSeek"] = a.ds
        out(f"DeepSeek 라벨 교체: {a.ds}")
    if a.suffix:
        OUT = OUT.with_name(OUT.stem + a.suffix + OUT.suffix)
    rows = {}
    for m, f in SETS.items():
        rs = json.loads((ROOT / f).read_text(encoding="utf-8"))
        rows[m] = {r["pidx"]: r for r in rs if r.get("label") is not None}
        gk = {r.get("gen_k") for r in rows[m].values()}
        gc = {r.get("gen_cap") for r in rows[m].values()}
        out(f"{m}: 라벨 {len(rows[m])}문제 · gen_k {gk} · gen_cap {gc} · "
            f"spurious {sum(1 for r in rows[m].values() if r['label'] is False)}")
    out()

    # ---- 1) 불일치율
    out("## 1. 모델 간 라벨 불일치 (두 모델 이상에 라벨이 있는 문제)")
    shared = defaultdict(dict)
    for m, d in rows.items():
        for pidx, r in d.items():
            shared[pidx][m] = r["label"]
    multi = {p: v for p, v in shared.items() if len(v) >= 2}
    split = {p: v for p, v in multi.items() if len(set(v.values())) > 1}
    out(f"   두 모델 이상 {len(multi)}문제 · **갈리는 문제 {len(split)} "
        f"({len(split) / max(1, len(multi)):.0%})**")
    out("   비교: 공식 train-main-v2 는 11문제 중 **0건** 갈렸다 (H2).")
    out()
    out("   쌍별 일치율:")
    for a, b in combinations(SETS, 2):
        both = [p for p in shared if a in shared[p] and b in shared[p]]
        agree = sum(1 for p in both if shared[p][a] == shared[p][b])
        out(f"     {a:<9} vs {b:<9} n={len(both):>3} · 일치 {agree:>3} "
            f"({agree / max(1, len(both)):.0%})")
    out()

    # ---- 2) #77 분리력용 쌍
    out("## 2. #77 '모델 간 분리력' 에 쓸 쌍 수")
    pairs = []
    for p, v in split.items():
        rb = [m for m, y in v.items() if y is True]
        sp = [m for m, y in v.items() if y is False]
        pairs += [(p, r, s) for r in rb for s in sp]
    out(f"   (robust 모델, spurious 모델) 쌍 **{len(pairs)}개** · 사전등록 기준 20개 "
        f"→ {'계산 가능' if len(pairs) >= 20 else '**판정 불가**(숫자를 주장하지 않는다)'}")
    out(f"   쌍의 모델 구성: {dict(Counter(f'{r}>{s}' for _, r, s in pairs))}")
    out()

    # ---- 3) 결정 유형이 모델 간에 같은가
    out("## 3. 같은 문제가 여러 모델에서 spurious 일 때 결정 유형이 같은가")
    same = diff = 0
    detail = []
    for p in multi:
        sp = {m: rows[m][p] for m in rows if p in rows[m] and rows[m][p]["label"] is False}
        if len(sp) < 2:
            continue
        kinds = {m: r.get("worst_kind") for m, r in sp.items()}
        if len(set(kinds.values())) == 1:
            same += 1
        else:
            diff += 1
        detail.append((p, kinds))
    out(f"   둘 이상에서 spurious 인 문제 {same + diff} · 결정 유형 **같음 {same} / 다름 {diff}**")
    for p, k in detail[:12]:
        out(f"     pidx {p:>3}: {k}")
    out()
    # ---- 4) 그 차이가 '모델이 아는가' 가 아니라 **base_acc 교란**은 아닌가
    #  max_drop = base_acc - min(var_acc) 이므로 원본 정확도가 낮은 모델은 **떨어질 여지가 더 크다**.
    #  Amendment 1 이 걱정했던 그 교란이고, k=8 이 Skywork 안에서는 고쳤지만 **모델 사이**에서는
    #  고쳤는지 확인된 적이 없다. 갈리는 쌍마다 두 모델의 그 문제 base_acc 를 맞대 본다.
    #  spurious 쪽이 체계적으로 base_acc 가 낮으면, 라벨 차이는 '아는가' 가 아니라 '임계에 가까운가' 다.
    out("## 4. 갈리는 쌍이 base_acc 교란으로 설명되는가")
    lower = higher = tie = 0
    gaps = []
    for pidx, rb, sp in pairs:
        ab, asp = rows[rb][pidx]["base_acc"], rows[sp][pidx]["base_acc"]
        gaps.append(asp - ab)
        if asp < ab:
            lower += 1
        elif asp > ab:
            higher += 1
        else:
            tie += 1
    n = max(1, len(pairs))
    out(f"   spurious 쪽 base_acc 가 **더 낮음 {lower}** / 더 높음 {higher} / 같음 {tie} "
        f"(쌍 {len(pairs)})")
    out(f"   base_acc 차이(spurious − robust) 평균 {sum(gaps) / n:+.3f}")
    out("   교란이 전부라면 '더 낮음' 이 쌍 거의 전부를 차지해야 한다. 같음이 많으면 교란 아닌 성분이 있다.")
    #  모델별 base_acc 평균도 같이 — 모델 전체가 낮으면 그 모델이 spurious 를 많이 받는다
    out("   모델별 평균 base_acc: " + " · ".join(
        f"{m} {sum(r['base_acc'] for r in d.values()) / max(1, len(d)):.3f}" for m, d in rows.items()))
    out()
    out("## 읽는 법")
    if len(split) == 0:
        out("   라벨이 **문제 단위**라는 H2 가 자체 라벨에서도 재현됐다 → 모델별 프로브의 전제가 없다.")
    else:
        out(f"   공식 학습셋에서 0 이던 모델 간 라벨 차이가 자체 라벨에서는 {len(split)}건 생겼다.")
        out("   즉 **모델×문제 성분이 실재한다** — 다만 그 차이가 '모델 성질' 인지 '라벨 잡음' 인지는")
        out("   이 표가 가르지 못한다. 3번(결정 유형 일치)이 그 구분의 단서다: 같은 문제를 같은 유형이")
        out("   무너뜨리면 문제 성질, 모델마다 다른 유형이면 모델 성질 쪽이다.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
