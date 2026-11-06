"""#TOK-AUDIT — vLLM DeepSeek 디토큰화 손상이 **라벨 집계까지 번졌나**. GPU 0. 텍스트 없이 판정한다.

## (1) 코드 경로 — 동일하다
`genlab_v2.Gen.samples()` **119행**: `return [[(o.text, o.finish_reason == "stop") for o in out.outputs] …]`
`genlab_v2.acc_of()` **170행**: `n_hit = sum(1 for t, f in groups if f and same_answer(extract_answer(t), gold))`
  → `acc_of` 가 받는 `t` 는 **vLLM 이 디토큰화한 `o.text` 그 자체**다. 호출부 세 곳(**231** 선별 ·
    **250** 변형 · **328** 선별재사용 경로의 변형)이 모두 그 값을 쓴다.
`jobs/genprobe.py` **62행** `Gen, chat = ns["Gen"], ns["chat"]` · **84행** `o.outputs[0].text`
  → genprobe 는 **같은 클래스·같은 필드**를 썼고 거기서 41/41 손상이 관측됐다.
  같은 파드·같은 이미지·같은 vllm 0.31.0·같은 모델이므로 **라벨 생성도 같은 손상 텍스트를 채점했다.**

## (2) 텍스트가 없는데 어떻게 입증하나 — 자연실험
손상은 **boxed 안에 선행 공백이 있을 때만** 터진다(`\boxed{ 90}` → 추출 `Ġ90`). 그 습관은 문제마다
고정적이므로, greedy 에서 손상된 문제는 k=8 샘플에서도 손상됐을 것이다. 따라서:
  **greedy 가 Ġ 로 뒤집힌 문제군의 기록된 base_acc 가 그렇지 않은 문제군보다 낮아야 한다.**
이건 라벨 파일에 저장된 수(base_acc)와 genprobe greedy 만으로 계산된다 — 버린 텍스트가 필요 없다.
대조군: Skywork·Olmo 는 손상 0건이므로 같은 검정에서 차이가 없어야 한다.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "experiments/results/runpod/genprobe/oi8q4v8ia3ojq0/out"
LB = ROOT / "experiments/results/runpod/labelgen32k"
OUT = ROOT / "experiments/results/tok_audit_report.txt"
lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


SETS = {
    "DeepSeek": ("DeepSeek-R1-0528-Qwen3-8B",
                 ["luvw8yu3mkxsza/out/DeepSeek-R1-0528-Qwen3-8B_v3_labels.json",
                  "1f7jul0dgk7o9s/out/DeepSeek-R1-0528-Qwen3-8B_off80_v2_labels.json"]),
    "Skywork": ("Skywork-OR1-Math-7B", ["pj7vh1qllei51i/out/Skywork-OR1-Math-7B_off40_v2_labels.json"]),
    "Olmo": ("Olmo-3-7B-Think", ["ofclt769pgvn6u/out/Olmo-3-7B-Think_v3_labels.json"]),
}


def main():
    out(__doc__.strip())
    out()
    out("## (2) 자연실험 결과")
    for alias, (fn, labs) in SETS.items():
        gf = G / f"{fn}_greedy.json"
        gfx = G / f"{fn}_greedy_fixed.json"
        rows = json.loads(gf.read_text(encoding="utf-8"))
        bad = {r["pidx"]: ("Ġ" in r["greedy_text"] or "Ċ" in r["greedy_text"]) for r in rows}
        # '뒤집힌' 문제 = 손상본에서는 오답, 복원본에서는 정답
        if gfx.is_file():
            fx = {r["pidx"]: r for r in json.loads(gfx.read_text(encoding="utf-8"))}
            flip = {p: (not r0["correct"]) and fx[p]["correct"]
                    for p, r0 in ((r["pidx"], r) for r in rows)}
        else:
            flip = {p: False for p in bad}
        lab = {}
        for rel in labs:
            for r in json.loads((LB / rel).read_text(encoding="utf-8")):
                lab[r["pidx"]] = r
        pid = [p for p in flip if p in lab]
        a = np.array([lab[p]["base_acc"] for p in pid if flip[p]])
        b = np.array([lab[p]["base_acc"] for p in pid if not flip[p]])
        out(f"   {alias:9s} greedy 손상행 {sum(bad.values())}/{len(bad)} · Ġ로 뒤집힌 문제 **{int(sum(flip.values()))}**")
        if len(a) and len(b):
            # 순위검정(맨-휘트니 U) — n 이 작아 정규근사 대신 정확 분포를 쓴다
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(a, b, alternative="less")
            out(f"   {'':9s} 기록된 base_acc: 뒤집힌 군 평균 **{a.mean():.3f}** (n={len(a)}) vs "
                f"나머지 **{b.mean():.3f}** (n={len(b)}) · 차 {a.mean() - b.mean():+.3f} · "
                f"맨-휘트니 단측 p = **{p:.4f}**")
        else:
            out(f"   {'':9s} 뒤집힌 군이 없어 검정 불가 (대조군으로서 정상)")
    out()
    out("## 기록된 평균 base_acc vs greedy 복원 정답률 — 세 모델 비교")
    out(f"   {'모델':9s}{'기록 base_acc(k=8)':>20}{'greedy 손상채점':>16}{'greedy 복원채점':>16}")
    for alias, (fn, labs) in SETS.items():
        rows = json.loads((G / f"{fn}_greedy.json").read_text(encoding="utf-8"))
        gfx = G / f"{fn}_greedy_fixed.json"
        lab = {}
        for rel in labs:
            for r in json.loads((LB / rel).read_text(encoding="utf-8")):
                lab[r["pidx"]] = r
        ba = np.mean([lab[r["pidx"]]["base_acc"] for r in rows if r["pidx"] in lab])
        cb = np.mean([r["correct"] for r in rows])
        cc = (np.mean([r["correct"] for r in json.loads(gfx.read_text(encoding="utf-8"))])
              if gfx.is_file() else cb)
        out(f"   {alias:9s}{ba:>20.3f}{cb:>16.3f}{cc:>16.3f}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
