"""vLLM 의 DeepSeek 디토큰화 손상을 복원하고 **그 영향을 정량화**한다. GPU 0.

## 무엇이 깨졌나 (실측)
RunPod vLLM(0.31.0) 이 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` 의 출력을 디코드할 때 **공백을 `Ġ`,
줄바꿈을 `Ċ` 로 남긴다** — CLAUDE.md 9/23 에 적어 둔 그 함정이다(옛 `LlamaTokenizer` 경로).
Skywork·Olmo 는 **0건**, DeepSeek 는 **41/41 전행**.

그 결과 `extract_answer` 가 `\boxed{ 90}` 을 **`Ġ90`** 으로 뽑고 `same_answer("Ġ90", 90)` 이 False 가
된다 → **정답이 오답으로 집계된다.** greedy 41행 실측: 정답 판정 **28 → 40**(12행이 뒤집힘).

## 복원 방법과 그것이 옳은 이유
손상 텍스트를 **같은(옛) 토크나이저로 다시 인코딩하면 vLLM 이 실제로 낸 토큰 id 가 복원된다** —
검증: 재인코딩 길이 / vLLM `n_tok` 비가 **1.000**(41행, 최소 0.999). 그 id 를 **올바른
`PreTrainedTokenizerFast` 로 디코드**하면 원래 텍스트가 나온다(복원본에 `Ġ`/`Ċ` 0건).
즉 복원은 추정이 아니라 **왕복 변환**이다. 단순 문자 치환(`Ġ`→공백)을 쓰지 않는 이유는 다른 마커
(탭·특수 바이트)를 놓칠 수 있기 때문이다.

## 라벨에 대한 함의 (돈이 드는 쪽)
라벨 생성(`genlab_v2`)도 **같은 vLLM 경로**로 돌았고 텍스트를 저장하지 않았다 → DeepSeek 의
`n_hit`·`base_acc`·`var_acc`·`max_drop`·`label` 은 **같은 방식으로 손상돼 있고 사후 복원이 불가능하다**.
이 스크립트는 그 크기를 greedy 로 추정해 보고만 한다(재생성 결정은 [author] 몫).

사용: python analysis/fix_deepseek_detok.py --write
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
OUT = ROOT / "experiments" / "results" / "deepseek_detok_report.txt"
lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", default=str(ROOT / "experiments/results/runpod/genprobe/"
                                           "oi8q4v8ia3ojq0/out/DeepSeek-R1-0528-Qwen3-8B_greedy.json"))
    ap.add_argument("--write", action="store_true", help="복원본을 <원본>_fixed.json 으로 쓴다")
    a = ap.parse_args()
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    tf = PreTrainedTokenizerFast.from_pretrained(MID, local_files_only=True)
    ta = AutoTokenizer.from_pretrained(MID, local_files_only=True)

    ns = {"__name__": "g"}
    for f in ("experiments/variants_official.py", "experiments/variants_nonofficial.py",
              "experiments/kaggle/solve_common.py", "experiments/kaggle/genlab_v2.py"):
        exec(compile((ROOT / f).read_text(encoding="utf-8"), f, "exec"), ns)
    ex, same = ns["extract_answer"], ns["same_answer"]

    rows = json.loads(Path(a.greedy).read_text(encoding="utf-8"))
    out("vLLM DeepSeek 디토큰화 손상 — 복원과 영향 (GPU 0)")
    out()
    fixed, nb, cb, cc, flips, rat = [], 0, 0, 0, [], []
    for r in rows:
        bad = r["greedy_text"]
        nb += ("Ġ" in bad or "Ċ" in bad)
        ids = ta(bad, add_special_tokens=False)["input_ids"]
        rat.append(len(ids) / max(1, r["n_tok"]))
        clean = tf.decode(ids)
        ab, ac = ex(bad), ex(clean)
        kb = bool(r["finished"] and same(ab, r["gold"]))
        kc = bool(r["finished"] and same(ac, r["gold"]))
        cb += kb
        cc += kc
        if kb != kc:
            flips.append((r["pidx"], r["gold"], ab, ac))
        fixed.append({**r, "greedy_text": clean, "answer": None if ac is None else str(ac),
                      "correct": kc, "answer_corrupt": None if ab is None else str(ab),
                      "correct_corrupt": kb, "detok_fixed": True})
    n = len(rows)
    out(f"손상 행 **{nb}/{n}** · 재인코딩 길이 / vLLM n_tok 비 "
        f"최소 {min(rat):.3f} 최대 {max(rat):.3f} (1.000 이면 왕복이 정확하다)")
    out(f"복원본에 Ġ/Ċ 남은 행 {sum(1 for x in fixed if 'Ġ' in x['greedy_text'] or 'Ċ' in x['greedy_text'])}")
    out(f"정답 판정: 손상본 **{cb}/{n}** → 복원본 **{cc}/{n}** · 뒤집힌 행 **{len(flips)}**")
    out()
    out(f"{'pidx':>6}{'gold':>8}{'손상 추출':>12}{'복원 추출':>12}")
    for p, g, ab, ac in flips:
        out(f"{p:>6}{g:>8}{str(ab):>12}{str(ac):>12}")
    out()
    out("## 라벨에 대한 함의")
    out(f"   greedy 에서 **참 정답률이 {cb / n:.3f} → {cc / n:.3f}** 로 올라간다. 라벨 생성도 같은 경로였고")
    out("   텍스트를 저장하지 않았으므로 DeepSeek 의 base_acc·var_acc·max_drop·label 은 **복원 불가**다.")
    out("   같은 묶음의 기록된 평균 base_acc 0.701 은 Skywork 0.942 · Olmo 0.859 보다 낮았는데,")
    out("   greedy 복원값이 0.976 이므로 그 낮음은 모델 능력이 아니라 **이 버그로 설명된다**.")
    if a.write:
        p = Path(a.greedy).with_name(Path(a.greedy).stem + "_fixed.json")
        p.write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
        out(f"\n복원본 저장 {p}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
