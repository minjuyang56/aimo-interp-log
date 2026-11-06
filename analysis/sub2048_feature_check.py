"""#SUB-GEN2048 특징 동일성 검증 — 제출물 `feature()` 가 학습 npz 와 같은 값인가. (지휘 #SUB-GEN2048)

**이게 이 제출물에서 가장 중요한 검사다.** 특징이 어긋나면 train 가중치가 무의미해지고,
점수는 조용히 잡음이 된다(hprobe 에서도 같은 검사를 통과 조건으로 걸었다 — cos 1.000000).

## 생성 없이 검증할 수 있는 이유
학습 특징은 저장된 greedy 풀이 텍스트를 다시 토크나이즈해 prefill 한 것이다
(`experiments/probe_gen.py`). 그 텍스트가 `experiments/results/runpod/genprobe/.../＊_greedy.json`
에 남아 있으므로, **같은 텍스트를 제출물 `feature()` 에 넣으면 생성 단계를 건너뛰고
특징 경로만 떼어 비교할 수 있다.** 로컬 4bit 로 돌고 비용 0 이다.

## 그래서 이 검사가 **덮지 못하는 것** (보고에 그대로 적는다)
* **생성 경로**: 워커에서 `model.generate(do_sample=False, max_new_tokens=2048)` 가 학습 때
  vLLM cap 32768 greedy 의 앞 2048 토큰과 같은 텍스트를 내는지. greedy 는 결정적이지만
  **엔진이 다르다**(vLLM vs HF) — 커널 차이로 소수점 아래가 갈리면 긴 생성에서 갈라질 수 있다.
  이건 GPU 실측이 필요하다.
* **정밀도**: 학습은 로컬 **4bit nf4**, 제출물 기본은 **bf16**(워커 96 GB). #PROBE-AUDIT (3b) 측정으로
  층별 코사인 0.997~0.998 · 행쌍 거리 상관 0.881~0.997 · 폴드밖 점수 순위상관 0.969 이므로
  작지만 **체계적인** 이동이 있다. 여기서는 `SUB2048_QUANT=4bit` 로 학습과 같은 정밀도를 쓴다.

사용:
  .venv/Scripts/python.exe analysis/sub2048_feature_check.py --model Skywork/Skywork-OR1-Math-7B --n 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "experiments" / "results" / "probe"
GEN = ROOT / "experiments" / "results" / "runpod" / "genprobe"
ALIAS = {"Skywork/Skywork-OR1-Math-7B": "Skywork-OR1-Math-7B",
         "allenai/Olmo-3-7B-Think": "Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B"}


def greedy_rows(alias):
    """genprobe 산출 greedy 텍스트. DeepSeek 는 병합본(복원 41 + 보충 25)을 쓴다."""
    for p in (GEN / f"{alias}_greedy_merged.json",
              GEN / "oi8q4v8ia3ojq0" / "out" / f"{alias}_greedy_fixed.json",
              GEN / "oi8q4v8ia3ojq0" / "out" / f"{alias}_greedy.json"):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), p
    raise SystemExit(f"{alias} greedy 텍스트를 못 찾았다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Skywork/Skywork-OR1-Math-7B")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--quant", default="4bit", help="학습과 같은 정밀도여야 비교가 성립한다")
    ap.add_argument("--out", default=str(ROOT / "experiments/results/sub2048_feature_check.txt"))
    a = ap.parse_args()
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    os.environ["SUB2048_QUANT"] = a.quant
    sys.path.insert(0, str(ROOT / "solutions" / "sub2048"))
    import sub2048_lib as S

    alias = ALIAS[a.model]
    w = S.weights_for(a.model)
    if w is None:
        raise SystemExit(f"{a.model} 가중치가 없다")
    npz = np.load(PROBE / w["source"], allow_pickle=True)
    pidx = npz["pidx"].astype(int)
    probs = [str(x) for x in npz["problems"]]
    ref = npz[f"h{w['layer']}"].astype(np.float64)
    rows, src = greedy_rows(alias)
    sol = {int(r["pidx"]): r["greedy_text"] for r in rows}

    say(f"#SUB-GEN2048 특징 동일성 — {alias} · 층 {w['layer']} · 정밀도 {a.quant} · 생성 없음")
    say(f"학습 특징 {w['source']} ({len(pidx)}행) · greedy 텍스트 {src.name} ({len(rows)}행)")
    say("")
    tok, model = S.get_model(a.model)
    if tok is None:
        raise SystemExit("모델을 못 열었다")

    say(f"{'pidx':>5} {'풀이토큰':>8} {'코사인':>10} {'최대절대차':>12} {'상대L2':>8} "
        f"{'학습점수':>9} {'제출점수':>9}")
    cos, mx, ok = [], [], 0
    for q, p, h_ref in list(zip(pidx, probs, ref))[:a.n]:
        if int(q) not in sol:
            say(f"{q:>5} (greedy 텍스트 없음 — 건너뜀)")
            continue
        h = S.feature(tok, model, p, S.debpe(sol[int(q)]), int(w["layer"]))
        h = np.asarray(h, dtype=np.float64)
        c = float(h @ h_ref / (np.linalg.norm(h) * np.linalg.norm(h_ref)))
        m = float(np.abs(h - h_ref).max())
        rl2 = float(np.linalg.norm(h - h_ref) / np.linalg.norm(h_ref))
        ns = len(tok(S.debpe(sol[int(q)]), add_special_tokens=False)["input_ids"][:2048])
        say(f"{q:>5} {ns:>8} {c:>10.6f} {m:>12.3e} {rl2:>8.2e} "
            f"{S.probe_score(w, h_ref):>+9.4f} {S.probe_score(w, h):>+9.4f}")
        cos.append(c); mx.append(m); ok += 1
    say("")
    if not ok:
        say("!! 비교한 행이 없다")
        Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
        return 1
    say(f"행 {ok}개 · 코사인 최소 **{min(cos):.6f}** · 최대절대차 최대 **{max(mx):.3e}**")
    passed = min(cos) > 0.99999 and max(mx) < 1e-2
    say(f"판정: {'통과 — 특징 경로가 학습과 같다' if passed else '!! 실패 — 특징이 어긋난다'}")
    say("")
    say("덮지 못한 것: (a) 워커 HF generate 가 학습 때 vLLM 생성의 앞 2048 토큰과 같은 텍스트를")
    say("내는지 (b) bf16 로 돌 때의 정밀도 이동 — 둘 다 GPU 실측이 필요하다(docstring 참조).")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
