"""#GEN-PROBE — '풀이 도중 벡터' 특징 추출. 로컬 4bit · **prefill forward 1회, 생성 없음**.

#77 은 **풀기 전**(문제 텍스트) 상태를 봤고 신호 미입증이었다. 라벨은 '푸는 동안 흔들리나' 이므로
이 스크립트는 **그 모델 자신의 greedy 풀이를 프롬프트 뒤에 붙여** 한 번 forward 하고 **풀이 토큰**
구간을 평균 풀링한다. 세 창(256 · 512 · 2048)을 **한 번의 forward 에서** 뜬다 — 창마다 다시 돌면
같은 계산을 세 번 하게 되고, 창 간 비교에 계산 차이가 섞인다.

## 토큰 경계를 이어 붙이지 않고 **따로 토크나이즈해 붙인다**
`tok(prompt + text)` 로 한 번에 세면 경계 토큰이 합쳐져 풀이 시작 위치가 밀릴 수 있다.
생성은 실제로 '프롬프트 토큰 → 생성 토큰' 순이었으므로 그 구조를 그대로 재현한다:
  ids = tok(prompt).ids + tok(greedy_text).ids   → 풀이 구간은 정확히 뒤쪽 len(sol) 개.

## 라벨 메타는 #77 특징 파일과 **같은 규약**으로 싣는다
`y`·`y_fin`·`y_major`(-1 = 그 규칙으로 라벨 없음)·`gen_k`·`gen_cap`·`pidx`·`words`.
추가로 기준선용 세 열을 같이 싣는다: `base_acc`(k=8) · `correct`(greedy 풀이 정답여부) ·
`n_tok`(greedy 풀이 전체 토큰 수). 기준선을 특징 파일 밖에서 만들면 행 순서가 어긋날 수 있다.

사용:
  .venv/Scripts/python.exe experiments/probe_gen.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
     --greedy experiments/results/genprobe/DeepSeek-R1-0528-Qwen3-8B_greedy.json \
     --labels <라벨 json...>  --sets ds_off40
출력: experiments/results/probe/<별칭>_gen{256,512,2048}<접미사>.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from h21_uncertainty import SOLVE_PROMPT, load                      # noqa: E402

OUT = ROOT / "experiments" / "results" / "probe"
WINS = (256, 512, 2048)


def label_map(paths):
    """pidx → 라벨 메타. 모델별로 pidx 가 고유함을 **확인한다**(슬라이스가 여럿이라 겹치면 조용히 섞인다)."""
    m = {}
    for f in paths:
        for r in json.loads(Path(f).read_text(encoding="utf-8")):
            if not isinstance(r.get("label"), bool):
                continue
            if r["pidx"] in m:
                raise SystemExit(f"pidx {r['pidx']} 가 두 라벨 파일에 있다 — 묶음을 확인하라")
            m[r["pidx"]] = r
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--greedy", required=True, help="genprobe 산출 <별칭>_greedy.json")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="이 묶음만 (예: ds_off40). 생략하면 전부")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    alias = a.model.split("/")[-1].replace(".", "_")

    rows = json.loads(Path(a.greedy).read_text(encoding="utf-8"))
    if a.sets:
        rows = [r for r in rows if r.get("set") in set(a.sets)]
    lab = label_map(a.labels)
    rows = [r for r in rows if r["pidx"] in lab]
    if not rows:
        raise SystemExit("라벨과 짝지은 행이 0개다")
    print(f"[gen] {alias} · 행 {len(rows)} · 묶음 {sorted({r['set'] for r in rows})}", flush=True)

    import torch
    tok, model = load(a.model, a.quant)
    L = model.config.num_hidden_layers
    layers = sorted({max(1, L // 4), L // 2, (3 * L) // 4, L})
    print(f"[gen] 층 {L} · 추출 인덱스 {layers}", flush=True)

    feats = {w: {l: [] for l in layers} for w in WINS}
    nsol, t0 = [], time.time()
    for n, r in enumerate(rows, 1):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{r['problem']}"}],
            tokenize=False, add_generation_prompt=True)
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        sid = tok(r["greedy_text"], add_special_tokens=False)["input_ids"][:max(WINS)]
        if not sid:
            raise SystemExit(f"pidx {r['pidx']} 의 greedy 풀이가 비었다")
        nsol.append(len(sid))
        ids = torch.tensor([pid + sid], device=model.device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        s = len(pid)                                    # 풀이 토큰 시작 위치 (프롬프트는 제외)
        for w in WINS:
            e = s + min(w, len(sid))
            for l in layers:
                feats[w][l].append(out.hidden_states[l][0, s:e, :].float().mean(0).cpu().numpy())
        del out
        torch.cuda.empty_cache()
        if n % 5 == 0 or n == len(rows):
            print(f"  [gen] {n}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    print(f"[gen] 풀이 토큰(창 상한 {max(WINS)} 적용 후) 최소 {min(nsol)} 중앙 "
          f"{sorted(nsol)[len(nsol) // 2]} 최대 {max(nsol)}", flush=True)

    def col(key, cast, miss=-1):
        return np.array([miss if lab[r["pidx"]].get(key) is None else cast(lab[r["pidx"]][key])
                         for r in rows])

    meta = dict(layers=np.array(layers),
                y=np.array([int(lab[r["pidx"]]["label"]) for r in rows]),
                y_fin=col("label_fin", int), y_major=col("label_major", int),
                gen_k=col("gen_k", int, 0), gen_cap=col("gen_cap", int, 0),
                pidx=np.array([r["pidx"] for r in rows]),
                words=np.array([len(r["problem"].split()) for r in rows]),
                # ---- 기준선 셋 (지휘 지정). 행 순서가 특징과 같아야 해서 같은 파일에 싣는다.
                base_acc=np.array([float(lab[r["pidx"]]["base_acc"]) for r in rows]),
                correct=np.array([int(bool(r["correct"])) for r in rows]),
                n_tok=np.array([int(r["n_tok"]) for r in rows]),
                finished=np.array([int(bool(r["finished"])) for r in rows]),
                n_sol_used=np.array(nsol),
                problems=np.array([r["problem"] for r in rows], dtype=object))
    for w in WINS:
        p = OUT / f"{alias}_gen{w}{a.suffix}.npz"
        np.savez_compressed(p, **meta, **{f"h{l}": np.stack(feats[w][l]) for l in layers})
        print(f"[gen] 저장 {p}", flush=True)


if __name__ == "__main__":
    main()
