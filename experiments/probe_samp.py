"""#SAMP-FEAT — n=4 샘플 '풀이 도중 벡터' 특징 추출. 로컬 4bit · prefill forward, 생성 없음.

사전등록: CLAUDE.md `## #SAMP-FEAT` (커밋 `c676d42`).

## `probe_gen.py` 와 **같은 경로**다 (바꾼 것은 샘플이 4개라는 것뿐)
프롬프트 조립·토크나이즈 순서·층 후보·풀링이 전부 같다:
  `ids = tok(chat_prompt).ids + tok(sample_text, add_special_tokens=False).ids[:2048]`
  → forward 1회 → `hidden_states[l][0, s:s+len(sid)].mean(0)`
**같아야 하는 이유**: 주비교가 '같은 행에서 greedy 2048 vs 샘플평균 2048' 이므로 특징 경로가
한 글자라도 다르면 그 차이가 '샘플 평균의 효과' 로 위장한다. 정밀도도 **4bit nf4** 로 맞춘다
(greedy 특징이 4bit 다 — #PROBE-AUDIT (3b) 가 측정한 이동은 코사인 0.997 의 결정적 편차다).

## 저장하는 것
`h{l}`      = **4샘플 벡터의 평균** (주특징, 사전등록)
`v{l}`      = 4샘플 벡터의 **성분별 분산** (부가 — 흔들림 단서 후보)
`vmean{l}`  = 그 분산의 평균 (스칼라 1차원, 부가)
`n_ok`      = 실제로 쓴 샘플 수 · `samp_tok` = 샘플별 풀이 토큰 수
라벨 메타(`y`·`y_fin`·`y_major`·`gen_k`·`pidx`·`words`)와 기준선 열(`base_acc`·`correct`·`n_tok`)은
**greedy 특징 npz 에서 그대로 복사**한다 — 행 순서가 어긋나면 비교가 깨지므로 거기서 읽는다.

사용:
  .venv/Scripts/python.exe experiments/probe_samp.py --model allenai/Olmo-3-7B-Think \\
      --samp experiments/results/runpod/genprobe/<pod>/out/<alias>_samp.json
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
WIN = 2048
BPE = {"Ġ": " ", "Ċ": "\n", "ĉ": "\t"}


def debpe(t):
    for k, v in BPE.items():
        t = t.replace(k, v)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samp", required=True, help="genprobe 샘플 모드 산출 <alias>_samp.json")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--ref", default=None, help="행 순서·라벨을 가져올 greedy 특징 npz (기본: <alias>_gen2048[_supp].npz)")
    ap.add_argument("--suffix", default="_samp")
    a = ap.parse_args()

    alias = a.model.split("/")[-1].replace(".", "_")
    ref = Path(a.ref) if a.ref else None
    if ref is None:
        for cand in (OUT / f"{alias}_gen{WIN}_supp.npz", OUT / f"{alias}_gen{WIN}.npz"):
            if cand.is_file():
                ref = cand
                break
    r = np.load(ref, allow_pickle=True)
    pidx = r["pidx"].astype(int)
    probs = [str(x) for x in r["problems"]] if "problems" in r else None
    rows = {int(x["pidx"]): x for x in json.loads(Path(a.samp).read_text(encoding="utf-8"))}
    print(f"[samp] 기준 npz {ref.name} · {len(pidx)}행 · 샘플 파일 {len(rows)}행", flush=True)
    miss = [int(q) for q in pidx if int(q) not in rows]
    if miss:
        raise SystemExit(f"샘플이 없는 pidx {miss}")
    if probs is None:
        probs = [rows[int(q)]["problem"] for q in pidx]

    tok, model = load(a.model, a.quant)
    L = model.config.num_hidden_layers
    layers = sorted({max(1, L // 4), L // 2, (3 * L) // 4, L})
    print(f"[samp] 층 {L} · 추출 인덱스 {layers} (probe_gen.py 와 같다)", flush=True)
    import torch

    mean = {l: [] for l in layers}
    var = {l: [] for l in layers}
    n_ok, samp_tok, t0 = [], [], time.time()
    for n, (q, p) in enumerate(zip(pidx, probs), 1):
        row = rows[int(q)]
        pid = tok(tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{p}"}],
                                          tokenize=False, add_generation_prompt=True),
                  add_special_tokens=False)["input_ids"]
        vecs = {l: [] for l in layers}
        ntk = []
        for s_ in row["samples"]:
            sid = tok(debpe(s_["text"]), add_special_tokens=False)["input_ids"][:WIN]
            if not sid:
                continue
            ids = torch.tensor([pid + sid], device=model.device)
            with torch.no_grad():
                o = model(input_ids=ids, output_hidden_states=True)
            s = len(pid)
            for l in layers:
                vecs[l].append(o.hidden_states[l][0, s:s + len(sid), :].float().mean(0).cpu().numpy())
            ntk.append(len(sid))
            del o
            torch.cuda.empty_cache()
        if not ntk:
            raise SystemExit(f"pidx {q} 의 샘플이 전부 비었다")
        for l in layers:
            M = np.stack(vecs[l])
            mean[l].append(M.mean(0))
            var[l].append(M.var(0))
        n_ok.append(len(ntk))
        samp_tok.append(ntk + [0] * (4 - len(ntk)))
        if n % 5 == 0 or n == len(pidx):
            print(f"  [samp] {n}/{len(pidx)} · {time.time() - t0:.0f}s", flush=True)

    keep = ["y", "y_fin", "y_major", "gen_k", "gen_cap", "pidx", "words",
            "base_acc", "correct", "n_tok", "finished", "problems"]
    meta = {k: r[k] for k in keep if k in r}
    meta["layers"] = np.array(layers)
    meta["n_ok"] = np.array(n_ok)
    meta["samp_tok"] = np.array(samp_tok)
    # `probe_gen_eval.py` 가 보고에 쓰는 키다 — 없으면 평가가 KeyError 로 죽는다.
    # 샘플마다 길이가 다르므로 **비영 샘플의 평균 풀이 토큰 수**로 정의한다.
    meta["n_sol_used"] = np.array([float(np.mean([t for t in ts if t])) for ts in samp_tok])
    pay = {}
    for l in layers:
        pay[f"h{l}"] = np.stack(mean[l])
        pay[f"v{l}"] = np.stack(var[l])
        pay[f"vmean{l}"] = np.stack(var[l]).mean(1)
    p = OUT / f"{alias}_gen{WIN}{a.suffix}.npz"
    np.savez_compressed(p, **meta, **pay)
    print(f"[samp] 저장 {p.name} · 샘플 수 {min(n_ok)}~{max(n_ok)} · "
          f"풀이 토큰 중앙 {int(np.median([t for ts in samp_tok for t in ts if t]))}", flush=True)


if __name__ == "__main__":
    main()
