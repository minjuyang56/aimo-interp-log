"""#CMP4 (D) — 풀이 앞 2048 토큰을 **16구간**으로 나눈 구간별 평균 벡터. 로컬 4bit · forward 1회.

사전등록: CLAUDE.md `## #CMP4` (커밋 `cc9cd43`).

## `probe_gen.py` 와 **같은 경로**다 — 바꾼 것은 '풀링을 16구간으로 쪼갠다' 하나뿐
프롬프트 조립·토크나이즈 순서(따로 토크나이즈해 이어 붙임)·층 후보·정밀도(4bit nf4)·
`debpe` 미적용까지 전부 같다. **같아야 하는 이유**: #CMP4 의 주비교가 '같은 특징·같은 폴드에서
분류기만 바꾼다' 이므로 특징 경로가 한 글자라도 다르면 그 차이가 'HDC 의 효과' 로 위장한다.
(A·B·C 가 쓰는 `h{l}` 은 같은 forward 의 **전 구간 평균**이고, 여기 `seg{l}` 의 구간 평균을
토큰 수로 가중 평균하면 그것과 같아야 한다 — 그 동일성을 `--check` 로 계측한다.)

## 구간 분할 (사전등록에서 고정)
풀이 토큰 `n = min(2048, len(sol))` 개를 **16개 연속 구간**으로 나눈다(`np.array_split`).
구간 수는 **16 고정**이고 `n` 이 2048 미만이면 구간이 짧아진다 — 행마다 `n_sol_used` 를 적는다.

## 저장
`seg{l}` = (행, 16, d) float32 · 라벨 메타는 **기준 npz 에서 그대로 복사**한다
(행 순서가 어긋나면 비교가 깨지므로 pidx 순서를 기준 npz 에 맞춘다).

사용:
  .venv/Scripts/python.exe experiments/probe_seg.py --model allenai/Olmo-3-7B-Think \\
      --greedy experiments/results/runpod/genprobe/oi8q4v8ia3ojq0/out/Olmo-3-7B-Think_greedy.json
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
NSEG = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--greedy", required=True, help="genprobe 산출 <별칭>_greedy*.json")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--ref", default=None, help="행 순서·라벨을 가져올 npz (기본 <별칭>_gen2048[_supp].npz)")
    ap.add_argument("--suffix", default="_seg")
    ap.add_argument("--check", action="store_true",
                    help="구간 평균의 토큰 가중 평균이 기준 npz 의 전구간 평균과 같은지 계측")
    a = ap.parse_args()

    alias = a.model.split("/")[-1].replace(".", "_")
    ref = Path(a.ref) if a.ref else None
    if ref is None:
        for cand in (OUT / f"{alias}_gen{WIN}_supp.npz", OUT / f"{alias}_gen{WIN}.npz"):
            if cand.is_file():
                ref = cand
                break
    r = np.load(ref, allow_pickle=True)
    pidx = [int(x) for x in r["pidx"]]
    rows = {int(x["pidx"]): x for x in json.loads(Path(a.greedy).read_text(encoding="utf-8"))}
    miss = [q for q in pidx if q not in rows]
    if miss:
        raise SystemExit(f"greedy 텍스트가 없는 pidx {miss}")
    print(f"[seg] 기준 npz {ref.name} · {len(pidx)}행 · greedy 파일 {len(rows)}행", flush=True)

    import torch
    tok, model = load(a.model, a.quant)
    L = model.config.num_hidden_layers
    layers = sorted({max(1, L // 4), L // 2, (3 * L) // 4, L})
    print(f"[seg] 층 {L} · 추출 인덱스 {layers} (probe_gen.py 와 같다)", flush=True)

    seg = {l: [] for l in layers}
    whole = {l: [] for l in layers}      # --check 용 전구간 평균 (같은 forward)
    nsol, t0 = [], time.time()
    for n, q in enumerate(pidx, 1):
        row = rows[q]
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{row['problem']}"}],
            tokenize=False, add_generation_prompt=True)
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        sid = tok(row["greedy_text"], add_special_tokens=False)["input_ids"][:WIN]
        if not sid:
            raise SystemExit(f"pidx {q} 의 greedy 풀이가 비었다")
        nsol.append(len(sid))
        ids = torch.tensor([pid + sid], device=model.device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        s = len(pid)
        bounds = np.array_split(np.arange(len(sid)), NSEG)
        for l in layers:
            H = out.hidden_states[l][0, s:s + len(sid), :].float().cpu().numpy()
            seg[l].append(np.stack([H[b].mean(0) for b in bounds]).astype(np.float32))
            whole[l].append(H.mean(0).astype(np.float32))
        del out
        torch.cuda.empty_cache()
        if n % 5 == 0 or n == len(pidx):
            print(f"  [seg] {n}/{len(pidx)} · {time.time() - t0:.0f}s", flush=True)

    keep = ["y", "y_fin", "y_major", "gen_k", "gen_cap", "pidx", "words",
            "base_acc", "correct", "n_tok", "finished", "problems"]
    meta = {k: r[k] for k in keep if k in r}
    meta["layers"] = np.array(layers)
    meta["n_sol_used"] = np.array(nsol)
    meta["nseg"] = np.array(NSEG)
    pay = {f"seg{l}": np.stack(seg[l]) for l in layers}
    p = OUT / f"{alias}_gen{WIN}{a.suffix}.npz"
    np.savez_compressed(p, **meta, **pay)
    print(f"[seg] 저장 {p.name} · 모양 {pay[f'seg{layers[0]}'].shape} · "
          f"풀이토큰 최소 {min(nsol)} 중앙 {sorted(nsol)[len(nsol) // 2]} 최대 {max(nsol)}", flush=True)

    if a.check:
        #  구간 평균을 토큰 수로 가중 평균하면 전구간 평균과 같아야 한다.
        #  그리고 그 전구간 평균이 기준 npz 의 h{l} 과 같아야 한다 = 특징 경로가 동일하다는 증거.
        for l in layers:
            W = np.stack([np.array([len(b) for b in np.array_split(np.arange(m), NSEG)], float)
                          for m in nsol])
            W = W / W.sum(1, keepdims=True)
            rec = np.einsum("ns,nsd->nd", W, np.stack(seg[l]))
            own = np.stack(whole[l])
            d1 = float(np.abs(rec - own).max())
            if f"h{l}" in r:
                H = np.asarray(r[f"h{l}"], np.float32)
                cos = float(np.mean([np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_))
                                     for a_, b_ in zip(own, H)]))
                d2 = float(np.abs(own - H).max())
                print(f"  [check] 층 {l} · 구간가중평균↔전구간 최대절대차 {d1:.3e} · "
                       f"전구간↔기준npz cos {cos:.6f} 최대절대차 {d2:.3e}", flush=True)
            else:
                print(f"  [check] 층 {l} · 구간가중평균↔전구간 최대절대차 {d1:.3e} "
                       f"(기준 npz 에 h{l} 없음)", flush=True)


if __name__ == "__main__":
    main()
