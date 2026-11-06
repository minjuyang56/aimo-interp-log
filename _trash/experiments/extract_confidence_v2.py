"""train-main-v2의 36문제를 로컬 4모델에 풀게 하되, 처음 N토큰의 '자신감'만 기록한다.

정지 사진(forward 1회) 프로브가 33문제에서 신호를 못 냈다(9/26). 주최측 자체 베이스라인은
생성 기반 불확실성 쪽이 프로빙보다 11점 높았다(69 vs 58). 그쪽을 예산 안에서 시험한다:
답까지 안 가고 **생각의 첫 N토큰**만 생성하면서 매 토큰의 확률분포에서 특징을 뽑는다.

토큰당 기록: 고른 토큰의 logprob · 분포 엔트로피 · top1 확률 · top1-top2 차이
문제당 요약(14개, 업스트림 uncertainty-profiling과 같은 계열):
    mean/min/std logprob · ppl · mean entropy · mean top1 · mean margin ·
    frac(logprob < -2) · frac(top1 > 0.9) · 앞 1/4 vs 뒤 1/4 logprob 차이 · 엔트로피 기울기 · n_tokens
+ 토큰별 원시 수열도 저장 (경로 모양 실험용).

저장: results/trainv2_conf_<모델>_N<N>.npz    F (문제, 특징) · seq (문제, N, 4) · y · drop · problems

비용: 문제당 N토큰 생성. 4bit 노트북 GPU 10 tok/s면 N=512 × 36문제 × 4모델 ≈ 2시간.

사용:  .venv/Scripts/python.exe experiments/extract_confidence_v2.py --models <하나> --n-tokens 512
       (VRAM 회수 문제 때문에 모델당 별도 프로세스로 부를 것)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import LOCAL_MODELS, SOLVE_PROMPT, load_model, problem_table  # noqa: E402

FEATURES = ["mean_lp", "min_lp", "std_lp", "ppl", "mean_ent", "mean_top1", "mean_margin",
            "frac_low_lp", "frac_conf", "lp_late_minus_early", "ent_slope", "n_tokens"]


def summarize(seq):
    """seq: (T, 4) = [logprob, entropy, top1, margin]"""
    lp, ent, top1, mg = seq.T
    T = len(lp); q = max(1, T // 4)
    x = np.arange(T)
    slope = float(np.polyfit(x, ent, 1)[0]) if T > 2 else 0.0
    return np.array([
        lp.mean(), lp.min(), lp.std(), np.exp(-lp.mean()), ent.mean(), top1.mean(), mg.mean(),
        (lp < -2.0).mean(), (top1 > 0.9).mean(), lp[-q:].mean() - lp[:q].mean(), slope, float(T),
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=LOCAL_MODELS)
    ap.add_argument("--n-tokens", type=int, default=512)
    ap.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    a = ap.parse_args()

    import torch
    tbl = problem_table()
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)

    for m in a.models:
        out = out_dir / f"trainv2_conf_{alias(m)}_N{a.n_tokens}.npz"
        t0 = time.time()
        tok = load_tokenizer(m); model = load_model(m, a.quant)
        print(f"\n== {m}  load {time.time() - t0:.0f}s  N={a.n_tokens}", flush=True)
        F, S = [], []
        for r in tbl.itertuples():
            text = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{r.problem}"}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            t1 = time.time()
            with torch.no_grad():
                out_g = model.generate(**ids, max_new_tokens=a.n_tokens, do_sample=False,
                                       output_scores=True, return_dict_in_generate=True,
                                       pad_token_id=tok.eos_token_id)
            new = out_g.sequences[0, ids["input_ids"].shape[1]:]
            rows = []
            for step, logits in enumerate(out_g.scores):
                lp_all = torch.log_softmax(logits[0].float(), -1)
                p_all = lp_all.exp()
                chosen = new[step].item()
                top2 = torch.topk(p_all, 2).values
                rows.append([lp_all[chosen].item(), -(p_all * lp_all).sum().item(),
                             top2[0].item(), (top2[0] - top2[1]).item()])
            seq = np.array(rows, dtype=np.float32)
            F.append(summarize(seq))
            pad = np.full((a.n_tokens, 4), np.nan, np.float32); pad[:len(seq)] = seq; S.append(pad)
            print(f"  [{r.pidx + 1:2d}/{len(tbl)}] T={len(seq):4d} {time.time() - t1:6.1f}s  mean_lp={F[-1][0]:+.3f} ent={F[-1][4]:.3f} y={r.y:+d}", flush=True)
        np.savez_compressed(out, F=np.stack(F), seq=np.stack(S), features=np.array(FEATURES),
                            y=tbl.y.values.astype(np.int8), drop=tbl["drop"].values.astype(np.float32),
                            problems=tbl.problem.values, model=m, n_tokens=a.n_tokens)
        print(f"  저장 {out}  총 {time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
