"""디코딩 속도 벤치 — 워커에서 83 tok/s 밖에 안 나온 원인을 좁힌다. (지휘 #57b)

같은 조건(Skywork · **시퀀스 8개** · max_new 8192 · greedy)에서 네 가지 생성 경로의 합산 tok/s 를 잰다.
  A baseline      model.generate, 좌측 패딩, 배치 8
  B static cache  model.generate(cache_implementation="static") — DynamicCache 의 매 스텝 cat 복사를 피한다
  C generate_batch 연속 배칭(paged) API — 있으면 쓰고 없으면 사유를 기록
  D per-problem   배치 1로 8번 (패딩 없음) — 패딩 마스크가 원인인지 가린다
워커가 아니라 Kaggle T4 라 절대값은 다르지만 **경로 간 상대 속도**는 옮겨 갈 것으로 본다.
짧은 프롬프트(올림피아드 문제)라 프리필은 무시할 수준이고 디코딩 비용이 전부다.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

V2_URL = "https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/data/train-00000-of-00001.parquet"
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")


def problems(n):
    import pandas as pd
    c = Path("train-main-v2.parquet")
    if not c.is_file():
        c.write_bytes(urllib.request.urlopen(urllib.request.Request(V2_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read())
    df = pd.read_parquet(c)
    return [p.rstrip() for p, _ in df.groupby("problem", sort=False)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=globals().get("RUN_MODEL", "Skywork/Skywork-OR1-Math-7B"))
    ap.add_argument("--seqs", type=int, default=globals().get("RUN_SEQS", 8))
    ap.add_argument("--max-new", type=int, default=globals().get("RUN_MAX_NEW", 8192))
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    t0 = time.time()
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(a.model)
    except Exception:
        tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="auto",
                                                 low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    import transformers
    print(f"[bench] 로드 {time.time() - t0:.0f}s · transformers {transformers.__version__} · "
          f"torch {torch.__version__} · GPU {torch.cuda.device_count()}", flush=True)

    texts = problems(a.seqs)
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    res = {}

    def count_new(out, plen):
        new = out[:, plen:]
        return int((new != tok.pad_token_id).sum())

    def run(tag, fn):
        torch.cuda.empty_cache()
        t = time.time()
        try:
            ntok = fn()
            el = time.time() - t
            res[tag] = {"sec": round(el, 1), "tokens": int(ntok), "tok_s": round(ntok / max(1e-6, el), 1),
                        "per_seq": round(ntok / max(1e-6, el) / a.seqs, 1)}
            print(f"[bench] {tag}: {el:.0f}s · {ntok} 토큰 · **{ntok / max(1e-6, el):.1f} tok/s 합산** "
                  f"({ntok / max(1e-6, el) / a.seqs:.1f}/시퀀스)", flush=True)
        except Exception as e:
            res[tag] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"[bench] {tag} 실패: {type(e).__name__} {str(e)[:200]}", flush=True)

    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    plen = enc["input_ids"].shape[1]
    print(f"[bench] 시퀀스 {a.seqs} · 프롬프트 토큰 {plen} · max_new {a.max_new}", flush=True)

    def a_baseline():
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        return count_new(out, plen)

    def b_static():
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                                 cache_implementation="static")
        return count_new(out, plen)

    def c_contbatch():
        from transformers.generation.configuration_utils import GenerationConfig
        gc = GenerationConfig(max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        if not hasattr(model, "generate_batch"):
            raise RuntimeError("generate_batch 없음 (transformers 버전)")
        ids = [tok(p)["input_ids"] for p in prompts]
        with torch.no_grad():
            out = model.generate_batch(inputs=ids, generation_config=gc)
        n = 0
        for o in (out.values() if isinstance(out, dict) else out):
            seq = getattr(o, "generated_tokens", o)
            n += len(seq)
        return n

    def d_perproblem():
        n = 0
        for p in prompts:
            e1 = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**e1, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            n += o.shape[1] - e1["input_ids"].shape[1]
        return n

    run("A_baseline_pad", a_baseline)
    run("B_static_cache", b_static)
    run("C_generate_batch", c_contbatch)
    run("D_per_problem", d_perproblem)
    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    Path("/kaggle/working/bench_decode.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("[bench] 저장 완료", flush=True)
    for k, v in res.items():
        print(f"  {k:18s} {v}", flush=True)


if __name__ == "__main__":
    main()
