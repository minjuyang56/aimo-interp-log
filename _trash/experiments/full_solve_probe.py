"""모델 하나에 train-v2 문제 하나를 끝까지 풀게 해서 '라벨 1개의 원가'를 잰다.

라벨 하나 = 원본 k회 + 변형 m종 × k회 풀이. 그 단위 비용이 "완주 1회"다.
budget forcing 없이 eos까지 두고 토큰 수·초·답 파싱 여부를 기록한다.

사용:  .venv/Scripts/python.exe experiments/full_solve_probe.py --model Qwen/Qwen3.5-4B --max-new-tokens 8192 --n 2
"""
import argparse, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from extract_hidden import load_tokenizer
from extract_train_v2 import SOLVE_PROMPT, load_model, problem_table
from mini_sim import extract_answer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--max-new-tokens", type=int, default=8192)
ap.add_argument("--n", type=int, default=2); ap.add_argument("--quant", default="4bit")
a = ap.parse_args()
import torch
tbl = problem_table(); tok = load_tokenizer(a.model); model = load_model(a.model, a.quant)
print(f"model={a.model} quant={a.quant} max_new_tokens={a.max_new_tokens}", flush=True)
for r in tbl.head(a.n).itertuples():
    text = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{r.problem}"}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device); t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    new = out[0, ids["input_ids"].shape[1]:]; sec = time.time() - t0
    body = tok.decode(new, skip_special_tokens=True); eos = bool((new == tok.eos_token_id).any())
    print(f"[{r.pidx}] tokens={new.numel():5d}  {sec:6.0f}s  {new.numel()/sec:.1f} tok/s  eos={eos}  think_closed={'</think>' in body}  answer={extract_answer(body)!r}  label_y={r.y}", flush=True)
    print("   tail:", body[-200:].replace("\n", " ⏎ "), flush=True)
