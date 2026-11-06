"""로컬 CPU에서 대리 모델의 생성 속도를 잰다 — tok/s 하나를 얻는 게 목적.

이 숫자로 "CPU에서 무엇이 가능한가"가 갈린다:
  - forward pass 1회(프로빙)      : 프롬프트 길이 / tok/s(prefill)
  - 풀이 생성(불확실성·라벨 재현) : 생각 토큰 수천 개 / tok/s(decode)

R1 계열은 생각 토큰을 수천 개 뽑으므로 decode 속도가 결정적이다.

사용:  uv run python analysis/cpu_throughput.py [--new-tokens 64]
"""

import argparse
import time

MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
PROMPT = "What is the value of $255^2 - 245^2$?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(torch.get_num_threads())
    print(f"threads={torch.get_num_threads()}  device=cpu  dtype=bfloat16", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).eval()
    load_s = time.time() - t0
    print(f"load      {load_s:6.1f}s", flush=True)

    messages = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    n_prompt = inputs["input_ids"].shape[1]

    # prefill만: forward 1회 (프로빙이 쓰는 경로)
    t0 = time.time()
    with torch.no_grad():
        model(**inputs)
    prefill_s = time.time() - t0
    print(f"prefill   {prefill_s:6.1f}s  for {n_prompt} prompt tokens"
          f"  -> {n_prompt / prefill_s:6.1f} tok/s", flush=True)

    # decode: greedy로 new_tokens개
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    decode_s = time.time() - t0
    n_new = out.shape[1] - n_prompt
    rate = n_new / decode_s
    print(f"decode    {decode_s:6.1f}s  for {n_new} new tokens  -> {rate:6.2f} tok/s", flush=True)

    print()
    print("환산 (decode 기준):")
    for n in (512, 2048, 4096, 8192):
        print(f"  {n:5d} 토큰 풀이 1회  ≈ {n / rate / 60:6.1f} 분")
    print(f"  프로빙 1문제(forward 1회) ≈ {prefill_s:.1f} 초")


if __name__ == "__main__":
    main()
