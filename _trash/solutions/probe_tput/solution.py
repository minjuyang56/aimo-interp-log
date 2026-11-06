"""워커 처리량 프로브 (Small 트랙) — 점수가 아니라 **계측**이 목적. (지휘 #53-5)

왜: 테스트타임 재현(H17)이 3600초 합산 예산 안에 들어가는지는 **워커의 실제 생성 처리량**에 달렸는데 우리는 그 값을 모른다.
    로컬(8GB·4bit)과 Kaggle T4 수치는 워커(RTX PRO 6000 96GB·bf16)와 다르다.
무엇을 재나 (전부 stderr 로만 기록, stdout 은 채점기 몫)
  1. vLLM import 가능 여부 + 버전 (런타임 이미지에 없을 것이라는 정보가 있어 직접 확인한다)
  2. 모델 로드 시간 (Qwen3.5-4B · Skywork-7B)
  3. `model.generate` 배치 32 · max_new 2048 처리량 (tok/s 합산)
  4. 토큰 단위 증분 루프 처리량 (같은 배치, 비교용)
안전장치: 전역 시계 · 모델당 예산 · 어떤 경우에도 **반환은 전부 True** (always-true 와 같은 점수 0.50 예상).
"""

import os
import sys
import time

T0 = time.time()
TOTAL_BUDGET = 3600.0
MARGIN = 300.0
PER_MODEL_BUDGET = float(os.environ.get("PROBE_MODEL_BUDGET", 700))
MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B"]
N_PROB, BATCH, MAX_NEW = 8, 32, 2048
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")
_done = False


def log(msg):
    print(f"[tput {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def remaining():
    return TOTAL_BUDGET - MARGIN - (time.time() - T0)


def probe_vllm():
    try:
        import vllm
        log(f"vLLM import OK · version {getattr(vllm, '__version__', '?')}")
    except Exception as e:
        log(f"vLLM import 실패: {type(e).__name__} {str(e)[:120]}")
    for mod in ("torch", "transformers", "bitsandbytes", "flash_attn"):
        try:
            m = __import__(mod)
            log(f"  {mod} {getattr(m, '__version__', '?')}")
        except Exception as e:
            log(f"  {mod} 없음 ({type(e).__name__})")


def probe_model(model_id, problems):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    t0 = time.time()
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, local_files_only=True,
                                                 low_cpu_mem_usage=True, device_map="auto").eval()
    log(f"{model_id}: 로드 {time.time() - t0:.1f}s · device_map {getattr(model, 'hf_device_map', None)}")
    texts = (problems * ((BATCH // max(1, len(problems))) + 1))[:BATCH]
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    log(f"  배치 {BATCH} · 프롬프트 토큰 {enc['input_ids'].shape[1]} · max_new {MAX_NEW}")

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
    el = time.time() - t1
    new = out[:, enc["input_ids"].shape[1]:]
    ntok = int((new != tok.pad_token_id).sum())
    log(f"  generate: {el:.1f}s · 생성 {ntok} 토큰 · **{ntok / max(1e-6, el):.1f} tok/s 합산** "
        f"({ntok / max(1e-6, el) / BATCH:.1f}/시퀀스)")

    if remaining() > 120 and time.time() - t0 < PER_MODEL_BUDGET:
        t2 = time.time()
        steps = min(256, MAX_NEW)
        with torch.no_grad():
            o = model(**enc, use_cache=True)
            past, nxt = o.past_key_values, o.logits[:, -1].argmax(-1)
            am = enc["attention_mask"]
            for _ in range(steps):
                am = torch.cat([am, torch.ones(am.shape[0], 1, dtype=am.dtype, device=am.device)], 1)
                o = model(input_ids=nxt[:, None], attention_mask=am, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1].argmax(-1)
        el2 = time.time() - t2
        log(f"  증분 루프: {el2:.1f}s · {steps}스텝 × {BATCH} = {steps * BATCH} 토큰 · "
            f"**{steps * BATCH / max(1e-6, el2):.1f} tok/s 합산**")
    del model
    torch.cuda.empty_cache()


def run_probe(problems):
    global _done
    if _done:
        return
    _done = True
    log(f"프로브 시작 · 문제 {len(problems)}개 · 전역 예산 {TOTAL_BUDGET:.0f}s")
    probe_vllm()
    for mid in MODELS:
        if remaining() < 400:
            log(f"예산 부족으로 {mid} 생략"); continue
        try:
            probe_model(mid, problems[:N_PROB])
        except Exception as e:
            log(f"{mid} 실패: {type(e).__name__} {str(e)[:160]}")
    log("프로브 끝")


def are_robust(model_id: str, reasoning_effort: str = "default",
               problems: list[str] | None = None) -> list[bool]:
    if problems is None:
        problems = []
    if not isinstance(reasoning_effort, str):
        problems, reasoning_effort = list(reasoning_effort), "default"
    if not problems:
        return []
    try:
        run_probe(list(problems))
    except Exception as e:
        log(f"프로브 전체 실패: {type(e).__name__} {str(e)[:160]}")
    return [True] * len(problems)
