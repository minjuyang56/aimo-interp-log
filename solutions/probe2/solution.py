"""워커 프로브 2 (Small 트랙) — 스텝당 시간이 배치 크기에 정말 무관한지, 그리고 compile 이 도움이 되는지. (지휘 #60-4)

왜: h20c 의 시간 모델은 "벽시계 ≈ cap × ms/step, 배치·k 에는 거의 무관" 이라는 가정 위에 서 있다.
    워커 실측 두 점(배치 10 @16k ≈ 79 ms/step, 배치 32 @2k ≈ 54 ms/step)은 그 가정과 맞지만 문맥 길이가 섞여 있다.
    **같은 스텝 수(2048)에서 배치만 8 → 32 → 64 로 바꿔** ms/step 이 평평한지 본다.
    더불어 torch.compile + static cache 를 **컴파일 시간까지 포함해** 한 번 재서, 한 호출 안에서 이득이 남는지 본다.
전부 stderr 로만 기록하고 반환은 항상 True (점수는 상수 예측과 같을 것).
"""

import os
import sys
import time

T0 = time.time()
TOTAL = float(os.environ.get("P2_TOTAL", 3600))
MARGIN = float(os.environ.get("P2_MARGIN", 420))
MODEL = os.environ.get("P2_MODEL", "Skywork/Skywork-OR1-Math-7B")
STEPS = int(os.environ.get("P2_STEPS", 2048))
BATCHES = [int(x) for x in os.environ.get("P2_BATCHES", "8,32,64").split(",")]
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")
_done = False


def log(m):
    print(f"[probe2 {time.time() - T0:7.1f}s] {m}", file=sys.stderr, flush=True)


def left():
    return TOTAL - MARGIN - (time.time() - T0)


def run_probe(problems):
    global _done
    if _done:
        return
    _done = True
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    t = time.time()
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(MODEL, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True,
                                                 low_cpu_mem_usage=True, device_map="auto",
                                                 attn_implementation="sdpa").eval()
    import transformers
    log(f"로드 {time.time() - t:.0f}s · transformers {transformers.__version__} · torch {torch.__version__} · "
        f"GPU {torch.cuda.device_count()} · 총 VRAM "
        f"{sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9:.0f} GB")

    base = (problems * 64)[:max(BATCHES)]
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t_}"}],
                                       tokenize=False, add_generation_prompt=True) for t_ in base]

    def once(n, tag, compile_static=False):
        enc = tok(prompts[:n], return_tensors="pt", padding=True).to(model.device)
        kw = dict(do_sample=False, max_new_tokens=STEPS, min_new_tokens=STEPS, pad_token_id=tok.pad_token_id)
        if compile_static:
            kw["cache_implementation"] = "static"
        torch.cuda.empty_cache()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        el = time.time() - t0
        steps = out.shape[1] - enc["input_ids"].shape[1]
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log(f"{tag}: 배치 {n} · 스텝 {steps} · {el:.0f}s · **{el * 1000 / max(1, steps):.1f} ms/step** · "
            f"시퀀스당 {steps / el:.1f} tok/s · 합산 {n * steps / el:.0f} tok/s · peak VRAM {mem:.1f} GB")
        return el

    for n in BATCHES:
        if left() < 300:
            log(f"예산 부족으로 배치 {n} 생략"); continue
        try:
            once(n, f"A_batch{n}")
        except Exception as e:
            log(f"A_batch{n} 실패: {type(e).__name__} {str(e)[:140]}")
    if left() > 600:
        try:
            t0 = time.time()
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
            log("torch.compile 적용 (컴파일은 첫 호출에서 일어남)")
            once(BATCHES[0], "B_compile_static_1st", compile_static=True)
            log(f"  1회차 총 소요(컴파일 포함) {time.time() - t0:.0f}s")
            if left() > 300:
                once(BATCHES[0], "B_compile_static_2nd", compile_static=True)
        except Exception as e:
            log(f"compile 경로 실패: {type(e).__name__} {str(e)[:160]}")
    else:
        log("compile 경로는 예산 부족으로 생략")
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
        log(f"프로브 실패: {type(e).__name__} {str(e)[:160]}")
    return [True] * len(problems)
