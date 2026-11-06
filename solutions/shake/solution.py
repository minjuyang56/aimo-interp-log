"""AIMO Interp — submission "shake": run the judged model on the problem and answer-preserving variants,
score by answer agreement, within-batch median split. No training.

Design (pre-registered 2026-09-29; constants below are set from the Kaggle run, not from the leaderboard):
  1. load the judged model_id (bf16, local_files_only, PreTrainedTokenizerFast first)
  2. for each problem build V answer-preserving variants: rename / typos / distract (rule-based, no model)
  3. greedy-generate original + variants in one padded batch (seeded, deterministic);
     if a generation ends without an answer, force one (append </think> + "ANSWER:" cue, 24 more tokens)
  4. answer = last "ANSWER:" line or last \\boxed{}; a missing answer counts as disagreement
  5. score = fraction of variants whose answer equals the original's (original None → 0)
  6. within the batch, the top ceil(n/2) scores are True (ties: input order)
Safety:
  * global clock from first import; per-call budget = min(PER_MODEL_BUDGET, remaining − MARGIN);
    when the budget is short, max_new_tokens shrinks; when exhausted, remaining problems return True
  * OOM → halve the batch and retry (down to 1)
  * any other exception → all True for the call
"""

import gc
import os
import re
import time

import numpy as np

T0 = time.time()                                   # 전역 시계: 모듈 import 시점 = 채점 시작
TOTAL_BUDGET = 3600.0
MARGIN = 240.0                                     # 채점기 오버헤드·마지막 배치 여유
PER_MODEL_BUDGET = float(os.environ.get("SHAKE_MODEL_BUDGET", 700))
MAX_NEW = int(os.environ.get("SHAKE_MAX_NEW", 2048))   # Kaggle run1 실측 보고 확정
FORCE_TOKENS = int(os.environ.get("SHAKE_FORCE_TOKENS", 24))
VARIANTS = ("rename", "typos", "distract")         # V = 3, 전부 정답 불변
BATCH = int(os.environ.get("SHAKE_BATCH", 16))
QUANT = os.environ.get("SHAKE_QUANT", "none")      # 로컬 8 GB 검증용 "4bit". 서버는 none
SEED = 0

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
FORCE_SUFFIX = "\n</think>\n\nBased on the reasoning so far, the final answer is:\nANSWER:"
ANSWER_RE = re.compile(r"ANSWER:\s*([^\n<]+)")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
DISTRACTORS = [
    "Note that the problem was originally posed at a regional competition in 2019.",
    "A friend of the author suggested this problem after a long hike.",
    "The committee that wrote this problem also enjoys chess on weekends.",
]


# ------------------------------------------------------------------ variants (rule-based)
def edit_rename(text, rng):
    def chunk(c):
        c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + 15) % 26 + 65), c)
        return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: LOWER_PAIRS.get(m.group(1), m.group(1)), c)
    def span(m):
        s, out, pos = m.group(0), [], 0
        for cmd in LATEX_CMD.finditer(s):
            out.append(chunk(s[pos:cmd.start()])); out.append(cmd.group(0)); pos = cmd.end()
        out.append(chunk(s[pos:])); return "".join(out)
    return MATH_SPAN.sub(span, text)


def edit_typos(text, rng, k=3):
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    words = [m for m in re.finditer(r"[A-Za-z]{5,}", text) if not any(a <= m.start() < b for a, b in spans)]
    if not words:
        return text
    out = list(text)
    for i in sorted(rng.choice(len(words), size=min(k, len(words)), replace=False), reverse=True):
        m = words[i]; j = m.start() + 1 + int(rng.integers(m.end() - m.start() - 3))
        out[j], out[j + 1] = out[j + 1], out[j]
    return "".join(out)


def edit_distract(text, rng):
    return text.rstrip() + " " + DISTRACTORS[int(rng.integers(len(DISTRACTORS)))]


EDITS = {"rename": edit_rename, "typos": edit_typos, "distract": edit_distract}


def extract_answer(text):
    text = THINK_RE.sub("", text)
    m = ANSWER_RE.findall(text); hit = [m[-1]] if m else BOXED_RE.findall(text)
    if not hit:
        return None
    a = hit[-1].strip().rstrip(".").replace(",", "").replace("$", "").replace(" ", "")
    return a or None


# ------------------------------------------------------------------ model
def load(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except (OSError, TypeError, ValueError):
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto")
    if QUANT == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    return tok, model


def _gen(tok, model, prompts, max_new):
    import torch
    torch.manual_seed(SEED)
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(r, skip_special_tokens=True) for r in new]


def generate_all(tok, model, prompts, max_new, batch):
    """OOM이면 배치를 반으로 줄여 재시도. 반환: 프롬프트 순서대로 텍스트."""
    import torch
    outs = []
    i = 0
    while i < len(prompts):
        chunk = prompts[i:i + batch]
        try:
            outs.extend(_gen(tok, model, chunk, max_new)); i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
    return outs


def budget_max_new(n_seq):
    """남은 시간으로 max_new_tokens를 정한다. 보수적 처리량 가정(서버 bf16 배치): ~8 tok/s per sequence."""
    remaining = TOTAL_BUDGET - MARGIN - (time.time() - T0)
    budget = min(PER_MODEL_BUDGET, remaining)
    if budget < 60:
        return 0
    est_tok_s_per_seq = 8.0
    rounds = max(1, -(-n_seq // BATCH))
    cap = int(budget * est_tok_s_per_seq / rounds)
    return max(0, min(MAX_NEW, cap))


def median_split(scores):
    n = len(scores); k = (n + 1) // 2
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    top = set(order[:k])
    return [bool(i in top) for i in range(n)]


def score_problems(model_id, problems):
    rng = np.random.default_rng(SEED)
    texts, owner, kind = [], [], []
    for pi, p in enumerate(problems):
        texts.append(p); owner.append(pi); kind.append("original")
        for v in VARIANTS:
            t = EDITS[v](p, rng)
            if t and t != p:
                texts.append(t); owner.append(pi); kind.append(v)
    max_new = budget_max_new(len(texts))
    if max_new == 0:
        return None
    tok, model = load(model_id)
    try:
        prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}],
                                           tokenize=False, add_generation_prompt=True) for t in texts]
        bodies = generate_all(tok, model, prompts, max_new, BATCH)
        answers = [extract_answer(b) for b in bodies]
        need = [i for i, a in enumerate(answers) if a is None]
        if need and FORCE_TOKENS > 0 and (TOTAL_BUDGET - MARGIN - (time.time() - T0)) > 30:
            forced = generate_all(tok, model, [prompts[i] + bodies[i] + FORCE_SUFFIX for i in need], FORCE_TOKENS, BATCH)
            for i, tail in zip(need, forced):
                answers[i] = extract_answer(FORCE_SUFFIX + tail)
    finally:
        del model
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    scores = []
    for pi in range(len(problems)):
        orig = next((answers[i] for i in range(len(texts)) if owner[i] == pi and kind[i] == "original"), None)
        var = [answers[i] for i in range(len(texts)) if owner[i] == pi and kind[i] != "original"]
        scores.append(0.0 if orig is None or not var else float(np.mean([a == orig for a in var])))
    return scores


def are_robust(model_id: str, reasoning_effort: str = "default",
               problems: list[str] | None = None) -> list[bool]:
    if problems is None:
        problems = []
    if not isinstance(reasoning_effort, str):          # legacy positional call
        problems, reasoning_effort = list(reasoning_effort), "default"
    if not problems:
        return []
    try:
        scores = score_problems(model_id, problems)
        if scores is None:
            return [True] * len(problems)               # 예산 소진
        out = median_split(scores)
    except Exception:
        return [True] * len(problems)
    if len(out) != len(problems) or any(type(x) is not bool for x in out):
        return [True] * len(problems)
    return out
