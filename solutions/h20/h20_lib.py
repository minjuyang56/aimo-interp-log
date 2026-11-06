"""H20 제출 라이브러리 — '원본만 풀기' 길이 신호. 사전등록 2026-10-02 (지휘 #55·#56).

판정: 판정 대상과 무관하게 **Skywork-OR1-Math-7B 하나**로 원본 문제를 k=2 생성하고,
      **평균 생성 토큰 수 <= 임계값이면 True(robust)**. 임계값은 공식 33문제에서 '두 클래스 중앙값의 중점'으로 고정.
      근거: H20a 가 공식 라벨에서 AUC 0.850(문제길이 잔차 0.792). 라벨이 문제 단위(H2)라 모델 하나로 충분하다.

설계
  * **문제 해시 캐시**: 첫 호출에서 생성한 결과를 /tmp 에 저장하고, 다른 model_id 로 같은 문제가 다시 오면 재사용한다.
    (채점기는 model_id 별로 그룹을 나눠 부르므로 같은 문제가 여러 번 온다 — 캐시가 없으면 비용이 모델 수만큼 곱해진다.)
  * **디코딩은 H14 와 동일**: temperature 0.6 · top_p 0.95 · seed 0 · k=2 · 같은 solve 프롬프트. 임계값 보정이 맞으려면 같아야 한다.
  * **상한 자동 강등**: 기본 16384. 시간이 모자랄 것 같으면 8192 로 낮추고 **그 상한에 맞는 임계값**을 쓴다.
  * **시간 가드**: 전역 시계. 남은 예산이 모자라면 남은 문제는 H15A 텍스트 규칙(n_sym)으로 판정한다. 예외는 전부 True.
로그는 stderr 로만(stdout 은 채점기 몫).
"""

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

T0 = time.time()
TOTAL_BUDGET = float(os.environ.get("H20_TOTAL_BUDGET", 3600))
SOFT_LIMIT = float(os.environ.get("H20_SOFT_LIMIT", 3300))     # 이 시각을 넘길 것 같으면 폴백
ART = Path(__file__).resolve().parent / "artifacts" / "h20.json"
CACHE = Path(os.environ.get("H20_CACHE", "/tmp/h20_cache.json"))
INSTRUMENT = os.environ.get("H20_MODEL", "Skywork/Skywork-OR1-Math-7B")
MAX_NEW = int(os.environ.get("H20_MAX_NEW", 16384))
MAX_NEW_LOW = int(os.environ.get("H20_MAX_NEW_LOW", 8192))
K = int(os.environ.get("H20_K", 2))
BATCH = int(os.environ.get("H20_BATCH", 8))
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
_state = {"model": None, "tok": None, "cap": MAX_NEW, "tried": False}


def log(msg):
    print(f"[h20 {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def remaining():
    return TOTAL_BUDGET - (time.time() - T0)


def art():
    return json.loads(ART.read_text(encoding="utf-8"))


def thr_for(a, cap):
    """상한에 맞는 임계값. 정확히 일치하는 키가 없으면 가장 가까운 상한의 값을 쓴다(폴백으로 떨어지지 않게)."""
    t = a["thr"]
    if str(cap) in t:
        return t[str(cap)]
    near = min(t, key=lambda k: abs(int(k) - cap))
    log(f"상한 {cap} 의 임계값이 없어 {near} 의 값을 쓴다")
    return t[near]


# ---------------------------------------------------------------- 텍스트 폴백 (H15A)
def n_sym(text):
    def chunk(c):
        c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + UPPER_SHIFT) % 26 + 65), c)
        return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: LOWER_PAIRS.get(m.group(1), m.group(1)), c)

    def span(m):
        s, out, pos = m.group(0), [], 0
        for cmd in LATEX_CMD.finditer(s):
            out.append(chunk(s[pos:cmd.start()])); out.append(cmd.group(0)); pos = cmd.end()
        out.append(chunk(s[pos:]))
        return "".join(out)
    rn = MATH_SPAN.sub(span, text)
    return sum(1 for a, b in zip(text, rn) if a != b) if len(rn) == len(text) else 0


def fallback(problems):
    thr = art()["fallback_n_sym_thr"]
    return [bool(n_sym(p) <= thr) for p in problems]


# ---------------------------------------------------------------- 캐시
def key_of(text):
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def load_cache():
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(c):
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(c), encoding="utf-8")
    except Exception as e:
        log(f"캐시 저장 실패(무시): {type(e).__name__}")


# ---------------------------------------------------------------- 모델
def get_model():
    if _state["model"] is not None or _state["tried"]:
        return _state["tok"], _state["model"]
    _state["tried"] = True
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    t0 = time.time()
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(INSTRUMENT, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(INSTRUMENT, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto", dtype=torch.bfloat16)
    if os.environ.get("H20_QUANT") == "4bit":                      # 로컬 스모크 전용
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
        kw.pop("dtype", None)
    try:
        kw["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(INSTRUMENT, **kw).eval()
    except Exception:
        kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(INSTRUMENT, **kw).eval()
    log(f"{INSTRUMENT} 로드 {time.time() - t0:.1f}s")
    _state["tok"], _state["model"] = tok, model
    return tok, model


def gen_tokens(texts, cap):
    """문제별 k 샘플의 생성 토큰 수 리스트. H14 와 같은 디코딩."""
    import torch
    tok, model = get_model()
    if model is None:
        raise RuntimeError("모델 없음")
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95,
                             num_return_sequences=K, max_new_tokens=cap, pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    res = []
    for i in range(len(texts)):
        rows = new[i * K:(i + 1) * K]
        res.append([int((r != tok.pad_token_id).sum()) for r in rows])
    return res


def decide(problems):
    a = art()
    cache = load_cache()
    out = [None] * len(problems)
    todo = []
    for i, p in enumerate(problems):
        c = cache.get(key_of(p))
        if c and c.get("cap") == _state["cap"]:
            out[i] = bool(sum(c["toks"]) / len(c["toks"]) <= thr_for(a, c["cap"]))
        else:
            todo.append(i)
    log(f"문제 {len(problems)} · 캐시 적중 {len(problems) - len(todo)} · 생성 대상 {len(todo)} · 상한 {_state['cap']}")
    batch = BATCH
    i = 0
    while i < len(todo):
        if remaining() < TOTAL_BUDGET - SOFT_LIMIT + 60:
            log(f"예산 부족({remaining():.0f}s) → 남은 {len(todo) - i}문제는 H15A 폴백")
            break
        if _state["cap"] == MAX_NEW and remaining() < 0.45 * TOTAL_BUDGET:
            _state["cap"] = MAX_NEW_LOW
            log(f"상한 강등 {MAX_NEW} → {MAX_NEW_LOW} (남은 {remaining():.0f}s)")
        idx = todo[i:i + batch]
        t0 = time.time()
        try:
            toks = gen_tokens([problems[j] for j in idx], _state["cap"])
        except Exception as e:
            import torch
            if "out of memory" in str(e).lower() and batch > 1:
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                log(f"OOM → 배치 {batch}")
                continue
            log(f"생성 실패: {type(e).__name__} {str(e)[:120]} → 폴백")
            break
        el = time.time() - t0
        tot = sum(sum(t) for t in toks)
        log(f"  배치 {len(idx)}문제 · {el:.1f}s · {tot} 토큰 · {tot / max(1e-6, el):.1f} tok/s · "
            f"문제당 {tot / max(1, len(idx)):.0f} 토큰")
        for j, tk in zip(idx, toks):
            cache[key_of(problems[j])] = {"toks": tk, "cap": _state["cap"]}
            out[j] = bool(sum(tk) / len(tk) <= thr_for(a, _state["cap"]))
        save_cache(cache)
        i += len(idx)
    if any(o is None for o in out):
        fb = fallback(problems)
        n_fb = sum(1 for o in out if o is None)
        for j in range(len(out)):
            if out[j] is None:
                out[j] = fb[j]
        log(f"폴백으로 채운 문제 {n_fb}")
    log(f"판정 완료 · True {sum(out)}/{len(out)}")
    return [bool(o) for o in out]
