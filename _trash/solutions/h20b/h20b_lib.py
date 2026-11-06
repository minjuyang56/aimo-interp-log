"""H20b — 예산에 맞춰 (k, cap) 을 스스로 고르는 '원본만 풀기' 판정. 사전등록 2026-10-02/03 (지휘 #55·#57).

h20 1차 제출에서 배운 것 (워커 로그 실측)
  * 호출은 모델별로 4번, 총 14 인스턴스였지만 **고유 문제는 10개**였고 해시 캐시가 1·1·2건 적중했다 → 캐시 유지.
  * 속도가 재앙이었다: 5문제(10시퀀스) 16384 에 **1297초 · 83 tok/s**(시퀀스당 8). 총 2443초.
    프로브의 2048토큰 구간에서는 시퀀스당 18 tok/s 였으므로 **긴 문맥에서 급격히 느려진다**.
  * 최종셋은 30~40문제라 같은 설정이면 3600초를 확실히 넘는다.

그래서 바뀐 점
  1. **호출당 예산**: 전체 호출 수를 모르므로 `남은예산 / max(1, 4 − 완료호출수)` 로 잡는다(모델 4종 가정).
  2. **설정 자동 선택**: 아티팩트의 (k, cap) 격자 중 **예산 안에 들어가는 것 중 train AUC 가 가장 높은 것**을 고른다.
     비용 추정 = 남은문제수 × k × mean_tokens(cap) / 처리율. 처리율은 80 tok/s 로 시작해 **배치마다 실측으로 갱신**한다.
  3. **임계값은 설정마다 train 에서 미리 고정**(두 클래스 중앙값의 중점) — 설정이 바뀌어도 보정이 맞는다.
  4. **기본 패딩 배치 generate** 를 쓴다. T4 벤치에서 static cache 는 1.9배, 문제별 개별 생성은 2.8배 느렸다.
     (연속 배칭 generate_batch 는 이 버전에서 토큰을 하나도 돌려주지 않아 사용 불가.)
  5. 예산이 모자라면 남은 문제는 H15A 텍스트 규칙(n_sym)으로. 예외는 전부 True.
로그는 stderr 로만.
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

T0 = time.time()
TOTAL_BUDGET = float(os.environ.get("H20_TOTAL_BUDGET", 3600))
MARGIN = float(os.environ.get("H20_MARGIN", 240))
EXPECTED_CALLS = int(os.environ.get("H20_CALLS", 4))
ART = Path(__file__).resolve().parent / "artifacts" / "h20_grid.json"
ART_BASE = Path(__file__).resolve().parent / "artifacts" / "h20.json"
CACHE = Path(os.environ.get("H20_CACHE", "/tmp/h20b_cache.json"))
STATE = Path(os.environ.get("H20_STATE", "/tmp/h20b_state.json"))
INSTRUMENT = os.environ.get("H20_MODEL", "Skywork/Skywork-OR1-Math-7B")
BATCH = int(os.environ.get("H20_BATCH", 8))
TPUT0 = float(os.environ.get("H20_TPUT", 80))          # 워커 실측 출발값
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
# T4 벤치(시퀀스 8 · 8192토큰): 기본 패딩 배치 24.3 tok/s vs static cache 12.9 vs 문제별 개별 8.7.
# static 이 1.9배 "느렸다" → 기본은 일반 캐시. H20_STATIC=1 로만 켠다.
_m = {"model": None, "tok": None, "tried": False, "static": os.environ.get("H20_STATIC") == "1"}


def log(msg):
    print(f"[h20b {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def remaining():
    return TOTAL_BUDGET - MARGIN - (time.time() - T0)


def jload(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def jsave(p, obj):
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------- 텍스트 폴백
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
    thr = jload(ART_BASE, {}).get("fallback_n_sym_thr", 12.5)
    return [bool(n_sym(p) <= thr) for p in problems]


def key_of(t):
    return hashlib.sha1(t.strip().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 설정 선택
def pick_setting(n_todo, budget, tput):
    """예산 안에 들어가는 (지표, k, cap) 중 train AUC 최고. 아무것도 안 되면 None."""
    art = jload(ART, {})
    order = art.get("order") or sorted(art.get("fit", {}), key=lambda k: -art["fit"][k]["auc"])
    for key in order:
        tag, ks, cs = key.split("_")
        k, cap = int(ks[1:]), int(cs[1:])
        cost = n_todo * k * art["mean_tokens"][str(cap)] / max(1.0, tput)
        if cost <= budget:
            return key, tag, k, cap, cost
    return None


# ---------------------------------------------------------------- 모델
def get_model():
    if _m["model"] is not None or _m["tried"]:
        return _m["tok"], _m["model"]
    _m["tried"] = True
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    t = time.time()
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(INSTRUMENT, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(INSTRUMENT, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto", dtype=torch.bfloat16,
              attn_implementation="sdpa")
    if os.environ.get("H20_QUANT") == "4bit":
        from transformers import BitsAndBytesConfig
        kw.pop("dtype")
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(INSTRUMENT, **kw).eval()
    except Exception:
        kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(INSTRUMENT, **kw).eval()
    log(f"{INSTRUMENT} 로드 {time.time() - t:.1f}s")
    _m["tok"], _m["model"] = tok, model
    return tok, model


def gen(texts, k, cap):
    import torch
    tok, model = get_model()
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(0)
    kw = dict(do_sample=True, temperature=0.6, top_p=0.95, num_return_sequences=k,
              max_new_tokens=cap, pad_token_id=tok.pad_token_id)
    if _m["static"]:
        try:
            with torch.no_grad():
                out = model.generate(**enc, cache_implementation="static", **kw)
        except Exception as e:
            log(f"static cache 실패({type(e).__name__}) → 기본 캐시")
            _m["static"] = False
            with torch.no_grad():
                out = model.generate(**enc, **kw)
    else:
        with torch.no_grad():
            out = model.generate(**enc, **kw)
    new = out[:, enc["input_ids"].shape[1]:]
    res = [[int((r != tok.pad_token_id).sum()) for r in new[i * k:(i + 1) * k]] for i in range(len(texts))]
    _m["last_steps"] = int(new.shape[1])        # 디코딩 스텝 수(배치에서 가장 긴 시퀀스 기준)
    return res


def decide(problems):
    art = jload(ART, {})
    cache = jload(CACHE, {})
    state = jload(STATE, {"calls": 0})
    state["calls"] = int(state.get("calls", 0)) + 1
    jsave(STATE, state)
    budget = max(0.0, remaining()) / max(1, EXPECTED_CALLS - (state["calls"] - 1))
    out = [None] * len(problems)
    todo = []
    for i, p in enumerate(problems):
        c = cache.get(key_of(p))
        if c:
            thr = art["thr"][c["key"]]
            v = sum(c["toks"]) / len(c["toks"]) if c["key"].startswith("H20a") else c["fin"]
            out[i] = bool(v <= thr) if c["key"].startswith("H20a") else bool(v >= thr)
        else:
            todo.append(i)
    log(f"호출 {state['calls']} · 문제 {len(problems)} · 캐시 적중 {len(problems) - len(todo)} · "
        f"남은예산 {remaining():.0f}s · 이번 호출 예산 {budget:.0f}s")
    tput = float(state.get("tput", TPUT0))
    i = 0
    while i < len(todo):
        left_budget = min(budget - (time.time() - T0 - state.get("t_start", time.time() - T0)), remaining())
        pick = pick_setting(len(todo) - i, max(0.0, min(budget, remaining())), tput)
        if pick is None or remaining() < 60:
            log(f"예산 부족(남은 {remaining():.0f}s, 처리율 {tput:.0f} tok/s) → 남은 {len(todo) - i}문제 폴백")
            break
        key, tag, k, cap, cost = pick
        n = min(BATCH, len(todo) - i)
        idx = todo[i:i + n]
        log(f"  설정 {key} (k={k}, cap={cap}) · 예상 {cost:.0f}s · 이번 배치 {n}문제")
        t = time.time()
        try:
            toks = gen([problems[j] for j in idx], k, cap)
        except Exception as e:
            log(f"  생성 실패: {type(e).__name__} {str(e)[:120]} → 폴백")
            break
        el = time.time() - t
        tot = sum(sum(x) for x in toks)
        tput = max(10.0, 0.5 * tput + 0.5 * tot / max(1e-6, el))
        state["tput"] = tput
        jsave(STATE, state)
        steps = _m.get("last_steps") or 1
        log(f"  배치 {n}문제 {el:.0f}s · {tot} 토큰 · {tot / max(1e-6, el):.1f} tok/s · "
            f"스텝 {steps} · **{el * 1000 / max(1, steps):.1f} ms/step** · 처리율 추정 {tput:.0f}")
        thr = art["thr"][key]
        for j, tk in zip(idx, toks):
            fin = sum(1 for t_ in tk if t_ < cap) / len(tk)
            cache[key_of(problems[j])] = {"toks": tk, "fin": fin, "key": key}
            out[j] = bool(sum(tk) / len(tk) <= thr) if tag == "H20a" else bool(fin >= thr)
        jsave(CACHE, cache)
        i += n
    if any(o is None for o in out):
        fb = fallback(problems)
        n_fb = sum(1 for o in out if o is None)
        out = [fb[j] if out[j] is None else out[j] for j in range(len(out))]
        log(f"폴백으로 채운 문제 {n_fb}")
    log(f"판정 완료 · True {sum(out)}/{len(out)}")
    return [bool(o) for o in out]
