"""H20c — 시간 모델을 '스텝 기반'으로 바꾼 '원본만 풀기' 판정. 사전등록 2026-10-03 (지휘 #60).

왜 바꿨나 (측정이 모델을 뒤집었다)
  디코딩은 메모리 대역폭에 묶여서 **배치를 키워도 스텝당 시간이 거의 그대로**다. 실측:
    워커  배치 10 · 16k 문맥 → 1297s / 16384스텝 = **79 ms/step**
    워커  프로브 배치 32 · 2k  → 약 54 ms/step
    T4    배치 8  · 8192스텝   → 2213s = 270 ms/step (static 478, 문제별 개별은 배치를 포기해 8배 벽시계)
  따라서 호출당 벽시계 ≈ **(가장 긴 시퀀스 길이 = cap) × ms/step** 이고 **k 와 문제 수에는 거의 무관**하다.
  '토큰 총량 ÷ tok/s' 로 잡던 옛 모델은 틀렸다. **비싼 것은 cap 하나뿐이고 k 를 늘리는 건 사실상 공짜다.**

그래서
  1. 한 호출의 모든 문제 × k 를 **한 배치**로 돌린다(OOM 이면 반으로 쪼갠다).
  2. 허용 스텝 = 호출예산(s) × 1000 / ms_per_step × 0.85(여유) → 그 이하의 **가장 큰 cap** 을 고른다.
  3. k 는 VRAM 이 허락하는 한 크게(기본 4, 문제가 많으면 2). 평균이 안정될수록 판정이 조용해진다.
  4. 임계값은 **cap 별 고정값**(train, 두 클래스 중앙값의 중점). 평균 토큰의 기댓값은 k 에 불변이므로 k 별로 나누지 않는다.
  5. 그 cap 에서 train AUC 가 더 높은 쪽(H20a 평균토큰 / H20b 완주율)을 쓴다.
  6. 호출은 모델 4종 = 4번으로 가정하고 호출당 예산 = (3300 − 경과) / 남은 호출.
안전장치: 문제 해시 캐시 · 예산 부족 시 H15A 텍스트 규칙 · 모든 예외 True. 로그는 stderr.
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
SOFT = float(os.environ.get("H20_SOFT", 3300))
EXPECTED_CALLS = int(os.environ.get("H20_CALLS", 4))
MS0 = float(os.environ.get("H20_MS_PER_STEP", 80))
SAFETY = float(os.environ.get("H20_SAFETY", 0.85))
K_BIG = int(os.environ.get("H20_K", 4))
K_SMALL = int(os.environ.get("H20_K_SMALL", 2))
N_BIG = int(os.environ.get("H20_N_BIG", 12))          # 이보다 문제가 많으면 k 를 줄인다
CAPS = (6144, 8192, 12288, 16384)
MIN_CAP = int(os.environ.get("H20_MIN_CAP", 6144))
ART = Path(__file__).resolve().parent / "artifacts" / "h20_grid.json"
ART_BASE = Path(__file__).resolve().parent / "artifacts" / "h20.json"
CACHE = Path(os.environ.get("H20_CACHE", "/tmp/h20c_cache.json"))
STATE = Path(os.environ.get("H20_STATE", "/tmp/h20c_state.json"))
INSTRUMENT = os.environ.get("H20_MODEL", "Skywork/Skywork-OR1-Math-7B")
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
_m = {"model": None, "tok": None, "tried": False}


def log(msg):
    print(f"[h20c {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def remaining():
    return SOFT - (time.time() - T0)


def jload(p, d):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return d


def jsave(p, o):
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(o), encoding="utf-8")
    except Exception:
        pass


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


def choose(budget_s, ms_per_step):
    """허용 스텝 → cap, 그 cap 에서 train AUC 가 높은 지표. 못 고르면 None."""
    art = jload(ART, {})
    allowed = budget_s * 1000.0 / max(1e-6, ms_per_step) * SAFETY
    caps = [c for c in CAPS if c <= allowed]
    if not caps:
        return None
    cap = max(caps)
    fit = art.get("fit", {})
    a_key, b_key = f"H20a_k2_c{cap}", f"H20b_k2_c{cap}"
    tag = "H20a" if fit.get(a_key, {}).get("auc", 0) >= fit.get(b_key, {}).get("auc", 0) else "H20b"
    return cap, tag, art["thr"][f"{tag}_k2_c{cap}"], fit.get(f"{tag}_k2_c{cap}", {}).get("auc"), allowed


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
    """한 배치로 생성. 반환: (문제별 토큰수 리스트, 스텝 수, 초)."""
    import torch
    tok, model = get_model()
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, num_return_sequences=k,
                             max_new_tokens=cap, pad_token_id=tok.pad_token_id)
    el = time.time() - t0
    new = out[:, enc["input_ids"].shape[1]:]
    toks = [[int((r != tok.pad_token_id).sum()) for r in new[i * k:(i + 1) * k]] for i in range(len(texts))]
    return toks, int(new.shape[1]), el


def decide(problems):
    art = jload(ART, {})
    cache = jload(CACHE, {})
    state = jload(STATE, {"calls": 0, "ms": MS0})
    state["calls"] = int(state.get("calls", 0)) + 1
    ms = float(state.get("ms", MS0))
    jsave(STATE, state)
    budget = max(0.0, remaining()) / max(1, EXPECTED_CALLS - (state["calls"] - 1))
    out = [None] * len(problems)
    todo = []
    for i, p in enumerate(problems):
        c = cache.get(key_of(p))
        if c:
            out[i] = bool(c["pred"])
        else:
            todo.append(i)
    log(f"호출 {state['calls']}/{EXPECTED_CALLS} · 문제 {len(problems)} · 캐시 적중 {len(problems) - len(todo)} · "
        f"남은 {remaining():.0f}s · 호출예산 {budget:.0f}s · ms/step 추정 {ms:.0f}")
    if todo:
        pick = choose(min(budget, remaining()), ms)
        if pick is None:
            log(f"허용 스텝이 최소 cap({MIN_CAP})에 못 미침 → 전부 폴백")
        else:
            cap, tag, thr, auc, allowed = pick
            k = K_BIG if len(todo) <= N_BIG else K_SMALL
            log(f"  선택 cap {cap} · k {k} · 지표 {tag} (train AUC {auc}) · 임계 {thr:.1f} · 허용스텝 {allowed:.0f}")
            batch = len(todo)
            i = 0
            while i < len(todo):
                idx = todo[i:i + batch]
                try:
                    toks, steps, el = gen([problems[j] for j in idx], k, cap)
                except Exception as e:
                    import torch
                    if "out of memory" in str(e).lower() and batch > 1:
                        batch = max(1, batch // 2)
                        torch.cuda.empty_cache()
                        log(f"  OOM → 배치 {batch}")
                        continue
                    log(f"  생성 실패: {type(e).__name__} {str(e)[:120]} → 폴백")
                    break
                ms = max(1.0, 0.5 * ms + 0.5 * el * 1000.0 / max(1, steps))
                state["ms"] = ms
                jsave(STATE, state)
                tot = sum(sum(x) for x in toks)
                log(f"  배치 {len(idx)}문제 × k{k} = {len(idx) * k}시퀀스 · {el:.0f}s · 스텝 {steps} · "
                    f"**{el * 1000 / max(1, steps):.1f} ms/step** · {tot} 토큰 · ms/step 추정 {ms:.0f}")
                for j, tk in zip(idx, toks):
                    v = sum(tk) / len(tk) if tag == "H20a" else sum(1 for t_ in tk if t_ < cap) / len(tk)
                    pred = bool(v <= thr) if tag == "H20a" else bool(v >= thr)
                    cache[key_of(problems[j])] = {"pred": pred, "v": v, "cap": cap, "tag": tag, "k": k}
                    out[j] = pred
                jsave(CACHE, cache)
                i += len(idx)
                if remaining() < 60:
                    log("  남은 예산 부족 → 중단")
                    break
    if any(o is None for o in out):
        fb = fallback(problems)
        n_fb = sum(1 for o in out if o is None)
        out = [fb[j] if out[j] is None else out[j] for j in range(len(out))]
        log(f"폴백으로 채운 문제 {n_fb}")
    log(f"판정 완료 · True {sum(out)}/{len(out)}")
    return [bool(o) for o in out]
