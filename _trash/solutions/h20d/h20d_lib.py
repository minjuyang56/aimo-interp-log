"""H20d — ms/step 을 (배치, 문맥)으로 예측해 cap 을 정하는 '원본만 풀기' 판정. 사전등록 2026-10-04 (지휘 #61).

#60 의 '배치는 공짜' 가정은 워커 probe2 가 기각했다:
  짧은 문맥 2048스텝에서 B=8 **25.0** · B=32 **54.4** · B=64 **95.5** ms/step → ms ≈ 15 + 1.25·B (거의 선형)
  긴 문맥(h20 1차, B=10, cap 16384)은 **79.2** ms/step — 같은 배치의 짧은 문맥 대비 약 3배
  합산 처리량은 600~670 tok/s 에서 포화. torch.compile 은 워커에 C++ 컴파일러가 없어 불가(InvalidCxxCompiler).
그래서 **k 를 늘리는 것도 공짜가 아니다**(k=4 는 배치 2배 = 시간 거의 2배).

시간 모델 (analysis/ms_per_step_model.py 와 동일, 보수 배수 1.15)
  ms(B, L) = 1.15 × (15 + 0.375·B + 7.21e-4·B·L)
  한 배치 소요(초) = C(15 + 0.375B + 7.21e-4·B·P)/1000 + 7.21e-4·B·C²/2000, 그 뒤 ×1.15
  배치를 쪼개면 고정비 15ms 가 그 수만큼 더 붙으므로 **한 배치가 항상 유리**.

정책
  k = 2 기본, 문제가 적고(≤6) 예산이 남으면 3. cap 은 예산 안에 들어가는 최대값.
  배치가 끝날 때마다 실측 ms/step 으로 **배율(scale)** 을 재보정해 다음 호출 예측에 쓴다.
  생성 중에도 마감이 가까우면 **StoppingCriteria 로 조기 종료**하고, 그 시점 스텝 수에 맞는 cap 임계값으로 판정한다.
  그래도 못 미치면 H15A 텍스트 규칙. 모든 예외 True. 로그는 stderr.
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

T0 = time.time()
TOTAL = float(os.environ.get("H20_TOTAL_BUDGET", 3600))
SOFT = float(os.environ.get("H20_SOFT", 3300))
EXPECTED_CALLS = int(os.environ.get("H20_CALLS", 4))
CAPS = (4096, 6144, 8192, 12288, 16384)
A_, B_, C_, MULT, PROMPT = 15.0, 0.375, 7.21e-4, 1.15, 200.0
K_DEFAULT = int(os.environ.get("H20_K", 2))
K_RICH = int(os.environ.get("H20_K_RICH", 3))
N_FEW = int(os.environ.get("H20_N_FEW", 6))
ART = Path(__file__).resolve().parent / "artifacts" / "h20_grid.json"
ART_BASE = Path(__file__).resolve().parent / "artifacts" / "h20.json"
CACHE = Path(os.environ.get("H20_CACHE", "/tmp/h20d_cache.json"))
STATE = Path(os.environ.get("H20_STATE", "/tmp/h20d_state.json"))
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
    print(f"[h20d {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


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


def seconds_for(batch, cap, scale=1.0):
    lin = (A_ + B_ * batch + C_ * batch * PROMPT) * cap / 1000.0
    quad = C_ * batch * cap * cap / 2000.0
    return (lin + quad) * MULT * scale


def max_cap(batch, budget, scale=1.0):
    ok = [c for c in CAPS if seconds_for(batch, c, scale) <= budget]
    return max(ok) if ok else None


def nearest_cap(steps):
    ok = [c for c in CAPS if c <= steps]
    return max(ok) if ok else None


def metric_for(cap):
    """그 cap 에서 train AUC 가 높은 지표와 임계값."""
    art = jload(ART, {})
    fit = art.get("fit", {})
    have = sorted({int(k.split("_c")[1]) for k in art.get("thr", {})})
    if cap not in have:                       # 격자에 없는 cap 이면 가장 가까운 값으로 (전체 폴백 방지)
        cap = min(have, key=lambda c: abs(c - cap)) if have else cap
    a_k, b_k = f"H20a_k2_c{cap}", f"H20b_k2_c{cap}"
    tag = "H20a" if fit.get(a_k, {}).get("auc", 0) >= fit.get(b_k, {}).get("auc", 0) else "H20b"
    return tag, art["thr"][f"{tag}_k2_c{cap}"], fit.get(f"{tag}_k2_c{cap}", {}).get("auc")


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


def gen(texts, k, cap, deadline):
    """한 배치 생성. 마감이 지나면 StoppingCriteria 로 조기 종료. 반환: (토큰수, 실제 스텝, 초)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    tok, model = get_model()

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return time.time() > deadline

    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, num_return_sequences=k,
                             max_new_tokens=cap, pad_token_id=tok.pad_token_id,
                             stopping_criteria=StoppingCriteriaList([Deadline()]))
    el = time.time() - t0
    new = out[:, enc["input_ids"].shape[1]:]
    toks = [[int((r != tok.pad_token_id).sum()) for r in new[i * k:(i + 1) * k]] for i in range(len(texts))]
    return toks, int(new.shape[1]), el


def decide(problems):
    cache = jload(CACHE, {})
    state = jload(STATE, {"calls": 0, "scale": 1.0})
    state["calls"] = int(state.get("calls", 0)) + 1
    scale = float(state.get("scale", 1.0))
    jsave(STATE, state)
    budget = max(0.0, remaining()) / max(1, EXPECTED_CALLS - (state["calls"] - 1))
    out = [None] * len(problems)
    todo = [i for i, p in enumerate(problems) if key_of(p) not in cache]
    for i, p in enumerate(problems):
        c = cache.get(key_of(p))
        if c:
            out[i] = bool(c["pred"])
    log(f"호출 {state['calls']}/{EXPECTED_CALLS} · 문제 {len(problems)} · 캐시 {len(problems) - len(todo)} · "
        f"남은 {remaining():.0f}s · 호출예산 {budget:.0f}s · 배율 {scale:.2f}")
    if todo:
        try:                                   # 로딩을 먼저 끝내고 cap 을 정한다 (로드 시간이 예산을 먹지 않도록)
            t_load = time.time()
            get_model()
            spent = time.time() - t_load
            if spent > 1:
                budget = max(0.0, budget - spent)
                log(f"  로드에 {spent:.0f}s → 남은 호출예산 {budget:.0f}s")
        except Exception as e:
            log(f"  모델 로드 실패: {type(e).__name__} {str(e)[:120]}")
        k = K_RICH if (len(todo) <= N_FEW and budget > 900) else K_DEFAULT
        batch = len(todo) * k
        cap = max_cap(batch, min(budget, remaining()), scale)
        if cap is None:
            k = 1
            batch = len(todo)
            cap = max_cap(batch, min(budget, remaining()), scale)
        if cap is None:
            log(f"배치 {batch} 로는 최소 cap({CAPS[0]})도 예산 초과 → 전부 폴백")
        else:
            tag, thr, auc = metric_for(cap)
            est = seconds_for(batch, cap, scale)
            log(f"  배치 {len(todo)}문제 × k{k} = {batch}시퀀스 · cap {cap} · 예상 {est:.0f}s · "
                f"지표 {tag}(train AUC {auc}) · 임계 {thr:.1f}")
            deadline = time.time() + min(budget, remaining()) - 30
            try:
                toks, steps, el = gen([problems[j] for j in todo], k, cap, deadline)
                pred_ms = seconds_for(batch, cap, 1.0) * 1000 / max(1, cap)
                obs_ms = el * 1000 / max(1, steps)
                if steps >= 256:               # 스텝이 적으면 프리필·오버헤드가 섞여 배율이 오염된다
                    scale = max(0.3, min(3.0, 0.5 * scale + 0.5 * obs_ms / max(1e-6, pred_ms)))
                    state["scale"] = scale
                    jsave(STATE, state)
                else:
                    log(f"  스텝 {steps} < 256 → 배율 갱신 생략")
                eff_cap = nearest_cap(steps)
                if eff_cap is None:
                    # 최소 cap 보다도 일찍 끊겼다 → 어떤 임계값으로도 보정이 안 맞는다. 이 배치는 버리고 폴백.
                    log(f"  스텝 {steps} < 최소 cap {CAPS[0]} → 보정 불가, 이 배치는 폴백으로 넘김")
                    raise RuntimeError("too few steps")
                if eff_cap != cap:
                    tag, thr, auc = metric_for(eff_cap)
                    log(f"  조기 종료: 스텝 {steps} < cap {cap} → 임계값을 cap {eff_cap} 기준으로 교체")
                log(f"  완료 {el:.0f}s · 스텝 {steps} · **{obs_ms:.1f} ms/step**(예측 {pred_ms:.1f}) · 배율 {scale:.2f}")
                for j, tk in zip(todo, toks):
                    v = sum(tk) / len(tk) if tag == "H20a" else sum(1 for t_ in tk if t_ < eff_cap) / len(tk)
                    pred = bool(v <= thr) if tag == "H20a" else bool(v >= thr)
                    cache[key_of(problems[j])] = {"pred": pred, "v": v, "cap": eff_cap, "tag": tag, "k": k}
                    out[j] = pred
                jsave(CACHE, cache)
            except Exception as e:
                log(f"  생성 실패: {type(e).__name__} {str(e)[:140]} → 폴백")
    if any(o is None for o in out):
        fb = fallback(problems)
        n_fb = sum(1 for o in out if o is None)
        out = [fb[j] if out[j] is None else out[j] for j in range(len(out))]
        log(f"폴백으로 채운 문제 {n_fb}")
    log(f"판정 완료 · True {sum(out)}/{len(out)}")
    return [bool(o) for o in out]
