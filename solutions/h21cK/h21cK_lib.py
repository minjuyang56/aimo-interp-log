"""H21cK — 상수 변형 **K개**의 drop 평균. 사전등록 2026-10-06 (지휘 #69).

h21c 와 다른 점은 하나다: 변형을 1개가 아니라 **K=4개**(결정적 격자) 만들어 drop 을 평균한다.
h21c 는 리더보드 0.43 이었고 총 13.5초만 썼다 — 예산의 0.4%%. 변형 1개로 잰 drop 은 주사위 한 번이고,
주최측 라벨은 여러 변형에서의 정확도 하락이다. 남는 예산으로 평균을 쌓는다.

측정 (train 33문제 × 4모델, 로컬 4bit, null 1000, 부트스트랩 1000 · `analysis/h21ck_eval.py`)
  K 안정화 곡선(v2 규칙, 평균 drop): K=1 → 2 → 4 → 8 → 16
    DeepSeek 0.712 · 0.719▲ · **0.765▲** · 0.715▲ · 0.715▲
    Skywork  0.685 · 0.712▲ · 0.658 · 0.650 · 0.650
    Qwen     0.431 · 0.538 · 0.588 · 0.523 · 0.492
    Olmo     0.365 · 0.388 · 0.404 · 0.404 · 0.381   (반대 방향)
  모델평균: K=1 0.548 · K=2 0.589 · **K=4 0.604** · K=8 0.573 · K=16 0.560 → **train 기준 K=4**
  규칙 개선(지수·첨자 제외)은 효과가 없었다: K=8 에서 +0.004 · +0.012 · −0.042 · −0.019.
  **중요**: h21c 가 쓴 변형(마지막 숫자 +1)의 0.802/0.741 은 여기서 재현되지 않는다(K=1 은 0.712/0.685,
  K=16 은 0.715/0.650). 단일 변형 결과는 **어느 변형을 고르느냐에 크게 좌우된다** — 평균이 추정을
  안정시키지만 안정된 값은 운 좋은 한 번보다 **낮다**. 그 사실을 알고 내는 제출물이다.

판정·안전장치는 h21c 와 동일: 호출 내 중앙값 분할 · 변형 못 만든 문제는 즉답 엔트로피 층 · 그 외 True ·
예산 가드 · 모델 해제 · 모든 예외 True.
"""


import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

T0 = time.time()
SOFT = float(os.environ.get("H21_SOFT", 3300))
EXPECTED_CALLS = int(os.environ.get("H21_CALLS", 4))
ANS_TOKENS = int(os.environ.get("H21_ANS_TOKENS", 8))
RESERVE = float(os.environ.get("H21_RESERVE", 20))        # 호출 끝에 남겨 두는 초
CACHE = Path(os.environ.get("H21_CACHE", "/tmp/h21ck_cache.json"))
STATE = Path(os.environ.get("H21_STATE", "/tmp/h21ck_state.json"))
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
FORCE = "</think>\nANSWER:"
NUM_RE = re.compile(r"(?<![\w.])([2-9]|[1-9]\d{1,3})(?![\w.])")
K_VAR = int(os.environ.get("H21_K_VAR", 4))               # train 에서 고른 K (모델평균 최고)
DELTAS = (1, -1, 2, 3, -2, 4, 5, -3, 6, 7, -4, 8)         # 결정적 증분 순서
RULE = os.environ.get("H21_RULE", "v2")                   # v2 = 지수·첨자 제외
_m = {"model": None, "tok": None, "id": None, "failed": set()}


def log(msg):
    print(f"[h21cK {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


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


def key_of(t, model_id, effort):
    """캐시는 (model_id, effort, 문제) 단위. 모델마다 자기 답·자기 확률이라 교차 재사용은 뜻이 없다."""
    raw = f"{model_id}|{effort}|{t.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _in_script(text, start):
    """그 숫자가 지수·첨자 안인가. 앞을 거슬러 `^`/`_`(+공백·중괄호)를 찾는다."""
    i = start - 1
    while i >= 0 and text[i] in " 	{":
        i -= 1
    return i >= 0 and text[i] in "^_"


def make_variants(text, k=None, rule=None):
    """결정적 상수 변형 k개. **난수 없음** — 같은 문제면 항상 같은 k개가 나온다.
    (위치 × 증분) 격자를 돌며 중복·무변화를 버린다. 새 값이 2 미만이면 +1 로 대체한다
    (0/1 은 수식 구조를 바꾸기 쉽다). rule='v2' 면 지수·첨자 위치를 후보에서 뺀다.

    규칙과 순서는 `experiments/variants_constants.py` 와 **같게 유지한다** — 측정과 제출물이 어긋나면
    train AUC 가 의미를 잃는다. 측정에서 v2·K=4 가 모델평균 최고였다."""
    k = K_VAR if k is None else k
    rule = RULE if rule is None else rule
    ms = [m for m in NUM_RE.finditer(text) if not (rule == "v2" and _in_script(text, m.start()))]
    if not ms:
        return []
    out, seen = [], {text}
    for d in DELTAS:
        for m in ms:
            if len(out) >= k:
                return out
            v = int(m.group(1))
            nv = v + d if v + d >= 2 else v + 1
            cand = text[:m.start()] + str(nv) + text[m.end():]
            if cand in seen:
                continue
            seen.add(cand)
            out.append(cand)
    return out


def get_model(model_id):
    """**판정 대상 모델 자신**을 bf16 으로 연다(워커는 96 GB 라 양자화 불필요).
    다른 모델이 오면 앞 모델을 놓아준다. 못 열면 (None, None)."""
    want = str(model_id or "").strip()
    if not want or want in _m["failed"]:
        return None, None
    if _m["id"] == want and _m["model"] is not None:
        return _m["tok"], _m["model"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    if _m["model"] is not None:
        log(f"  앞 모델 {_m['id']} 해제")
        _m["model"], _m["tok"], _m["id"] = None, None, None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa")
    if os.environ.get("H21_QUANT") == "4bit":              # 로컬 8 GB 스모크용
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    for name in [want] + ([want.split(":")[0]] if ":" in want else []):
        t = time.time()
        try:
            try:                                           # DeepSeek 은 AutoTokenizer 가 옛 Llama 로 열어 인코딩이 달라진다
                tok = PreTrainedTokenizerFast.from_pretrained(name, local_files_only=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            tok.padding_side = "left"
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            try:
                model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
            except Exception:
                k2 = dict(kw)
                k2.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(name, **k2).eval()
        except Exception as e:
            log(f"  {name} 로드 실패: {type(e).__name__} {str(e)[:120]}")
            continue
        log(f"  {name} 로드 {time.time() - t:.1f}s")
        _m["tok"], _m["model"], _m["id"] = tok, model, want
        return tok, model
    _m["failed"].add(want)
    return None, None


def chat(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}],
                                tokenize=False, add_generation_prompt=True)
    return s + FORCE


def score_continuation(tok, model, prompt, cont):
    """logp(cont | prompt) 의 토큰 평균. teacher forcing forward 1회."""
    import torch
    pi = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ci = tok(cont, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ci.shape[1] == 0:
        return None
    ids = torch.cat([pi, ci], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, pi.shape[1] - 1:-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    tgt = ci[0].to(lp.device)
    return float(lp[torch.arange(len(tgt)), tgt].mean())


def forced_answer(tok, model, text):
    """즉답 8토큰 greedy + 그 구간 엔트로피 평균. 반환 (답 문자열, 엔트로피)."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class Ent(LogitsProcessor):                            # 스텝마다 스칼라로 줄인다(로짓을 쌓지 않는다)
        def __init__(self):
            self.v = []

        def __call__(self, input_ids, scores):
            lp = torch.log_softmax(scores.float(), dim=-1)
            self.v.append(float((-(lp.exp() * lp).sum(-1)).mean()))
            return scores

    p = chat(tok, text)
    enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
    e = Ent()
    with torch.no_grad():
        g = model.generate(**enc, do_sample=False, max_new_tokens=ANS_TOKENS, pad_token_id=tok.pad_token_id,
                           logits_processor=LogitsProcessorList([e]))
    raw = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    # **첫 줄만 답으로 쓴다.** 강제 종료를 뚫고 계속 떠드는 모델이 있어, 그 수다가 평균 logprob 을 지배하면
    # '자기 답에 대한 확신' 이 아니라 '수다의 확신' 을 재게 된다(실측으로 logp 가 -1.2 → -3.7~-9.8 로 벌어졌다).
    ans = raw.splitlines()[0].strip() if raw.strip() else ""
    return ans, (sum(e.v) / len(e.v) if e.v else None)


def rank_split(values, higher_is_robust=True):
    """호출 내 중앙값 분할: 좋은 절반을 True. 동점은 입력 순서가 앞인 쪽을 True. 1개면 True."""
    n = len(values)
    if n == 1:
        return [True]
    order = sorted(range(n), key=lambda i: (values[i] if higher_is_robust else -values[i], -i))
    out = [True] * n
    for i in order[:n // 2]:                               # 나쁜 쪽 절반 = False
        out[i] = False
    return out


def two_tier(n, drops, ents):
    """drop 을 구한 층은 drop 으로, 못 구한 층은 엔트로피로 — **각 층을 따로** 중앙값 분할한다.
    두 값을 한 점수로 합치지 않는다: 단위가 달라 순위를 이어 붙이면 층 간 순서가 뒤섞인다
    (`analysis/h20e_coverage.py` 에서 계측해 이 방식으로 고정했다)."""
    out = [True] * n
    a = [i for i in range(n) if drops[i] is not None]
    b = [i for i in range(n) if drops[i] is None and ents[i] is not None]
    for idx, vals, hi in ((a, [drops[i] for i in a], True),          # drop 큼 = robust
                          (b, [ents[i] for i in b], False)):         # 엔트로피 작음 = robust
        if not idx:
            continue
        for i, ok in zip(idx, rank_split([float(v) for v in vals], higher_is_robust=hi)):
            out[i] = ok
    return out


def decide(problems, model_id="", effort="default"):
    cache = jload(CACHE, {})
    state = jload(STATE, {"calls": 0})
    state["calls"] = int(state.get("calls", 0)) + 1
    jsave(STATE, state)
    budget = max(0.0, remaining()) / max(1, EXPECTED_CALLS - (state["calls"] - 1))
    n = len(problems)
    drops, ents, n_var = [None] * n, [None] * n, [0] * n
    todo = []
    for i, p in enumerate(problems):
        c = cache.get(key_of(p, model_id, effort))
        if c:
            drops[i], ents[i], n_var[i] = c.get("drop"), c.get("ent"), c.get("nv", 0)
        else:
            todo.append(i)
    log(f"호출 {state['calls']}/{EXPECTED_CALLS} · {str(model_id).split('/')[-1]} · effort {effort} · "
        f"문제 {n} · 캐시 {n - len(todo)} · 남은 {remaining():.0f}s · 호출예산 {budget:.0f}s")
    if todo:
        tok = model = None
        try:
            t_load = time.time()
            tok, model = get_model(model_id)
            if time.time() - t_load > 1:
                budget = max(0.0, budget - (time.time() - t_load))
                log(f"  로드에 {time.time() - t_load:.0f}s → 남은 호출예산 {budget:.0f}s")
        except Exception as e:
            log(f"  로드 실패: {type(e).__name__} {str(e)[:120]}")
        if model is None:
            log(f"{model_id} 를 열 수 없음 → 전부 True")
        else:
            end = time.time() + max(5.0, min(budget, remaining()) - RESERVE)
            done = 0
            for i in todo:
                if time.time() > end:
                    log(f"  예산 소진 → 남은 {len(todo) - done}문제는 측정 생략")
                    break
                try:
                    ans, ent = forced_answer(tok, model, problems[i])
                    ents[i] = ent
                    d = None
                    if ans:
                        lp0 = score_continuation(tok, model, chat(tok, problems[i]), ans)
                        vs = make_variants(problems[i])
                        if lp0 is not None and vs:
                            ds = []
                            for v in vs:                 # 변형 K개 각각의 drop → 평균(사전등록 주지표)
                                lp1 = score_continuation(tok, model, chat(tok, v), ans)
                                if lp1 is not None:
                                    ds.append(lp0 - lp1)
                            if ds:
                                d = sum(ds) / len(ds)
                                n_var[i] = len(ds)
                    drops[i] = d
                    cache[key_of(problems[i], model_id, effort)] = {"drop": d, "ent": ent, "nv": n_var[i]}
                except Exception as e:
                    log(f"  문제 {i} 측정 실패: {type(e).__name__} {str(e)[:100]}")
                done += 1
            jsave(CACHE, cache)
    n_drop = sum(1 for d in drops if d is not None)
    n_ent = sum(1 for i in range(n) if drops[i] is None and ents[i] is not None)
    out = two_tier(n, drops, ents)
    dv = [d for d in drops if d is not None]
    log(f"판정: drop {n_drop}문제(변형 평균 {sum(n_var) / max(1, n_drop):.1f}개)" + (f"({min(dv):+.3f}~{max(dv):+.3f})" if dv else "")
        + f" + 엔트로피 {n_ent}문제 + 기본 True {n - n_drop - n_ent} · True {sum(out)}/{n}")
    return [bool(o) for o in out]
