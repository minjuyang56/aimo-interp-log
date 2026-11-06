"""H21c — **그 모델 자신의 답이 상수 변형에서 떨어지는 정도**로 판정. 사전등록 2026-10-06 (지휘 #65/#67/#68).

무엇을 재는가
  문제 뒤에 `</think>\nANSWER:` 를 강제해 그 모델의 **즉답**을 8토큰 받는다. 그 다음 문제의 상수 하나만
  결정적 규칙으로 바꾼 변형을 만들고, **같은 즉답 문자열**이 원본·변형에서 받는 평균 logprob 을 재서
      drop = logp(즉답 | 원본) − logp(즉답 | 상수변형)
  **부호 고정: drop 이 클수록 robust.** 상수가 바뀌면 답도 바뀌어야 하므로 진짜 푸는 모델은 옛 답을 버리고,
  암기·운으로 맞춘 모델은 상수와 무관하게 같은 답을 계속 높게 본다.

측정 근거 (로컬 4bit, 공식 train-main-v2 문제 단위 라벨 33, permutation null 1000)
  drop AUC: DeepSeek **0.802 ▲**(길이잔차 0.688) · Skywork **0.741 ▲**(0.672) · Qwen 0.636 · Olmo 0.466
  같이 시험하고 **탈락시킨 것들**: H21b 즉답 logprob(모델 간 방향 갈림) · H21a 1024토큰 엔트로피(4모델 방향은
  같지만 전부 null 안, 0.608~0.658) · H21p = drop − d_irrel(민감도 프로파일 결합 — d_irrel 의 부호가 가설과
  반대라 결합이 단독보다 나빴다: Skywork 0.741 → 0.640).

왜 이 설계인가 (앞선 제출들에서 배운 것)
  * 문제 단위로 같은 답을 뿌리면 구조적으로 ~0.50 이다(검증셋이 모델별로 라벨을 섞어 놓았다).
    그래서 **호출마다 그 model_id 자신을 로드**해 그 모델의 답·확률로만 판정한다.
  * h20 계열(생성 길이)은 실전에서 4모델 중 2모델이 토큰 상한에 포화해 측정이 성립하지 않았다(h20e 0.43).
    여기서는 생성이 **8토큰**뿐이라 포화가 원리상 없다.
  * 절대 임계값은 쓰지 않는다(최종셋이 균형이 아닐 수 있다는 공지). **호출 내 중앙값 분할**로 간다.

판정
  1. drop 을 구한 문제들: 호출 내에서 **drop 상위 절반 = True(robust)**.
  2. 상수 변형을 못 만든 문제(학습셋에서 1/33): 그 층만 따로 **엔트로피 낮은 절반 = True**.
     (즉답 8토큰 구간의 엔트로피다. 검증된 H21a 는 1024토큰 버전이므로 이건 **싼 대리**이고,
      주지표를 못 구한 소수 문제에만 쓴다.)
  3. 둘 다 없으면 True. 예외는 전부 True 로 떨어뜨린다(무효보다 낫다).
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
CACHE = Path(os.environ.get("H21_CACHE", "/tmp/h21c_cache.json"))
STATE = Path(os.environ.get("H21_STATE", "/tmp/h21c_state.json"))
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
FORCE = "</think>\nANSWER:"
NUM_RE = re.compile(r"(?<![\w.])([2-9]|[1-9]\d{1,3})(?![\w.])")
_m = {"model": None, "tok": None, "id": None, "failed": set()}


def log(msg):
    print(f"[h21c {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


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


def perturb_constant(text):
    """상수 하나만 결정적으로 바꾼다(주최측 섭동 성격: 풀이 보존·상수만 변경). 못 바꾸면 None.
    규칙: 마지막에 나오는 2~9 또는 2~4자리 수를 +1. 0·1 과 5자리 이상은 건드리지 않는다
    (0/1 은 수식 구조를 바꾸기 쉽고, 긴 수는 연도·식별자일 확률이 높다). 난수 없음 = 재현된다.

    **이 규칙은 측정 때와 글자 하나까지 같게 유지한다.** 지수를 건드리는 경우도 있다
    ($x^2+y^2$ → $x^2+y^3$ 처럼). 주최측 섭동($7^{2024}$ → $7^{2025}$ 류)보다 거친 편집이지만,
    지표가 요구하는 것은 '답이 바뀌는 편집' 이고, 규칙을 더 곱게 다듬으면
    AUC 0.802/0.741 을 낸 그 방법이 아니게 된다. 개선은 새 측정과 함께만 한다."""
    ms = list(NUM_RE.finditer(text))
    if not ms:
        return None
    m = ms[-1]
    out = text[:m.start()] + str(int(m.group(1)) + 1) + text[m.end():]
    return out if out != text else None


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
    drops, ents = [None] * n, [None] * n
    todo = []
    for i, p in enumerate(problems):
        c = cache.get(key_of(p, model_id, effort))
        if c:
            drops[i], ents[i] = c.get("drop"), c.get("ent")
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
                        pert = perturb_constant(problems[i])
                        if lp0 is not None and pert:
                            lp1 = score_continuation(tok, model, chat(tok, pert), ans)
                            if lp1 is not None:
                                d = lp0 - lp1
                    drops[i] = d
                    cache[key_of(problems[i], model_id, effort)] = {"drop": d, "ent": ent}
                except Exception as e:
                    log(f"  문제 {i} 측정 실패: {type(e).__name__} {str(e)[:100]}")
                done += 1
            jsave(CACHE, cache)
    n_drop = sum(1 for d in drops if d is not None)
    n_ent = sum(1 for i in range(n) if drops[i] is None and ents[i] is not None)
    out = two_tier(n, drops, ents)
    dv = [d for d in drops if d is not None]
    log(f"판정: drop {n_drop}문제" + (f"({min(dv):+.3f}~{max(dv):+.3f})" if dv else "")
        + f" + 엔트로피 {n_ent}문제 + 기본 True {n - n_drop - n_ent} · True {sum(out)}/{n}")
    return [bool(o) for o in out]
