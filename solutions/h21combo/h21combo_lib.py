"""H21combo — drop(상수 변형에서 자기 답 하락) + 1024토큰 구간 엔트로피의 결합. (지휘 #68 2순위)

**이 조합은 사후(post-hoc)다.** H21a~d 의 모델별 결과를 본 뒤에 "drop 이 제일 세고 엔트로피는 4모델 방향이
같다" 를 보고 만들었다. 그래서 측정된 AUC 는 가설 검정이 아니라 **설계 후보 비교**이고, 이 제출물의 리더보드
점수가 이 조합의 **첫 확인 데이터**가 된다. 리포트에 그대로 적는다.

combo = z(drop) + z(−mean_entropy) · 모델별(=호출 내) 표준화 · **가중치 학습 없음** · 큼 = robust
  drop         = logp(즉답|원본) − logp(즉답|상수변형)         [H21c 주지표, 큼 = robust]
  mean_entropy = 원본 풀이 1024토큰 구간의 평균 토큰 엔트로피   [H21a 주지표, 큼 = spurious]

측정값 (로컬 4bit, 공식 33문제, null 1000 · `analysis/h21_combo.py`)
  combo AUC: DeepSeek **0.796 ▲**(길이잔차 0.750) · Skywork **0.742 ▲**(0.696) · Qwen 0.696 · Olmo 0.531
  drop 단독:        0.802 ▲ / 0.741 ▲ / 0.636 / 0.466
  → 두 모델에서는 사실상 같고(0.796 vs 0.802 · 0.742 vs 0.741), **Qwen·Olmo 에서 올라간다**(0.636→0.696 · 0.466→0.531).
    즉 combo 는 약한 모델을 끌어올리는 쪽이다. 대가는 1024토큰 생성 비용(모델당 약 54s)이다.
  부수 관측: combo 의 모델 간 순위상관 0.186~0.634(중앙값 ~0.43) — H20 길이 계열(0.713)보다 훨씬 낮고
  포화 탓이 아니다 → 길이보다 덜 문제-지배적이다(= 모델×문제 성분에 더 가깝다).

판정 (절대 임계값 없음 — 최종셋이 균형이 아닐 수 있다는 공지)
  1. drop·엔트로피 **둘 다** 있는 문제: 호출 내에서 z(drop)+z(−ent) **상위 절반 = True**
  2. drop 만: 그 층만 따로 drop 상위 절반 = True
  3. 엔트로피만: 그 층만 따로 엔트로피 낮은 절반 = True
  4. 아무것도 없으면 True. 예외도 전부 True.
  층을 하나의 점수로 합치지 않는다: 단위가 달라 순위를 이어 붙이면 층 간 순서가 뒤섞인다
  (`analysis/h20e_coverage.py` 에서 계측해 이 방식으로 고정했다).

시간: 엔트로피 생성이 비용의 거의 전부다. 예산이 모자라면 **엔트로피를 건너뛰고 drop 만으로** 판정한다
(그 경우 h21c 와 같은 동작). OOM 이면 배치를 반으로 줄여 재시도한다.
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
ENT_TOKENS = int(os.environ.get("H21_ENT_TOKENS", 1024))  # 엔트로피 구간 길이(측정과 동일)
ENT_BATCH = int(os.environ.get("H21_ENT_BATCH", 16))      # OOM 이면 반으로 줄인다
ENT_MIN_S = float(os.environ.get("H21_ENT_MIN_S", 90))    # 이보다 예산이 적으면 엔트로피를 포기하고 drop 만 쓴다        # 호출 끝에 남겨 두는 초
CACHE = Path(os.environ.get("H21_CACHE", "/tmp/h21combo_cache.json"))
STATE = Path(os.environ.get("H21_STATE", "/tmp/h21combo_state.json"))
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
    print(f"[h21combo {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


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


def batch_entropy(tok, model, texts, deadline):
    """원본 문제들을 1024토큰 greedy 로 생성하며 **스텝마다 엔트로피 스칼라만** 모은다.
    로짓을 쌓지 않는 이유: 1024스텝 × 배치 × 어휘 152k 를 들고 있으면 수 GB 다(측정 때 실제로 터졌다).
    반환: texts 와 같은 길이의 리스트(실패분은 None)."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    class Ent(LogitsProcessor):
        def __init__(self):
            self.rows = []

        def __call__(self, input_ids, scores):
            lp = torch.log_softmax(scores.float(), dim=-1)
            self.rows.append((-(lp.exp() * lp).sum(-1)).cpu())
            return scores

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return time.time() > deadline

    out = [None] * len(texts)
    bs = max(1, ENT_BATCH)
    i = 0
    while i < len(texts):
        if time.time() > deadline:
            log(f"  엔트로피: 마감 → {len(texts) - i}문제 생략")
            break
        chunk = list(range(i, min(i + bs, len(texts))))
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{texts[j]}"}],
            tokenize=False, add_generation_prompt=True) for j in chunk]
        try:
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            e = Ent()
            t0 = time.time()
            with torch.no_grad():
                g = model.generate(**enc, do_sample=False, max_new_tokens=ENT_TOKENS,
                                   pad_token_id=tok.pad_token_id,
                                   logits_processor=LogitsProcessorList([e]),
                                   stopping_criteria=StoppingCriteriaList([Deadline()]))
            ent = torch.stack(e.rows, dim=1) if e.rows else None
            new = g[:, enc["input_ids"].shape[1]:]
            for r, j in enumerate(chunk):
                if ent is None:
                    continue
                live = (new[r] != tok.pad_token_id).cpu()[:ent.shape[1]]
                if live.sum() == 0:
                    live[:] = True
                out[j] = float(ent[r][live].mean())
            log(f"  엔트로피 {chunk[-1] + 1}/{len(texts)} · 배치 {len(chunk)} · 스텝 {0 if ent is None else ent.shape[1]} "
                f"· {time.time() - t0:.0f}s")
            i += len(chunk)
            del e, ent
            torch.cuda.empty_cache()
        except Exception as ex:
            name = type(ex).__name__
            torch.cuda.empty_cache()
            if "OutOfMemory" in name or "CUDA" in str(ex)[:120]:
                if bs > 1:
                    bs = max(1, bs // 2)
                    log(f"  엔트로피 OOM → 배치 {bs} 로 줄여 재시도")
                    continue
            log(f"  엔트로피 실패: {name} {str(ex)[:100]} → 이 배치 포기")
            i += len(chunk)
    return out


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


def z(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return [(v - m) / (sd + 1e-9) for v in vals]


def tiers(n, drops, ent1024, ent8):
    """4층 분할, 각 층을 따로 중앙값 분할한다. 층을 한 점수로 합치지 않는다(단위가 달라 순서가 뒤섞인다).
      1층 combo   = drop 과 **1024토큰 엔트로피** 둘 다 있는 문제 → z(drop)+z(−ent) 상위 절반 True
      2층 drop만  = drop 상위 절반 True
      3층 ent1024 = 엔트로피 낮은 절반 True
      4층 ent8    = 즉답 8토큰 엔트로피(싼 대리) 낮은 절반 True — 1024 를 못 구한 문제의 마지막 수단
      그 외 True."""
    out = [True] * n
    both = [i for i in range(n) if drops[i] is not None and ent1024[i] is not None]
    only_d = [i for i in range(n) if drops[i] is not None and ent1024[i] is None]
    only_e = [i for i in range(n) if drops[i] is None and ent1024[i] is not None]
    rest_e8 = [i for i in range(n) if drops[i] is None and ent1024[i] is None and ent8[i] is not None]
    if both:
        combo = [a + b for a, b in zip(z([float(drops[i]) for i in both]),
                                       z([-float(ent1024[i]) for i in both]))]
        for i, ok in zip(both, rank_split(combo, higher_is_robust=True)):
            out[i] = ok
    for idx, vals, hi in ((only_d, [float(drops[i]) for i in only_d], True),
                          (only_e, [float(ent1024[i]) for i in only_e], False),
                          (rest_e8, [float(ent8[i]) for i in rest_e8], False)):
        if idx:
            for i, ok in zip(idx, rank_split(vals, higher_is_robust=hi)):
                out[i] = ok
    return out, len(both), len(only_d), len(only_e) + len(rest_e8)


def decide(problems, model_id="", effort="default"):
    cache = jload(CACHE, {})
    state = jload(STATE, {"calls": 0})
    state["calls"] = int(state.get("calls", 0)) + 1
    jsave(STATE, state)
    budget = max(0.0, remaining()) / max(1, EXPECTED_CALLS - (state["calls"] - 1))
    n = len(problems)
    drops, ent1024, ent8 = [None] * n, [None] * n, [None] * n
    todo = []
    for i, p in enumerate(problems):
        c = cache.get(key_of(p, model_id, effort))
        if c:
            drops[i], ent1024[i], ent8[i] = c.get("drop"), c.get("ent1024"), c.get("ent8")
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
            call_end = time.time() + max(5.0, min(budget, remaining()) - RESERVE)   # 한 번만 정한다
            done = 0
            for i in todo:                              # 1) drop (문제당 forward 3회 수준, 싸다)
                if time.time() > call_end:
                    log(f"  예산 소진 → 남은 {len(todo) - done}문제 drop 생략")
                    break
                try:
                    ans, e8 = forced_answer(tok, model, problems[i])
                    ent8[i] = e8
                    d = None
                    if ans:
                        lp0 = score_continuation(tok, model, chat(tok, problems[i]), ans)
                        pert = perturb_constant(problems[i])
                        if lp0 is not None and pert:
                            lp1 = score_continuation(tok, model, chat(tok, pert), ans)
                            if lp1 is not None:
                                d = lp0 - lp1
                    drops[i] = d
                    cache[key_of(problems[i], model_id, effort)] = {"drop": d, "ent8": e8}
                except Exception as e:
                    log(f"  문제 {i} 측정 실패: {type(e).__name__} {str(e)[:100]}")
                done += 1
            jsave(CACHE, cache)
            # 2) 1024토큰 엔트로피 — 비용의 거의 전부. 남은 예산이 모자라면 포기하고 drop 만 쓴다(= h21c 동작).
            need = [i for i in todo if ent1024[i] is None]
            left = call_end - time.time()
            if need and left >= ENT_MIN_S:
                vals = batch_entropy(tok, model, [problems[i] for i in need], time.time() + left)
                for i, v in zip(need, vals):
                    if v is None:
                        continue
                    ent1024[i] = v
                    c = cache.get(key_of(problems[i], model_id, effort), {})
                    c["ent1024"] = v
                    cache[key_of(problems[i], model_id, effort)] = c
                jsave(CACHE, cache)
            elif need:
                log(f"  엔트로피 생략(남은 {left:.0f}s < {ENT_MIN_S:.0f}s) → drop 만으로 판정")
    out, n_both, n_d, n_e = tiers(n, drops, ent1024, ent8)
    dv = [d for d in drops if d is not None]
    log(f"판정: combo {n_both}문제 + drop만 {n_d} + 엔트로피만 {n_e} + 기본 True {n - n_both - n_d - n_e}"
        + (f" · drop {min(dv):+.3f}~{max(dv):+.3f}" if dv else "") + f" · True {sum(out)}/{n}")
    return [bool(o) for o in out]
