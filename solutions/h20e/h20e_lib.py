"""H20e — 모델별 자기 생성 + 호출 내 상대 순위 판정. 사전등록 2026-10-04 (지휘 #62).

왜 바꿨나 (리더보드 0.50/0.57 의 구조적 원인)
  주최측 공지(9/15~16): 검증셋/리더보드는 **문제별·모델별 라벨 균형**으로 재구성됐다. 같은 문제가 모델에 따라
  robust 도 되고 spurious 도 된다. 그러면 **문제 단위로 모델 간 같은 답을 뿌리는 방식은 구조적으로 ~50%** 다.
  h15(텍스트 규칙)·h20/h20d(문제 해시 교차모델 캐시)가 전부 거기 걸렸다 — 0.50 · 0.50 · 0.57 · 0.57 이 그것으로 설명된다.
  (학습셋 train-main-v2 자체는 여전히 문제 단위다: 2개 이상 모델에 걸친 15문제 중 **모델 간 라벨이 갈리는 문제 0개**,
   유일한 1건은 gpt-oss-120b 가 reasoning_effort 에 따라 갈린 경우다. 즉 **모델별 차이는 학습셋에서 배울 수 없고**,
   검증셋에만 존재한다 — 그래서 모델별 절대 임계값을 학습할 수 없고 **호출 내 상대 순위**로 간다.)

바뀐 점
  1. **판정 대상 모델 자신으로 생성한다.** h20d 는 고정 계측기(Skywork) 하나로 모든 model_id 를 재고 문제 해시로
     캐시를 공유했다 — 그게 모델별 차이를 0 으로 만들었다. 이제 호출마다 `model_id` 를 열고(앞 모델은 해제),
     캐시는 (model_id, effort, 문제) 단위로만 재사용한다. 모델을 못 열면 **다른 모델로 대신 재지 않고** 텍스트 규칙으로 간다.
  2. **판정은 호출 내 중앙값 분할.** 길이 상위 절반 = spurious(False). 모델별 라벨이 균형이라는 공지에 맞춘 설계이고,
     절대 임계값(train 고정)은 모델별 분포 차이에 취약해 쓰지 않는다. 동점은 입력 순서가 앞인 쪽을 robust 로.
  3. **reasoning_effort 를 생성 설정에 반영**한다(low → 짧은 cap, high → 긴 cap 쪽으로 한 칸).
  4. 폴백도 상대 순위로: 텍스트 규칙 n_sym 의 **호출 내 중앙값 분할**.
  5. **조기 종료된 배치도 쓴다.** h20d 는 cap 별 고정 임계값을 썼기 때문에 마감 임박으로 cap 전에 끊기면
     보정이 안 맞아 배치를 버렸다. 상대 순위는 보정이 필요 없고 한 배치의 모든 시퀀스가 같은 스텝에서
     끊기므로 길이 비교가 그대로 성립한다. 버리는 조건은 **순위를 만들 수 없을 때**로만 좁혔다
     (시퀀스 90% 이상이 절단되고 평균 길이 퍼짐이 스텝의 2% 미만, 또는 서로 다른 값이 1개).
     그래서 cap 별 train 임계값(artifacts/h20_grid.json)은 더 이상 읽지 않는다.
  6. **VRAM 도 같이 본다(모델별 로드가 만든 새 제약).** KV 캐시는 토큰당 2x층x kv헤드 x head_dim x 2바이트인데
     Skywork 56 KB · Qwen3.5 128 · DeepSeek 144 · **Olmo-3-7B-Think 512**(kv헤드 32 = MHA)로 9배 차이가 난다.
     Olmo 를 배치 70 x cap 4096 으로 돌리면 KV 만 140 GB 라 96 GB 워커에서 터진다(h20d 는 Skywork 만 열었으니
     이 문제가 없었다). 그래서 로드 후 실제 여유 VRAM 으로 '배치 x cap' 칸 수를 재고, 시간과 VRAM 을
     둘 다 만족하는 가장 큰 cap 을 고른 뒤 모자라면 **그룹으로 쪼갠다**. 그룹마다 끊긴 스텝이 다를 수 있으므로
     모든 길이를 **min(스텝)으로 잘라** 비교 가능하게 만든다(analysis/h20_grid.py 의 cap 시뮬과 같은 방식).
     예상 시간은 `analysis/h20e_time.py` (모델별 KV 를 넣어 계산).
  나머지(k=2, ms(B,L) 모델로 cap 결정, StoppingCriteria 조기 종료, 호출당 예산, 로드시간 차감)는 h20d 그대로.
주의: 호출 내 문제가 1개뿐이면 중앙값 분할이 의미가 없다 — 그 경우 robust(True) 로 둔다(상수 베이스라인과 동률).
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
KV_SAFE = float(os.environ.get("H20_KV_SAFE", 0.55))   # 여유 VRAM 중 KV 캐시에 허용할 몫 (활성값·단편화 몫을 남긴다)
K_RICH = int(os.environ.get("H20_K_RICH", 3))
N_FEW = int(os.environ.get("H20_N_FEW", 6))
ART = Path(__file__).resolve().parent / "artifacts" / "h20_grid.json"
EFFORT_MULT = {"low": 0.8, "default": 1.0, "medium": 1.0, "high": 1.1}
CACHE = Path(os.environ.get("H20_CACHE", "/tmp/h20e_cache.json"))
STATE = Path(os.environ.get("H20_STATE", "/tmp/h20e_state.json"))
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
_m = {"model": None, "tok": None, "id": None, "failed": set()}


def log(msg):
    print(f"[h20e {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


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


def max_cap(batch, budget, scale=1.0, effort="default"):
    """시간만 보는 단순 선택기(그룹 분할 없음). 시나리오 추정과 단위 점검에 쓴다."""
    ok = [c for c in CAPS if seconds_for(batch, c, scale) <= budget]
    if not ok:
        return None
    cap = max(ok)
    i = CAPS.index(cap)
    if effort == "low" and i > 0:
        cap = CAPS[i - 1]
    elif effort in ("high",) and i + 1 < len(CAPS) and seconds_for(batch, CAPS[i + 1], scale) <= budget * 1.1:
        cap = CAPS[i + 1]
    return cap


def kv_kb_per_token(model):
    """KV 캐시의 토큰당 KB. Olmo-3 는 MHA(kv헤드 32)라 Skywork 의 9배다 — 배치×cap 한도가 모델마다 다르다."""
    cfg = getattr(model, "config", None)
    t = getattr(cfg, "text_config", None) or cfg
    L = getattr(t, "num_hidden_layers", None)
    kv = getattr(t, "num_key_value_heads", None) or getattr(t, "num_attention_heads", None)
    hd = getattr(t, "head_dim", None)
    if not hd:
        hs, nh = getattr(t, "hidden_size", 0), getattr(t, "num_attention_heads", 0)
        hd = hs // nh if (hs and nh) else 0
    if not (L and kv and hd):
        return 512.0                           # 못 읽으면 가장 나쁜 쪽(MHA)으로 가정한다
    return 2 * L * kv * hd * 2 / 1024


def token_slots(model):
    """'배치 x cap' 으로 쓸 수 있는 토큰 칸 수. 측정 실패 시 None = 제한 없음."""
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return None
    return max(1, int(free / 1024 * KV_SAFE / kv_kb_per_token(model)))


def grid_auc(k, cap):
    """train 격자(H14 저장 데이터)의 H20a AUC. 없으면 0 — 선택에서 밀린다."""
    try:
        return float(jload(ART, {})["fit"][f"H20a_k{k}_c{cap}"]["auc"])
    except Exception:
        return 0.0


def plan(n, budget, scale, effort, slots, k_pref):
    """(cap, k, 그룹당 문제 수) — 시간·VRAM 을 만족하는 후보 중 **train AUC 가 가장 높은** 설정.

    '가장 큰 cap' 이 아니다: 격자를 보면 cap 4096 은 k1·k2 모두 임계값이 4096(= 전부 절단) · True 비율 1.00 ·
    AUC 0.600/0.625 로 정보가 없다. 문제가 30개를 넘으면 예산상 그 구간만 남는데, 거기서 긴 생성을 해도
    길이가 전부 상한에 붙어 순위가 생기지 않는다. 그래서 같은 예산 안에서 **train AUC 최대**를 고른다
    (리더보드 점수가 아니라 train 격자 값이다). 동점이면 더 큰 cap."""
    budget = budget * EFFORT_MULT.get(effort, 1.0)

    def groups_time(n_, cap, k, g):
        ngr = -(-n_ // g)
        return sum(seconds_for(min(g, n_ - i * g) * k, cap, scale) for i in range(ngr))

    def fit(cap, k):
        return n if slots is None else max(1, min(n, int(slots // (cap * k))))

    cands = []
    for k in {k_pref, 2, 1}:
        if k < 1:
            continue
        for cap in CAPS:
            g = fit(cap, k)
            if groups_time(n, cap, k, g) <= budget:
                cands.append((grid_auc(k, cap), cap, k, g))
    if not cands:
        return None
    _, cap, k, g = max(cands)
    return cap, k, g


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


def rank_split(values, higher_is_spurious=True):
    """호출 내 중앙값 분할: 상위 절반을 False 로. 동점은 입력 순서가 앞인 쪽을 True.
    문제가 1개뿐이면 분할이 무의미하므로 True."""
    n = len(values)
    if n == 1:
        return [True]
    order = sorted(range(n), key=lambda i: (-values[i] if higher_is_spurious else values[i], -i))
    n_false = n // 2
    out = [True] * n
    for i in order[:n_false]:
        out[i] = False
    return out


def fallback(problems):
    """텍스트 규칙도 절대 임계값이 아니라 호출 내 상대 순위로."""
    return rank_split([float(n_sym(p)) for p in problems], higher_is_spurious=True)


def key_of(t, model_id="", effort=""):
    """캐시 키에 **model_id 와 effort 를 넣는다** — 교차모델 재사용 금지(모델별 라벨이 다르므로)."""
    raw = f"{model_id}|{effort}|{t.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def get_model(model_id):
    """**판정 대상 모델 자신**을 연다 (지휘 #62). 고정 계측기 하나로 모든 model_id 를 재던 h20d 와 다르다:
    모델별 라벨이 다른 검증셋에서 교차모델 대리는 구조적으로 ~50% 다.
    같은 모델이 다시 오면 재사용하고, 다른 모델이 오면 앞 모델을 버려 VRAM 을 반납한다.
    못 열면 (None, None) — 호출자는 텍스트 규칙 폴백으로 간다. 다른 모델로 대신 재지 않는다."""
    want = str(model_id or "").strip()
    if not want or want in _m["failed"]:
        return None, None
    if _m["id"] == want and _m["model"] is not None:
        return _m["tok"], _m["model"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    if _m["model"] is not None:                # 호출마다 모델이 바뀐다 → 앞 모델을 먼저 놓아준다
        log(f"  앞 모델 {_m['id']} 해제")
        _m["model"], _m["tok"], _m["id"] = None, None, None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto", dtype=torch.bfloat16,
              attn_implementation="sdpa")
    if os.environ.get("H20_QUANT") == "4bit":
        from transformers import BitsAndBytesConfig
        kw.pop("dtype")
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    # 별칭이 붙어 올 수도 있으니(옛 val-sample 의 'qwen3-8b:low') 접미사를 떼고 한 번 더 시도한다.
    cands = [want] + ([want.split(":")[0]] if ":" in want else [])
    for name in cands:
        t = time.time()
        try:
            try:                               # DeepSeek 은 AutoTokenizer 가 옛 LlamaTokenizer 로 열어 인코딩이 달라진다
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


def gen(texts, k, cap, deadline, model_id):
    """한 배치 생성. 마감이 지나면 StoppingCriteria 로 조기 종료. 반환: (토큰수, 실제 스텝, 초)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    tok, model = get_model(model_id)
    if model is None:
        raise RuntimeError("model unavailable")

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


def decide(problems, model_id="", effort="default"):
    cache = jload(CACHE, {})
    state = jload(STATE, {"calls": 0, "scale": 1.0})
    state["calls"] = int(state.get("calls", 0)) + 1
    scale = float(state.get("scale", 1.0))
    jsave(STATE, state)
    budget = max(0.0, remaining()) / max(1, EXPECTED_CALLS - (state["calls"] - 1))
    vals = [None] * len(problems)
    hits = {}                                  # i -> (샘플별 토큰, 그때의 상한)
    for i, p in enumerate(problems):
        c = cache.get(key_of(p, model_id, effort))
        if c and c.get("toks"):
            hits[i] = ([int(t) for t in c["toks"]], int(c.get("steps", max(c["toks"]))))
    todo = [i for i in range(len(problems)) if i not in hits]
    log(f"호출 {state['calls']}/{EXPECTED_CALLS} · {model_id.split('/')[-1]} · effort {effort} · 문제 {len(problems)} · "
        f"캐시 {len(hits)} · 남은 {remaining():.0f}s · 호출예산 {budget:.0f}s · 배율 {scale:.2f}")
    if todo:
        have_model = False
        try:                                   # 로딩을 먼저 끝내고 cap 을 정한다 (로드 시간이 예산을 먹지 않도록)
            t_load = time.time()
            have_model = get_model(model_id)[1] is not None
            spent = time.time() - t_load
            if spent > 1:
                budget = max(0.0, budget - spent)
                log(f"  로드에 {spent:.0f}s → 남은 호출예산 {budget:.0f}s")
        except Exception as e:
            log(f"  모델 로드 실패: {type(e).__name__} {str(e)[:120]}")
        slots = token_slots(_m["model"]) if have_model else None
        got = plan(len(todo), min(budget, remaining()), scale, effort, slots,
                   K_RICH if (len(todo) <= N_FEW and budget > 900) else K_DEFAULT) if have_model else None
        if not have_model:
            log(f"{model_id} 를 열 수 없음 → 다른 모델로 대신 재지 않고 텍스트 규칙 폴백")
        elif got is None:
            log(f"문제 {len(todo)} 로는 최소 cap({CAPS[0]})도 예산 초과 → 전부 폴백")
        else:
            cap, k, g = got
            groups = [todo[i:i + g] for i in range(0, len(todo), g)]
            kvkb = kv_kb_per_token(_m["model"])
            est = sum(seconds_for(len(gr) * k, cap, scale) for gr in groups)
            log(f"  {len(todo)}문제 × k{k} · cap {cap} · 그룹 {len(groups)}개(그룹당 ≤{g}문제) · 예상 {est:.0f}s · "
                f"KV {kvkb:.0f} KB/토큰 · 칸 {slots if slots else '제한없음'}")
            raw, steps_all = {}, []
            # 호출 마감을 **한 번만** 정한다. 그룹마다 다시 재면 앞 그룹이 쓴 시간이 빠지지 않아
            # 뒷 그룹이 2배 긴 share 를 받고, 끊긴 지점이 달라져 길이 비교가 깨진다(로컬 스모크에서 실제로 났다).
            call_end = time.time() + max(5.0, min(budget, remaining()) - 30)
            cap_g = cap
            try:
                for gi, gr in enumerate(groups):
                    share = (call_end - time.time()) / (len(groups) - gi)
                    deadline = time.time() + max(5.0, share)
                    toks, steps, el = gen([problems[j] for j in gr], k, cap_g, deadline, model_id)
                    steps_all.append(steps)
                    # 첫 그룹이 상한 전에 끊겼으면 이후 그룹도 **같은 스텝 수**로 맞춘다 → 길이가 비교 가능해진다.
                    if steps < cap_g:
                        cap_g = steps
                    for j, tk in zip(gr, toks):
                        raw[j] = tk
                    pred_ms = seconds_for(len(gr) * k, cap_g, 1.0) * 1000 / max(1, cap_g)
                    obs_ms = el * 1000 / max(1, steps)
                    if steps >= 256:           # 스텝이 적으면 프리필·오버헤드가 섞여 배율이 오염된다
                        scale = max(0.3, min(3.0, 0.5 * scale + 0.5 * obs_ms / max(1e-6, pred_ms)))
                        state["scale"] = scale
                        jsave(STATE, state)
                    log(f"  그룹 {gi + 1}/{len(groups)} 완료 {el:.0f}s · 스텝 {steps} · "
                        f"**{obs_ms:.1f} ms/step**(예측 {pred_ms:.1f}) · 배율 {scale:.2f}")
                if not raw:
                    raise RuntimeError("no output")
                for j, tk in raw.items():      # 새로 생성한 것도 캐시 히트와 같은 형태로 모은다
                    hits[j] = (tk, steps_all[-1] if len(steps_all) == 1 else min(steps_all))
                # 그룹·캐시마다 끊긴 스텝이 다를 수 있다 → 모두 **가장 짧은 스텝으로 잘라** 비교 가능하게 만든다.
                eff = min([st for _, st in hits.values()])
                if eff < min(256, cap):
                    raise RuntimeError(f"too few steps ({eff})")
                vv = {j: sum(min(t, eff) for t in tk) / len(tk) for j, (tk, _) in hits.items()}
                flat = [min(t, eff) for tk, _ in hits.values() for t in tk]
                trunc = sum(1 for t in flat if t >= eff - 1) / max(1, len(flat))
                means = sorted({round(v, 3) for v in vv.values()})
                spread = (means[-1] - means[0]) / max(1.0, float(eff))
                if eff < cap or len(steps_all) > 1:
                    log(f"  유효 길이 {eff}(cap {cap}, 그룹 스텝 {steps_all}) · 절단 {trunc:.0%} · 퍼짐 {spread:.1%}")
                if len(means) < 2 or (trunc > 0.9 and spread < 0.02):
                    # 거의 전부 같은 지점에서 잘렸다 → 길이로 순위를 만들 수 없다. 이 호출은 폴백.
                    log(f"  길이 퍼짐 부족(서로 다른 값 {len(means)}개, 퍼짐 {spread:.1%}) → 순위 불가, 폴백")
                    raise RuntimeError("no usable spread")
                for j, v in vv.items():
                    if j in raw:               # 캐시에는 **자르기 전 원본 토큰 수**를 남긴다(나중에 다시 자를 수 있게)
                        cache[key_of(problems[j], model_id, effort)] = {
                            "toks": raw[j], "steps": hits[j][1], "k": k}
                    vals[j] = v
                jsave(CACHE, cache)
            except Exception as e:
                log(f"  생성 실패: {type(e).__name__} {str(e)[:140]} → 폴백")
    if vals.count(None) and len(hits) == len(problems):
        # 전부 캐시 히트(생성 안 함) → 같은 상한으로 잘라 그대로 순위 판정
        eff = min(st for _, st in hits.values())
        for j, (tk, _) in hits.items():
            vals[j] = sum(min(t, eff) for t in tk) / len(tk)
        if len({round(v, 3) for v in vals}) < 2:
            vals = [None] * len(problems)      # 퍼짐 없음 → 폴백
    if all(v is not None for v in vals):
        out = rank_split([float(v) for v in vals], higher_is_spurious=True)
        log(f"호출 내 중앙값 분할 · 길이 범위 {min(vals):.0f}~{max(vals):.0f} · True {sum(out)}/{len(out)}")
    else:
        n_missing = sum(1 for v in vals if v is None)
        out = fallback(problems)
        log(f"길이를 못 구한 문제 {n_missing} → 전체를 텍스트 규칙의 호출 내 순위로 판정 · True {sum(out)}/{len(out)}")
    return [bool(o) for o in out]
