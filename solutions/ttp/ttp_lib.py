"""#TT-PERTURB — 제출물 안에서 **직접 흔들어 보는** 판정. 해석이 아니라 **행동** 기반이다.

무엇을 하는가 (한 줄): 판정 대상 모델 자신에게 원문제와 **정답 불변 섭동 K개**를 풀리고,
**짧은 CoT 뒤 답을 강제로 받아** 섭동의 답이 원문제의 답과 갈리는 비율로 robust/spurious 를 가른다.

왜 이게 예산 안에 들어가나 (h20·H22 를 죽인 벽을 넘는 지점):
  * h20 계열은 **생성 길이**를 봤기 때문에 cap 8192 이상이 필요했다 → 3600s 합산을 못 샀다.
  * H22 는 섭동을 **끝까지** 풀어야 했다(cap 24576) → 리더보드에서 텍스트 폴백만 돌았다.
  * 이번은 **답을 강제로 끊어 받는다**. 짧은 CoT 뒤 `</think>\nANSWER:` 를 박으면 cap 과 무관하게
    모든 시퀀스가 답을 내놓는다 — **절단이 측정을 못 죽인다**(h20e 를 죽인 상한 포화가 원리상 없다).
    실측 기반 예산표(`analysis/ttp_budget.py`): K=8·cap 512 가 Small 4모델x40문제에서 **911s = 28%**,
    cap 1024 도 2,176s = 66% 로 들어간다.

점수 (부호 고정 — 결과 보고 뒤집지 않는다):
      score_spurious = (답이 갈린 섭동 수) / (쓸 수 있는 섭동 수)     **높을수록 spurious**
      robust  <=>  score_spurious <= THR
THR 은 주최측 공개 라벨(train-main-v2 문제단위 33)에서 **예측 True 비율 = 그 라벨의 robust 비율**이
되도록 교정해 고정한다(#LEN-CAL 과 같은 사전확률 교정. AUC 최적화가 아니다).

주최측 기준 흉내: 라벨 정의는 '정답 불변 섭동의 과반에서 틀리면 spurious' 이고 임계는 `max_drop<=0.10`
이다(#ORG-LABEL 로 82/82 재현 확인). 우리는 테스트타임에 **정답을 모르므로** '틀렸다' 를
'원문제의 답과 달라졌다' 로 바꿔 읽는다 — 자기일관성이지 정확도가 아니다. 그 한계를 CLAUDE.md 에 적었다.

폴백: 예산이 모자라거나 모델을 못 열거나 답을 못 받으면 **h15C**(−log 단어수 · 고정 임계)로 떨어진다.
최악의 경우 h15C 와 같은 제출물이 되고 **전량 무효는 원리상 없다**.
"""

import json
import math
import os
import re
import sys
import time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts", "ttp.json")

SOFT = float(os.environ.get("TTP_SOFT", 3300))        # 전 호출 합산 안전 예산
CAP_LADDER = [1024, 512, 256, 128]                    # 예산이 허락하는 최대 cap 을 위에서부터 고른다
if os.environ.get("TTP_LADDER"):
    # **스모크 전용 훅.** 로컬 8 GB 4bit 로는 cap 512 조차 분 단위라 `decide` end-to-end 를 계측할 수
    # 없다(첫 스모크가 900s 타임아웃). 제출 시에는 설정하지 않으므로 기본 사다리가 그대로 쓰인다.
    CAP_LADDER = [int(x) for x in os.environ["TTP_LADDER"].split(",")]
ANS_STEPS = 8                                         # 강제 즉답 토큰 수 (h21c 와 같다)
K_MIN = 4                                             # 쓸 수 있는 섭동이 이보다 적으면 그 문제는 폴백
A_, B_, C_, MULT = 15.0, 0.375, 7.21e-4, 1.15         # ms(B,L) 워커 실측 적합 (±0.6 ms, 4점)
PROMPT_TOK = 600.0
KV_SAFE = float(os.environ.get("TTP_KV_SAFE", 0.55))
# 프리필 한 번에 넣는 토큰 상한. 실측(241 시퀀스 × 600 토큰 = 144k → 활성값 10.47 GiB 요구 → OOM)에서
# 역산했다(약 72 KB/토큰). 40k 토큰이면 약 2.9 GB 로 80~96 GB 에서 안전하다.
MAX_PREFILL_TOK = int(os.environ.get("TTP_MAX_PREFILL_TOK", 40000))
EFFORT_MULT = {"low": 0.8, "default": 1.0, "medium": 1.0, "high": 1.1}

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
FORCE = "\n</think>\nANSWER:"      # 짧은 CoT 뒤에 박아 답을 끌어낸다. 절단이 측정을 죽이지 못하게 하는 장치.

# ---------------------------------------------------------------- 섭동 규칙 (전부 결정적 · LLM 불필요)
# 이 블록은 `solutions/h22/h22_lib.py` 의 규칙과 **글자 그대로 같아야 한다**.
# 베껴 적어 어긋나는 사고가 이 저장소에 여러 번 있었으므로 `analysis/ttp_smoke.py` 가
# 두 구현의 출력이 전 문제에서 같은지 **수치로 검사**한다(다르면 제출물이 아니다).
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT, UPPER_SHIFT2 = 15, 7
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
LOWER_PAIRS2 = {"x": "s", "s": "x", "y": "v", "v": "y", "z": "q", "q": "z", "n": "h", "h": "n",
                "k": "g", "g": "k", "a": "e", "e": "a", "b": "d", "d": "b", "c": "f", "f": "c"}
DISTRACTORS = [
    "Note that the problem was originally posed at a regional competition in 2019.",
    "A friend of the author suggested this problem after a long hike.",
    "The committee that wrote this problem also enjoys chess on weekends.",
]
ANS_RE = re.compile(r"ANSWER\s*:\s*\$?\s*(-?[\d,]+)")
BOXED_RE = re.compile(r"\\boxed\s*\{\s*\$?\s*(-?[\d,]+)\s*\}")
_ALONE = re.compile(r"(?<![A-Za-z])([a-z])(?![A-Za-z])")
# reorder·notation (규칙 기반 '재서술' — `experiments/variants_nonofficial.py` 와 같은 규칙)
SENT_END = re.compile(r"(?<=[.!?])\s+")
FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
UNSAFE = re.compile(r"\b(the total|the result|the sum of (these|those|them)|this (number|value|result|sum|total)|"
                    r"these|those|the above|the former|the latter|it|they|them)\b", re.I)
IMPERATIVE = re.compile(r"^(find|compute|what|how|determine|calculate|evaluate|let|suppose|express|give|state|write|if)\b", re.I)
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}

_m = {"model": None, "tok": None, "id": None, "failed": set()}


def log(msg):
    print(f"[ttp {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _chunk(c, shift, pairs):
    c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + shift) % 26 + 65), c)
    return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: pairs.get(m.group(1), m.group(1)), c)


def _safe_pairs(span_text, pairs):
    """기호 충돌을 일으키지 않는 치환만 남긴다 (h22 의 `_safe_pairs` 와 같은 규칙).
    `\\frac{kc+l}{mc+n}` 에서 홀로 선 `n`→`m` 이 되면 `mc` 의 `m` 과 뭉쳐 정답이 안 보존된다."""
    alone = set(_ALONE.findall(span_text))
    any_l = set(re.findall(r"[a-z]", span_text))
    return {L: M for L, M in pairs.items() if L in alone and (M not in any_l or M in alone)}


def _rename(text, shift, pairs):
    def span(m):
        s, out, pos, segs = m.group(0), [], 0, []
        for cmd in LATEX_CMD.finditer(s):
            segs.append(s[pos:cmd.start()])
            pos = cmd.end()
        segs.append(s[pos:])
        p = _safe_pairs("".join(segs), pairs)
        pos = 0
        for cmd in LATEX_CMD.finditer(s):
            out.append(_chunk(s[pos:cmd.start()], shift, p))
            out.append(cmd.group(0))
            pos = cmd.end()
        out.append(_chunk(s[pos:], shift, p))
        return "".join(out)
    new = MATH_SPAN.sub(span, text)
    return new if new != text else None


def _typos(text, seed, k=3):
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    words = [m for m in re.finditer(r"[A-Za-z]{5,}", text)
             if not any(a <= m.start() < b for a, b in spans)]
    if not words:
        return None
    pick = sorted({(seed * 7 + i * 3) % len(words) for i in range(min(k, len(words)))}, reverse=True)
    out = list(text)
    for i in pick:
        m = words[i]
        j = m.start() + 1 + ((seed + i) % max(1, m.end() - m.start() - 3))
        if j + 1 < m.end():
            out[j], out[j + 1] = out[j + 1], out[j]
    new = "".join(out)
    return new if new != text else None


def _distract(text, idx):
    return text.rstrip() + " " + DISTRACTORS[idx % len(DISTRACTORS)]


def _protected(text):
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    spans += [(m.start(), m.end()) for m in FRAC.finditer(text)]
    return spans


def _inside(i, spans):
    return any(a <= i < b for a, b in spans)


def _num_to_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        t, o = n // 10 * 10, n % 10
        return TENS[t] + ("" if o == 0 else "-" + ONES[o])
    if n < 1000 and n % 100 == 0:
        return ONES[n // 100] + " hundred"
    return None


def _sentences(text):
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    parts, last = [], 0
    for m in SENT_END.finditer(text):
        if _inside(m.start(), spans):
            continue
        parts.append(text[last:m.start()].rstrip())
        last = m.end()
    parts.append(text[last:])
    return [p for p in parts if p.strip()]


def _reorder(text):
    """조건 문장 순서를 뒤집는다(질문 문장은 고정). 참조어·명령형·물음표가 조건에 있으면 포기 —
    "그 합을 4로 나눈다" 처럼 순서가 곧 계산인 문장이 있으면 정답이 바뀐다."""
    parts = _sentences(text)
    if len(parts) < 4:
        return None
    body, tail = parts[:-1], parts[-1]
    for p in body:
        if UNSAFE.search(p) or IMPERATIVE.match(p.strip()) or "?" in p:
            return None
    new = " ".join(list(reversed(body)) + [tail])
    return new if new != text else None


def _notation(text):
    """수식 밖 1~999 정수를 영어 단어로, `\\frac{a}{b}` → `a/b`, `\\%` → percent.
    수식 안·`\\frac` 안은 건드리지 않는다(값이 깨진다)."""
    spans = _protected(text)
    res, pos, changed = [], 0, False
    for m in re.finditer(r"(?<![\w.$\\/])(\d{1,3})(?![\w.$/])", text):
        if _inside(m.start(), spans):
            continue
        w = _num_to_words(int(m.group(1)))
        if w is None:
            continue
        res.append(text[pos:m.start()])
        res.append(w)
        pos = m.end()
        changed = True
    res.append(text[pos:])
    out = "".join(res)
    out2 = FRAC.sub(lambda m: f"{m.group(1)}/{m.group(2)}", out).replace("\\%", " percent")
    changed = changed or out2 != out
    return out2 if changed and out2 != text else None


def perturbations(text):
    """**정답 불변 섭동 8종** (kind, text). 적용 불가한 것·원문과 같아지는 것은 빠진다.
    순서는 고정이고 난수가 없다 → 같은 문제에 늘 같은 섭동이 나온다(재현 가능).
    앞 6종은 h22 의 `rule_perturbations` 와 같고, 뒤 2종은 규칙 기반 '재서술' 이다(LLM 불필요)."""
    cands = [
        ("rename", _rename(text, UPPER_SHIFT, LOWER_PAIRS)),
        ("rename2", _rename(text, UPPER_SHIFT2, LOWER_PAIRS2)),
        ("typos0", _typos(text, 0)),
        ("typos1", _typos(text, 1)),
        ("distract0", _distract(text, 1)),       # 숫자 없는 distractor 만 (연도는 정답을 바꿀 수 있다)
        ("distract1", _distract(text, 2)),
        ("reorder", _reorder(text)),
        ("notation", _notation(text)),
    ]
    out, seen = [], {text}
    for k, v in cands:
        if v and v not in seen:
            out.append((k, v))
            seen.add(v)
    return out


# ---------------------------------------------------------------- 답 추출 · 점수

def answer_of(text):
    """`ANSWER: n` 우선, 없으면 마지막 `\\boxed{n}`. 쉼표는 떼고 정수로. 못 뽑으면 None."""
    for rex in (ANS_RE, BOXED_RE):
        ms = rex.findall(text)
        if ms:
            try:
                return int(str(ms[-1]).replace(",", ""))
            except Exception:
                continue
    return None


def score_one(orig_ans, pert_answers):
    """(score_spurious, 쓸 수 있는 섭동 수). 판정 불가면 (None, n).

    원문제 답이 없으면 비교 기준이 없다 → None.
    섭동 답이 없는 칸은 **분모에서 뺀다** — 측정 실패를 '흔들렸다' 로 세면 잡음이 신호로 위장한다
    (H22-c 의 '절단 제외' 와 같은 원리). 쓸 수 있는 섭동이 K_MIN 미만이면 판정 불가.
    """
    if orig_ans is None:
        return None, 0
    usable = [a for a in pert_answers if a is not None]
    if len(usable) < K_MIN:
        return None, len(usable)
    diff = sum(1 for a in usable if a != orig_ans)
    return diff / len(usable), len(usable)


# ---------------------------------------------------------------- 길이 폴백 (h15C 그대로)

def log_words(text):
    return math.log(max(1, len(str(text).split())))


def h15c_decide(problems, thr=-4.084031):
    """h15C 와 **같은 점수·같은 임계**. 리더보드 0.57 로 우리 최고치다."""
    return [bool(-log_words(p) >= thr) for p in problems]


def load_art():
    try:
        with open(ART, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------- 시간·VRAM 계획

def kv_kb_per_token(model):
    cfg = model.config
    nl = getattr(cfg, "num_hidden_layers", 32)
    nkv = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 32)
    hd = getattr(cfg, "head_dim", None) or (getattr(cfg, "hidden_size", 4096) //
                                            getattr(cfg, "num_attention_heads", 32))
    return 2 * nl * nkv * hd * 2 / 1024.0


def token_slots(model, cap, prompt_tok=None):
    """한 배치에 들 수 있는 시퀀스 수 = min(KV 제약, **프리필 토큰 제약**).

    **왜 두 번째 제약이 필요한가 (2026-10-18, 검증 파드에서 OOM 으로 확인):** KV 만 세면 Skywork·
    cap 512 에서 칸이 927 로 나와 241 시퀀스를 한 배치에 넣는다. 그러면 프리필 한 번에 14.4만 토큰이
    들어가고 MLP 중간 활성값이 **10.47 GiB** 를 요구해 80 GB 에서도 터진다(`torch.OutOfMemoryError`).
    디코드 때는 활성값이 B×1 토큰뿐이라 작고, **프리필이 유일한 봉우리**다. 그래서 토큰 예산으로 묶는다.
    `MAX_PREFILL_TOK` 은 실측(241×600 = 144k 토큰 → 10.47 GiB, 약 72 KB/토큰)에서 역산한 보수값이다."""
    pt = float(prompt_tok or PROMPT_TOK)
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        free_kb = free / 1024.0 * KV_SAFE
        kv = max(1, int(free_kb / ((pt + cap) * kv_kb_per_token(model))))
    except Exception:
        kv = 8
    return max(1, min(kv, int(MAX_PREFILL_TOK / max(1.0, pt))))


def seconds_for(n_seq, cap, slots, scale=1.0):
    """패딩 배치 HF generate 는 그룹마다 cap 스텝을 전부 돈다 + 강제 즉답 8스텝."""
    groups = max(1, math.ceil(n_seq / max(1, slots)))
    tot = 0.0
    for g in range(groups):
        B = min(slots, n_seq - g * slots)
        tot += MULT * (A_ + B_ * B + C_ * B * (PROMPT_TOK + cap / 2)) * cap / 1000.0
        tot += MULT * (A_ + B_ * B + C_ * B * (PROMPT_TOK + cap)) * ANS_STEPS / 1000.0
    return tot * scale


def plan(n_seq, budget, model, scale=1.0):
    """예산 안에 드는 **최대 cap** 을 사다리에서 고른다. 바닥도 못 사면 None → 폴백."""
    for cap in CAP_LADDER:
        s = token_slots(model, cap)
        est = seconds_for(n_seq, cap, s, scale)
        if est <= budget * 0.80:
            return cap, s, est
    return None


# ---------------------------------------------------------------- 생성

def get_model(model_id):
    """**판정 대상 모델 자신**을 bf16 으로 연다. 다른 모델이 오면 앞 모델을 놓아준다."""
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
    # 평가 컨테이너는 인터넷이 차단되고 `HF_HUB_OFFLINE=1` 이라 **캐시만** 쓴다(기본값).
    # `TTP_ALLOW_DOWNLOAD=1` 은 **검증 파드 전용 훅**이다 — 파드에는 모델 캐시가 없어서 받아야 한다.
    # 제출 시에는 설정하지 않으므로 오프라인 경로가 그대로 쓰인다.
    local_only = os.environ.get("TTP_ALLOW_DOWNLOAD", "") != "1"
    kw = dict(local_files_only=local_only, low_cpu_mem_usage=True, device_map="auto",
              dtype=torch.bfloat16, attn_implementation="sdpa")
    if os.environ.get("TTP_QUANT") == "4bit":
        from transformers import BitsAndBytesConfig
        kw.pop("dtype")
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    for name in [want] + ([want.split(":")[0]] if ":" in want else []):
        t = time.time()
        try:
            try:        # DeepSeek 은 AutoTokenizer 가 옛 LlamaTokenizer 로 열어 인코딩이 달라진다
                tok = PreTrainedTokenizerFast.from_pretrained(name, local_files_only=local_only)
            except Exception:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=local_only)
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


def chat_prompt(tok, problem):
    return tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{problem}"}],
                                   tokenize=False, add_generation_prompt=True)


def _gen_ids(tok, model, prompts, max_new, deadline, sample=False, temperature=1.0, top_k=40, seed=0):
    """생성 토큰 **id** 를 돌려준다(디코드는 호출부에서). 체크포인트 절단을 토큰 단위로 하려면 id 가 필요하다."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return time.time() > deadline

    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    gkw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id, do_sample=bool(sample),
               stopping_criteria=StoppingCriteriaList([Deadline()]))
    if sample:
        gkw.update(temperature=temperature, top_p=0.95)
        if top_k:
            gkw["top_k"] = int(top_k)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, **gkw)
    new = out[:, enc["input_ids"].shape[1]:]
    pads = {tok.pad_token_id}
    return [[t for t in row if t not in pads] for row in new.tolist()]


def _gen_safe(tok, model, prompts, max_new, deadline, **kw):
    """OOM 이면 **배치를 반으로 쪼개 다시** 시도한다 (h22 가 쓰는 패턴).

    토큰 예산(`MAX_PREFILL_TOK`)으로 봉우리를 눌러도 모델·문맥 조합에 따라 틀릴 수 있다. 추정이
    틀렸을 때 **실행 전체를 날리지 않고** 느려지기만 하게 만드는 것이 이 함수의 목적이다.
    시퀀스 1개에서도 OOM 이면 그 행은 빈 생성으로 두고 넘어간다(판정은 폴백으로 간다)."""
    import torch
    if not prompts:
        return []
    try:
        return _gen_ids(tok, model, prompts, max_new, deadline, **kw)
    except Exception as e:
        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        if not oom:
            raise
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if len(prompts) == 1:
            log(f"  OOM — 시퀀스 1개도 안 들어간다. 그 행은 빈 생성으로 둔다")
            return [[]]
        h = len(prompts) // 2
        log(f"  OOM → 배치 반감 {len(prompts)} → {h}+{len(prompts) - h}")
        return (_gen_safe(tok, model, prompts[:h], max_new, deadline, **kw)
                + _gen_safe(tok, model, prompts[h:], max_new, deadline, **kw))


def cot_then_answer_multi(tok, model, texts, cap, slots, deadline, ckpts=None,
                          sample=False, temperature=1.0, top_k=40, seed=0):
    """CoT 를 cap 토큰까지 **한 번** 생성하고, 여러 체크포인트에서 각각 `</think>\nANSWER:` 를 박아
    답을 강제로 받는다. 반환 (cots, {ckpt: [(ans_text, answer_int)]}).

    **왜 다중 체크포인트인가**: CoT 생성이 비용의 거의 전부이고 강제 즉답은 8스텝이라 사실상 공짜다.
    한 번 생성해 여러 cap 을 재면 'CoT 예산' 격자가 **추가 GPU 비용 0** 으로 생긴다. 제출물은
    체크포인트 하나(= cap)만 쓰고, 검증은 같은 함수를 여러 체크포인트로 부른다 → **같은 코드 경로**다.
    ckpt=0 은 CoT 없이 바로 답을 강제하는 경우(= h21c 와 같은 양)."""
    ckpts = sorted(set(ckpts or [cap]))
    base = [chat_prompt(tok, t) for t in texts]
    # **추정 대신 실측**: 실제 프롬프트 최대 토큰 길이로 배치 칸을 다시 잰다(프리필 봉우리가 그걸로 정해진다)
    pmax = int(PROMPT_TOK)
    try:
        pmax = max(len(tok(b)["input_ids"]) for b in base)
        slots = max(1, min(int(slots), int(MAX_PREFILL_TOK / max(1, pmax))))
        log(f"  프롬프트 최대 {pmax} 토큰 → 배치 칸 {slots} (프리필 상한 {MAX_PREFILL_TOK} 토큰)")
    except Exception:
        pass
    cot_ids = []
    for i in range(0, len(base), slots):
        if time.time() > deadline:
            cot_ids += [[] for _ in range(len(base) - len(cot_ids))]
            break
        cot_ids += _gen_safe(tok, model, base[i:i + slots], cap, deadline,
                             sample=sample, temperature=temperature, top_k=top_k, seed=seed)
    cot_ids = (cot_ids + [[] for _ in base])[:len(base)]
    cots = [tok.decode(ids, skip_special_tokens=True) for ids in cot_ids]
    res = {}
    pmax0 = pmax
    for ck in ckpts:
        pre = [tok.decode(ids[:ck], skip_special_tokens=True) if ck else "" for ids in cot_ids]
        forced = [b + p + FORCE for b, p in zip(base, pre)]
        # 강제 즉답의 프리필은 **프롬프트 + CoT** 라 ckpt 가 클수록 길다 → 칸을 다시 잰다
        slots_ck = max(1, min(int(slots), int(MAX_PREFILL_TOK / max(1, pmax0 + ck + 8))))
        ans = []
        for i in range(0, len(forced), slots_ck):
            if time.time() > deadline:
                ans += [""] * (len(forced) - len(ans))
                break
            ids = _gen_safe(tok, model, forced[i:i + slots_ck], ANS_STEPS, deadline, sample=False)
            ans += [tok.decode(x, skip_special_tokens=True) for x in ids]
        ans = (ans + [""] * len(forced))[:len(forced)]
        res[ck] = [(a, answer_of("ANSWER:" + a)) for a in ans]
    return cots, res


def cot_then_answer(tok, model, texts, cap, slots, deadline, sample=False, seed=0):
    """제출물 경로 — 체크포인트 하나(cap)만 쓰는 얇은 래퍼. [(cot, ans_text, answer_int)]."""
    cots, res = cot_then_answer_multi(tok, model, texts, cap, slots, deadline, [cap],
                                      sample=sample, seed=seed)
    return [(c, a, v) for c, (a, v) in zip(cots, res[cap])]


# ---------------------------------------------------------------- 판정

def decide(problems, model_id="", effort="default"):
    """문제마다 bool. 생성·모델이 안 되면 h15C 로 떨어진다(전량 무효 없음)."""
    art = load_art()
    thr = float(art.get("thr", 0.10))
    probs = list(problems)
    fb = h15c_decide(probs)
    if not probs:
        return []
    tok, model = get_model(model_id)
    if model is None:
        log(f"  모델 없음 → h15C 폴백 ({len(probs)}문제)")
        return fb
    perts = [perturbations(p) for p in probs]
    n_seq = sum(1 + len(ps) for ps in perts)
    budget = max(0.0, SOFT - (time.time() - T0)) * EFFORT_MULT.get(effort, 1.0)
    pl = plan(n_seq, budget, model)
    if pl is None:
        log(f"  예산 부족(시퀀스 {n_seq} · 남음 {budget:.0f}s) → h15C 폴백")
        return fb
    cap, slots, est = pl
    log(f"  생성 계획: 시퀀스 {n_seq} · cap {cap} · 배치칸 {slots} · 추정 {est:.0f}s / 예산 {budget:.0f}s")
    deadline = T0 + SOFT
    flat, owner = [], []
    for i, (p, ps) in enumerate(zip(probs, perts)):
        flat.append(p)
        owner.append((i, -1))
        for j, (_, v) in enumerate(ps):
            flat.append(v)
            owner.append((i, j))
    try:
        res = cot_then_answer(tok, model, flat, cap, slots, deadline)
    except Exception as e:
        log(f"  생성 실패 {type(e).__name__} {str(e)[:120]} → h15C 폴백")
        return fb
    orig = [None] * len(probs)
    pert = [[] for _ in probs]
    for (i, j), (_, _, a) in zip(owner, res):
        if j < 0:
            orig[i] = a
        else:
            pert[i].append(a)
    out, nfb = [], 0
    for i in range(len(probs)):
        s, nus = score_one(orig[i], pert[i])
        if s is None:
            out.append(fb[i])
            nfb += 1
        else:
            out.append(bool(s <= thr))
    log(f"  판정 {len(probs) - nfb}/{len(probs)} · 폴백 {nfb} · True {sum(out)}/{len(out)} · 임계 {thr:.4f}")
    return [bool(x) for x in out]
