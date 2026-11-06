"""#FINAL - H-FIN 자연종료 규칙 + 모델별 라우팅. **라우팅은 사후 선택(EXPLORATORY)이다.**

무엇을 하는가 (한 줄): Olmo/Skywork 는 **자기 모델이 원문제를 cap 12288 안에 스스로 끝내면 robust**,
DeepSeek/Qwen3.5/그 외는 **h15C**(-log 단어수 고정 임계)로 판정한다.

      score_fin = (cap 안에서 자연종료한 샘플 수) / k        **큼 = robust**
      robust  <=>  score_fin >= 0.5        (절대 임계. k=1 이면 '그 한 샘플이 끝났나')

출처와 숫자 (CLAUDE.md H-FIN 절, hfin_report.txt, hfin_hybrid_report.txt):
  * 공식 33문제 저장 데이터(h20m, k=1, cap 12288)에서 모델별 정확도
    **Olmo 0.8182 / Skywork 0.6970 / DeepSeek 0.4848**.
  * 그래서 라우팅은 **통과한 두 모델만** H-FIN 이고 나머지는 h15C 다 -
    **이건 결과를 본 뒤의 선택이므로 사전등록이 아니다.** 리포트에도 EXPLORATORY 로 적는다.
  * 하이브리드 모델평균 정확도 **0.6818**, 셔플 null 95% 상단 0.6367, 상수 0.6061, 길이 0.6364.
    지휘가 정한 게이트(>0.65 그리고 >null 상단) 통과.

**예산 벽을 코드에 적어 둔다 - 이게 이 제출물의 가장 큰 한계다.**
게이트를 통과한 설정은 cap 12288 **하나**다. 공식 33 절대임계 0.5 정확도(hfin_report.txt):
  cap 12288 -> Olmo **0.8182** / Skywork **0.6970** / DeepSeek 0.4848 / 3모델 평균 **0.6667**
  cap  8192 -> Olmo **0.6061**(= 상수와 같다) / Skywork **0.5758**(상수 아래) / 평균 0.5455
  cap  4096 -> 평균 0.4040 (무정보 - 자연종료가 전부 False 로 포화한다)
그런데 ms(B,L) 워커 모형으로 재면 35문제 전수 생성이 Skywork 약 2,802s, Olmo 약 3,650s 라
**최종셋 규모에서는 cap 12288 이 예산 밖**이다.

**예산 사다리 (지휘 #HFIN-MAIN)**: 12288 -> 8192 -> 폴백. 지시대로 구현했고, 동시에
**8192 칸은 우리 측정에서 h15C 폴백(0.6061)보다 낫다는 증거가 없다** - Olmo 는 같고
Skywork 는 더 나쁘다. 그래서 어느 칸으로 판정했는지 로그에 찍어 사후에 가를 수 있게 했고,
`HFIN_LADDER=12288` 로 두면 사다리 없이 폴백한다.
부분 생성(일부 문제만 H-FIN)은 쓰지 않는다 - #63 혼합 측정에서 예산이 허용하는 범위(m<=20)의
혼합이 텍스트 단독을 **못 넘었다**.
-> 즉 리더보드 규모에서는 H-FIN(cap 12288)이 돌고, 최종셋 규모에서는 8192 칸이나 h15C 로 내려간다.

라벨 정의와의 순환 위험 (해석에 반드시 붙인다): 주최측 라벨도 **정해진 예산 안의 정확도**다
(공식 FAQ 7/8). 그러면 '예산 안에 스스로 끝내는가' 는 라벨 정의의 일부를 다시 읽는 것일 수 있고,
최종셋 예산이 다르면 전이가 깨진다.

샘플링은 측정과 같게 맞춘다: **T=0.6, top_p=0.95, k=1** (h20m 커널의 SamplingParams).
프롬프트도 같은 SOLVE_PROMPT + 채팅 템플릿이다.

폴백: 라우팅 대상이 아닌 model_id, 모델 로드 실패, 예산이 cap 12288 을 못 사는 경우,
생성 실패, 예외 -> **h15C**. 최악의 경우 h15C 와 같은 제출물이고 **전량 무효는 원리상 없다**.
"""

import math
import os
import sys
import time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))

SOFT = float(os.environ.get("HFIN_SOFT", 3300))        # 전 호출 합산 안전 예산
CAP = int(os.environ.get("HFIN_CAP", 12288))           # 게이트를 통과한 유일한 상한 (스모크만 낮춘다)
#  **예산 사다리 (지휘 #HFIN-MAIN)**: cap 12288 이 안 들어가면 8192 를 시도하고, 그것도 안 되면 폴백.
#  정직하게 적어 둔다 — **8192 칸은 우리 측정에서 상수보다 낫지 않다**:
#    공식 33 절대임계 0.5 정확도가 cap 8192 에서 Olmo 0.6061(= 상수와 같다) · Skywork 0.5758(상수 아래),
#    3모델 평균 0.5455. cap 12288 은 0.8182 / 0.6970 / 평균 0.6667 이다 (hfin_report.txt).
#  즉 사다리의 두 번째 칸은 '폴백보다 낫다' 는 증거가 없고, Skywork 에서는 더 나쁠 수 있다.
#  지휘 지정이라 구현하지만 어느 칸으로 판정했는지 로그에 찍어 사후에 가를 수 있게 한다.
CAP_LADDER = [int(x) for x in str(os.environ.get("HFIN_LADDER", "12288,8192")).split(",") if x.strip()]
KSAMP = int(os.environ.get("HFIN_K", 1))               # 측정과 같은 k=1
THR_FIN = 0.5                                          # 절대 임계 (EconLearn 의 '3 중 2' 는 0.667)
H15C_THR = -4.084030533093263                          # solutions/h15C/artifacts/h15.json thr["C"]
A_, B_, C_, MULT = 15.0, 0.375, 7.21e-4, 1.15           # ms(B,L) 워커 실측 적합 (+-0.6 ms, 4점)
PROMPT_TOK = 600.0
KV_SAFE = float(os.environ.get("HFIN_KV_SAFE", 0.55))
MAX_PREFILL_TOK = int(os.environ.get("HFIN_MAX_PREFILL_TOK", 40000))
EFFORT_MULT = {"low": 0.8, "default": 1.0, "medium": 1.0, "high": 1.1}

#  H-FIN 을 쓰는 모델 - **게이트를 통과한 둘만**. 사후 선택임을 여기 적어 둔다.
#  여기 없는 model_id 는 **모델을 열지도 않고** h15C 로 간다. 특히 `openai/gpt-oss-120b`
#  (Main 전용 · 195.8 GB · 로컬 캐시에 없어 한 번도 열어 본 적이 없다)는 **어떤 effort 에서도**
#  로드하지 않는다 - routing 이 effort 를 보지 않으므로 low/medium/high/default 가 모두 같은 경로다.
HFIN_MODELS = {"Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think"}

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)

_m = {"model": None, "tok": None, "id": None, "failed": set()}


def log(msg):
    print("[hfin %7.1fs] %s" % (time.time() - T0, msg), file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 점수 / 폴백

def log_words(text):
    n = max(1, len(str(text or "").split()))
    return math.log(n)


def h15c_decide(problems):
    """h15C 와 **글자 그대로 같은 식**: -log(단어수) >= 고정 임계면 robust."""
    return [bool((-log_words(p)) >= H15C_THR) for p in problems]


def score_fin(fin_flags):
    """fin_flags = 그 문제의 k개 샘플이 cap 안에서 EOS 로 끝났는지. 비율을 돌려준다."""
    if not fin_flags:
        return None
    return float(sum(1 for f in fin_flags if f)) / float(len(fin_flags))


# ---------------------------------------------------------------- 시간 / VRAM 계획

def kv_kb_per_token(model):
    cfg = model.config
    nl = getattr(cfg, "num_hidden_layers", 32)
    nkv = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 32)
    hd = getattr(cfg, "head_dim", None) or (getattr(cfg, "hidden_size", 4096) //
                                            getattr(cfg, "num_attention_heads", 32))
    return 2 * nl * nkv * hd * 2 / 1024.0


def token_slots(model, cap, prompt_tok=None):
    """한 배치에 들 수 있는 시퀀스 수 = min(KV 제약, 프리필 토큰 제약). flipent/ttp 와 같은 규칙."""
    pt = float(prompt_tok or PROMPT_TOK)
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        kv = max(1, int(free / 1024.0 * KV_SAFE / ((pt + cap) * kv_kb_per_token(model))))
    except Exception:
        kv = 8
    return max(1, min(kv, int(MAX_PREFILL_TOK / max(1.0, pt))))


def seconds_for(n_seq, cap, slots, scale=1.0):
    """패딩 배치 HF generate 는 그룹마다 cap 스텝을 전부 돈다.

    **H-FIN 에서는 이게 보수적이다** - 완주한 시퀀스가 있으면 배치가 일찍 끝날 수 있지만,
    한 시퀀스라도 cap 까지 가면 그룹 전체가 cap 스텝을 돈다. 예산 판단은 최악으로 한다."""
    groups = max(1, math.ceil(n_seq / max(1, slots)))
    tot = 0.0
    for g in range(groups):
        B = min(slots, n_seq - g * slots)
        tot += MULT * (A_ + B_ * B + C_ * B * (PROMPT_TOK + cap / 2)) * cap / 1000.0
    return tot * scale


# ---------------------------------------------------------------- 모델 / 생성

def get_model(model_id):
    """판정 대상 모델 자신을 연다(기본 bf16). 다른 모델이 오면 앞 모델을 놓아준다(규칙 6)."""
    want = str(model_id or "").strip()
    if not want or want in _m["failed"]:
        return None, None
    if _m["id"] == want and _m["model"] is not None:
        return _m["tok"], _m["model"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    if _m["model"] is not None:
        log("  앞 모델 %s 해제" % _m["id"])
        _m["model"], _m["tok"], _m["id"] = None, None, None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    local_only = os.environ.get("HFIN_ALLOW_DOWNLOAD", "") != "1"
    dt = os.environ.get("HFIN_DTYPE", "bf16")
    kw = dict(local_files_only=local_only, low_cpu_mem_usage=True, device_map="auto",
              attn_implementation="sdpa")
    if os.environ.get("HFIN_QUANT") == "4bit":          # 로컬 8 GB 스모크 전용
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16 if dt == "bf16" else torch.float16
    for name in [want] + ([want.split(":")[0]] if ":" in want else []):
        t = time.time()
        try:
            try:    # DeepSeek 은 AutoTokenizer 가 옛 LlamaTokenizer 로 열어 인코딩이 달라진다
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
            log("  %s 로드 실패: %s %s" % (name, type(e).__name__, str(e)[:120]))
            continue
        log("  %s 로드 %.1fs / dtype %s" % (name, time.time() - t,
                                            os.environ.get("HFIN_QUANT") or dt))
        _m["tok"], _m["model"], _m["id"] = tok, model, want
        return tok, model
    _m["failed"].add(want)
    return None, None


def chat_prompt(tok, problem):
    """라벨 생성/h20m 커널과 **같은 문자열**(SOLVE_PROMPT + 문제, 생성 프롬프트 붙임)."""
    return tok.apply_chat_template([{"role": "user", "content": "%s\n\n%s" % (SOLVE_PROMPT, problem)}],
                                   tokenize=False, add_generation_prompt=True)


def eos_ids(tok, model):
    """완주 판정에 쓸 EOS 집합. generation_config 가 목록을 줄 수 있다."""
    out = set()
    for src in (getattr(getattr(model, "generation_config", None), "eos_token_id", None),
                getattr(tok, "eos_token_id", None)):
        if src is None:
            continue
        out.update(src if isinstance(src, (list, tuple, set)) else [src])
    return set(int(x) for x in out if x is not None)


def _gen_fin(tok, model, prompts, cap, deadline):
    """샘플링 생성 -> 행별 **완주 여부**(cap 안에서 EOS 가 나왔나)와 생성 토큰 수.

    max_new_tokens=cap 으로 두고 **생성 구간에 EOS 토큰이 있는지**로 완주를 판정한다.
    h20m 측정의 finish_reason == 'stop' 과 같은 뜻이다."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = dict((k, v.to(model.device)) for k, v in enc.items())
    plen = enc["input_ids"].shape[1]
    ES = eos_ids(tok, model)
    gkw = dict(max_new_tokens=cap, do_sample=True, temperature=0.6, top_p=0.95,
               pad_token_id=tok.pad_token_id)
    if time.time() > deadline:
        return None
    with torch.no_grad():
        out = model.generate(**enc, **gkw)
    gen = out[:, plen:]
    res = []
    for row in gen.tolist():
        n, fin = len(row), False
        for i, t in enumerate(row):
            if t in ES:
                n, fin = i, True
                break
        res.append((bool(fin), int(n)))
    return res


def _gen_safe(tok, model, prompts, cap, deadline, slots):
    """OOM 이면 배치를 반감해 재시도한다. 추정이 틀려도 실행 전체를 날리지 않는다."""
    import torch
    res, i, b = [], 0, max(1, slots)
    while i < len(prompts):
        chunk = prompts[i:i + b]
        try:
            r = _gen_fin(tok, model, chunk, cap, deadline)
        except torch.cuda.OutOfMemoryError:
            if b == 1:
                return None
            b = max(1, b // 2)
            log("  OOM -> 배치 %d 로 반감" % b)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        if r is None:
            return None
        res.extend(r)
        i += len(chunk)
    return res


# ---------------------------------------------------------------- 판정

def decide(problems, model_id="", effort="default"):
    texts = [str(p) for p in problems]
    n = len(texts)
    if n == 0:
        return []
    mid = str(model_id or "").strip()
    fb = h15c_decide(texts)      # 폴백은 늘 준비해 둔다

    if mid not in HFIN_MODELS:
        log("모델 %s 은 라우팅 대상 아님 -> h15C 폴백 / 문제 %d개" % (mid or "(빈)", n))
        return fb

    budget = max(0.0, SOFT - (time.time() - T0)) * EFFORT_MULT.get(str(effort), 1.0)
    tok, model = get_model(mid)
    if tok is None or model is None:
        log("모델 로드 실패 -> h15C 폴백 / 문제 %d개" % n)
        return fb

    prompts = []
    for t in texts:
        p = chat_prompt(tok, t)
        for _ in range(KSAMP):
            prompts.append(p)
    try:
        ptok = float(sum(len(tok(p, add_special_tokens=False)["input_ids"])
                         for p in prompts[:n])) / max(1, n)
    except Exception:
        ptok = PROMPT_TOK
    #  **예산 사다리**: 사전등록 cap 부터 내려가며 '추정 <= 예산 x 0.8' 인 첫 칸을 고른다.
    #  per-case 로도 같이 찍는다 — 지휘 #HFIN-BUILD 의 'per_case = limit / n' 판정과 같은 양이다
    #  (호출 예산을 문제 수로 나눈 것이므로 총합 비교와 수학적으로 동일하다).
    use_cap, use_slots, use_est = None, None, None
    for cand in CAP_LADDER:
        s = token_slots(model, cand, ptok)
        e = seconds_for(len(prompts), cand, s)
        log("생성 계획 cap %5d / 시퀀스 %d / 배치칸 %2d / 프롬프트 %.0f토큰 / 추정 %6.0fs "
            "(문제당 %5.1fs) / 예산 %.0fs (문제당 %.1fs)"
            % (cand, len(prompts), s, ptok, e, e / max(1, n), budget, budget / max(1, n)))
        if e <= budget * 0.8:
            use_cap, use_slots, use_est = cand, s, e
            break
    if use_cap is None:
        log("  사다리 전 칸이 예산 밖 -> **생성하지 않고** h15C 폴백")
        return fb
    if use_cap != CAP_LADDER[0]:
        #  낮은 칸은 신호가 약하다(docstring·상수 주석 참조). 어느 칸을 썼는지 반드시 남긴다.
        log("  **사다리 하강: cap %d 로 판정한다** — 이 칸은 공식 33 에서 상수를 넘지 않는다"
            % use_cap)

    t = time.time()
    r = _gen_safe(tok, model, prompts, use_cap, T0 + SOFT, use_slots)
    if r is None:
        log("  생성 실패/예산 소진 -> h15C 폴백")
        return fb
    log("  생성 %.0fs / 완주 %d/%d" % (time.time() - t, sum(1 for f, _ in r if f), len(r)))

    outp, nfb = [], 0
    for i in range(n):
        flags = [r[i * KSAMP + j][0] for j in range(KSAMP)]
        s = score_fin(flags)
        if s is None:
            outp.append(fb[i])
            nfb += 1
        else:
            outp.append(bool(s >= THR_FIN))
    log("  판정 %d/%d / 폴백 %d / True %d/%d"
        % (n - nfb, n, nfb, sum(1 for x in outp if x), n))
    return [bool(x) for x in outp]
