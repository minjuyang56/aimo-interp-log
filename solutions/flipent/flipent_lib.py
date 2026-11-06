"""#FLIPENT-ZIP — 풀이 앞구간 토큰 엔트로피로 판정한다. **실험용 제출물**이다.

무엇을 하는가 (한 줄): 판정 대상 모델 자신에게 원문제를 greedy 로 **최대 1024 토큰** 풀게 하고,
생성 **앞 25%** 토큰의 전어휘 엔트로피 평균(`ent_front`)이 **모델별 고정 임계 이상이면 robust**.

      score = ent_front = mean_{t < ceil(T/4)} ( -sum_v p_v ln p_v )     **큼 = robust**
      robust  <=>  ent_front >= thr[model_id]

부호와 임계의 출처 (사전등록 CLAUDE.md `## #FLIP-ENT`, 커밋 4fcd838 — 결과 보고 바꾸지 않는다):
  * 부호는 P2 의 '엔트로피 큼 = spurious' 를 **뒤집은 것**이다. 지휘 #FLIP-ENT 가 그렇게 박았다.
  * 임계는 공식 `train-main-v2` 33문제를 **같은 생성 설정**으로 돌린 P2 산출물의 모델별 **중앙값**
    이다. 공식 라벨을 보지 않으므로 AUC 최적화가 아니고, 절대 임계라 그룹 크기에 무관하다.

**이것이 실험 제출물인 이유를 코드에도 적어 둔다.** #FLIP-ENT 에서 이 점수는 자체 라벨 네 팔
전부에서 셔플 null 을 못 넘었다(AUC 0.721/0.562/0.337/0.526 · null 상단 0.740/0.686/0.756/0.618)
그리고 그 팔들의 고정임계 정확도는 각 팔의 상수 기준선 **아래**였다. 상설 빌드 규칙(공식 33 에서
0.645 초과)의 **예외로 [author]가 지명한 것**이고, 점수 기대값은 상수 수준이다.

cap 을 사다리로 내리지 않는 이유 (h20e·ttp 와 다른 점):
  `ent_front` 는 '**앞 25%**' 라서 T 가 바뀌면 재는 구간이 바뀐다 → cap 을 512 로 깎으면 임계를
  만든 양과 다른 양이 된다. 그래서 **cap 1024 를 못 사면 생성을 아예 하지 않고 h15C 로 떨어진다**
  (절반만 맞는 측정보다 길이 규칙이 낫다). 예산은 넉넉하다 — 문제당 시퀀스가 1개뿐이라
  `analysis/flipent_time.py` 로 최종셋형 4모델x40문제가 3300s 의 10% 안이다.

정밀도를 **fp16 으로 맞춘 이유**: 임계가 Kaggle fp16 측정에서 나왔고, P2 규칙③ 감사에서 같은
greedy 설정인데 정밀도만 바뀌어 ent_front 가 0.034 움직인 관측이 있다(Qwen3.5, 문제별 rho_s 0.449).
임계가 0.136~0.251 이라 그 크기는 12~25% 다. `FLIPENT_DTYPE=bf16` 으로 바꿀 수 있다.

폴백: 모델을 못 열거나 임계가 없는 `model_id`(gpt-oss-120b 등) 이거나 예산이 cap 1024 를 못 사거나
생성이 실패하거나 엔트로피가 유한하지 않으면 **h15C**(-log 단어수 · 고정 임계)로 떨어진다.
최악의 경우 h15C 와 같은 제출물이 되고 **전량 무효는 원리상 없다**.
"""

import json
import math
import os
import sys
import time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts", "flipent.json")

SOFT = float(os.environ.get("FLIPENT_SOFT", 3300))     # 전 호출 합산 안전 예산
CAP = int(os.environ.get("FLIPENT_CAP", 1024))         # 사전등록 생성 상한 (스모크만 낮춘다)
FRONT_FRAC = 0.25                                      # 앞 25% — ent_feats 와 같은 정의
MIN_T = 8                                              # T < 8 이면 사분면에 2개가 안 들어간다 → 폴백
A_, B_, C_, MULT = 15.0, 0.375, 7.21e-4, 1.15           # ms(B,L) 워커 실측 적합 (±0.6 ms, 4점)
PROMPT_TOK = 600.0
KV_SAFE = float(os.environ.get("FLIPENT_KV_SAFE", 0.55))
#  프리필 한 번에 넣는 토큰 상한. ttp 검증 파드 실측(241x600 = 144k 토큰 -> 활성값 10.47 GiB 요구 ->
#  80 GB 에서도 OOM)에서 역산한 보수값. 디코드는 활성값이 Bx1 토큰뿐이라 **프리필이 유일한 봉우리**다.
MAX_PREFILL_TOK = int(os.environ.get("FLIPENT_MAX_PREFILL_TOK", 40000))
EFFORT_MULT = {"low": 0.8, "default": 1.0, "medium": 1.0, "high": 1.1}

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)

_m = {"model": None, "tok": None, "id": None, "failed": set()}


def log(msg):
    print(f"[flipent {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 점수

def ent_front(ent):
    """생성 앞 25% 의 엔트로피 평균. `analysis/entlens_eval.ent_feats` 의 첫 성분과 **같은 식**이다
    (`nf = ceil(T/4)`). 그 동일성은 `analysis/flipent_smoke.py` 가 수치로 검사한다."""
    a = [float(x) for x in (ent or [])]
    T = len(a)
    if T < MIN_T:
        return None
    nf = -(-T // 4)                       # ceil(T/4) — 정수 연산으로 둔다(부동소수 반올림 회피)
    v = a[:nf]
    s = math.fsum(v) / len(v)
    return s if math.isfinite(s) else None


def log_words(text):
    return math.log(max(1, len(str(text).split())))


def h15c_decide(problems, thr=-4.084030533093263):
    """h15C 와 **같은 점수·같은 임계**(그 가중치 파일의 값 그대로). 리더보드 0.57 로 우리 최고치다."""
    return [bool(-log_words(p) >= thr) for p in problems]


def load_art():
    try:
        with open(ART, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def thr_for(art, model_id):
    """그 `model_id` 의 고정 임계. 없으면 None → 그 호출은 전부 폴백(다른 모델 임계를 빌리지 않는다 —
    엔트로피 수준이 모델마다 다르다: 중앙값 0.136~0.251)."""
    per = (art or {}).get("per_model") or {}
    want = str(model_id or "").strip()
    row = per.get(want) or per.get(want.split(":")[0])
    try:
        return float(row["thr"]) if row else None
    except Exception:
        return None


# ---------------------------------------------------------------- 시간 · VRAM 계획

def kv_kb_per_token(model):
    cfg = model.config
    nl = getattr(cfg, "num_hidden_layers", 32)
    nkv = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 32)
    hd = getattr(cfg, "head_dim", None) or (getattr(cfg, "hidden_size", 4096) //
                                            getattr(cfg, "num_attention_heads", 32))
    return 2 * nl * nkv * hd * 2 / 1024.0


def token_slots(model, cap, prompt_tok=None):
    """한 배치에 들 수 있는 시퀀스 수 = min(KV 제약, 프리필 토큰 제약). ttp_lib 와 같은 규칙."""
    pt = float(prompt_tok or PROMPT_TOK)
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        kv = max(1, int(free / 1024.0 * KV_SAFE / ((pt + cap) * kv_kb_per_token(model))))
    except Exception:
        kv = 8
    return max(1, min(kv, int(MAX_PREFILL_TOK / max(1.0, pt))))


def seconds_for(n_seq, cap, slots, scale=1.0):
    """패딩 배치 HF generate 는 그룹마다 cap 스텝을 전부 돈다(완주가 드물어 조기 종료를 안 센다)."""
    groups = max(1, math.ceil(n_seq / max(1, slots)))
    tot = 0.0
    for g in range(groups):
        B = min(slots, n_seq - g * slots)
        tot += MULT * (A_ + B_ * B + C_ * B * (PROMPT_TOK + cap / 2)) * cap / 1000.0
    return tot * scale


# ---------------------------------------------------------------- 모델 · 생성

def get_model(model_id):
    """**판정 대상 모델 자신**을 연다(기본 fp16 — 임계를 만든 측정과 같은 정밀도).
    다른 모델이 오면 앞 모델을 놓아준다(규칙 6: 로드는 호출당 1회)."""
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
    # 평가 컨테이너는 인터넷 차단 · `HF_HUB_OFFLINE=1` 이라 **캐시만** 쓴다.
    # `FLIPENT_ALLOW_DOWNLOAD=1` 은 검증 전용 훅이고 제출 시에는 설정하지 않는다.
    local_only = os.environ.get("FLIPENT_ALLOW_DOWNLOAD", "") != "1"
    dt = os.environ.get("FLIPENT_DTYPE", "fp16")
    kw = dict(local_files_only=local_only, low_cpu_mem_usage=True, device_map="auto",
              attn_implementation="sdpa")
    if os.environ.get("FLIPENT_QUANT") == "4bit":       # 로컬 8 GB 스모크 전용
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
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
            log(f"  {name} 로드 실패: {type(e).__name__} {str(e)[:120]}")
            continue
        log(f"  {name} 로드 {time.time() - t:.1f}s · dtype {os.environ.get('FLIPENT_QUANT') or dt}")
        _m["tok"], _m["model"], _m["id"] = tok, model, want
        return tok, model
    _m["failed"].add(want)
    return None, None


def chat_prompt(tok, problem):
    """라벨 생성·P2·#FLIP-ENT 와 **같은 문자열**(`SOLVE_PROMPT` + 문제, 생성 프롬프트 붙임)."""
    return tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{problem}"}],
                                   tokenize=False, add_generation_prompt=True)


def _gen_ent(tok, model, prompts, cap, deadline):
    """greedy 생성 + **스텝마다 전어휘 엔트로피만** 쌓아 돌려준다. [[H_t ...] ...] (행별 생성 길이)

    `output_scores=True` 를 쓰지 않는 이유: cap x B x V(151k) float 를 전부 들고 있으면 수십 GB 다.
    `LogitsProcessor` 는 logits 를 **그대로 돌려주므로 생성을 바꾸지 않는다**(P2 커널과 같은 수법)."""
    import torch
    from transformers import (LogitsProcessor, LogitsProcessorList, StoppingCriteria,
                              StoppingCriteriaList)

    class EntProc(LogitsProcessor):
        def __init__(self, n):
            self.H = [[] for _ in range(n)]

        def __call__(self, input_ids, scores):
            lp = torch.log_softmax(scores.float(), dim=-1)
            h = -(lp.exp() * lp).sum(-1)
            for i, v in enumerate(h.tolist()):
                if i < len(self.H):
                    self.H[i].append(float(v))
            return scores

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return time.time() > deadline

    #  **`add_special_tokens=False` 가 아니면 측정과 다른 양이 된다.** 채팅 템플릿이 이미 BOS 를
    #  넣으므로 기본값(True)으로 토크나이즈하면 **BOS 가 두 번** 들어가고, 그 프롬프트에서 나온
    #  생성·엔트로피는 임계를 만든 P2 커널(`experiments/kaggle/entlens.py:146`)의 것과 다르다.
    #  `analysis/flipent_smoke.py` 가 커널과 **프롬프트 문자열·토큰 id 동일성**을 수치로 검사한다.
    enc = tok(prompts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    proc = EntProc(len(prompts))
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cap, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             logits_processor=LogitsProcessorList([proc]),
                             stopping_criteria=StoppingCriteriaList([Deadline()]))
    #  패딩 배치에서 **먼저 끝난 행**은 그 뒤 스텝에서 pad 만 받는다. 엔트로피는 스텝마다 전 행을
    #  기록하므로 그 꼬리를 잘라야 '앞 25%' 가 밀리지 않는다 → 생성 길이로 자른다.
    new = out[:, enc["input_ids"].shape[1]:].tolist()
    #  EOS 집합도 커널과 같게 모은다 — `generation_config.eos_token_id` 가 **목록**인 모델이 있고
    #  (Qwen 계열의 `<|im_end|>`/`<|endoftext|>`), 그걸 빼면 꼬리를 못 잘라 '앞 25%' 가 밀린다.
    eos = {tok.eos_token_id, tok.pad_token_id}
    gcfg = getattr(model, "generation_config", None)
    e = getattr(gcfg, "eos_token_id", None) if gcfg is not None else None
    if e is not None:
        eos.update(e if isinstance(e, (list, tuple)) else [e])
    eos -= {None}
    res = []
    for row, H in zip(new, proc.H):
        n = len(row)
        for j, t in enumerate(row):                  # 첫 EOS 까지(그 토큰 포함)만 센다
            if t in eos:
                n = j + 1
                break
        res.append(H[:n])
    return res


def _gen_safe(tok, model, prompts, cap, deadline):
    """OOM 이면 **배치를 반으로 쪼개 다시** 시도한다(h22·ttp 와 같은 패턴).
    추정이 틀렸을 때 실행 전체를 날리지 않고 느려지기만 하게 만드는 것이 목적이다."""
    import torch
    if not prompts:
        return []
    try:
        return _gen_ent(tok, model, prompts, cap, deadline)
    except Exception as e:
        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        if not oom:
            raise
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if len(prompts) == 1:
            log("  OOM — 시퀀스 1개도 안 들어간다. 그 행은 빈 생성으로 둔다")
            return [[]]
        h = len(prompts) // 2
        log(f"  OOM → 배치 반감 {len(prompts)} → {h}+{len(prompts) - h}")
        return (_gen_safe(tok, model, prompts[:h], cap, deadline)
                + _gen_safe(tok, model, prompts[h:], cap, deadline))


# ---------------------------------------------------------------- 판정

def decide(problems, model_id="", effort="default"):
    """문제마다 bool. 생성·모델·예산·임계 중 하나라도 없으면 h15C 로 떨어진다(전량 무효 없음)."""
    probs = list(problems)
    if not probs:
        return []
    art = load_art()
    fb = h15c_decide(probs)
    thr = thr_for(art, model_id)
    if thr is None:
        log(f"  {model_id!r} 임계 없음 → h15C 폴백 ({len(probs)}문제) · 모델을 열지 않는다")
        return fb
    tok, model = get_model(model_id)
    if model is None:
        log(f"  모델 없음 → h15C 폴백 ({len(probs)}문제)")
        return fb
    budget = max(0.0, SOFT - (time.time() - T0)) * EFFORT_MULT.get(effort, 1.0)
    slots = token_slots(model, CAP)
    est = seconds_for(len(probs), CAP, slots)
    if est > budget * 0.80:
        #  cap 을 깎지 않는다 — 앞 25% 의 정의가 바뀌면 임계가 무의미해진다(모듈 docstring 참조).
        log(f"  예산 부족: 추정 {est:.0f}s > 예산 {budget:.0f}s x 0.8 → h15C 폴백")
        return fb
    log(f"  생성 계획: 문제 {len(probs)} · cap {CAP} · 배치칸 {slots} · 추정 {est:.0f}s / 예산 {budget:.0f}s")
    deadline = T0 + SOFT
    try:
        prompts = [chat_prompt(tok, p) for p in probs]
        ents = []
        for i in range(0, len(prompts), max(1, slots)):
            ents += _gen_safe(tok, model, prompts[i:i + max(1, slots)], CAP, deadline)
    except Exception as e:
        log(f"  생성 실패 {type(e).__name__} {str(e)[:120]} → h15C 폴백")
        return fb
    out, nfb, got = [], 0, []
    for i, p in enumerate(probs):
        f = ent_front(ents[i]) if i < len(ents) else None
        if f is None:
            out.append(fb[i])
            nfb += 1
        else:
            out.append(bool(f >= thr))
            got.append(f)
    if got:
        log(f"  ent_front n={len(got)} · 평균 {sum(got) / len(got):.4f} · "
            f"범위 [{min(got):.4f}, {max(got):.4f}] · 임계 {thr:.4f}")
    log(f"  판정 {len(probs) - nfb}/{len(probs)} · 폴백 {nfb} · True {sum(out)}/{len(out)}")
    return [bool(x) for x in out]
