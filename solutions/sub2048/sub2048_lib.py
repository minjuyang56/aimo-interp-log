"""sub2048 — 창 2048 '풀이 도중 벡터' 프로브. (지휘 #SUB-GEN2048, 2026-10-12)

절차 (호출 = (model_id, effort) 그룹 하나):
  1. 그 `model_id` **자신**을 bf16 으로 연다(앞 모델은 해제). 가중치가 없는 모델은 **열지도 않는다**.
  2. 문제마다 greedy 로 **최대 2048 토큰** 생성. greedy 는 결정적이므로 **학습 때 cap 32768 로
     생성한 풀이의 앞 2048 토큰과 같다** — 그래서 창을 줄이지 않고도 생성을 2048 에서 끊을 수 있다.
  3. 생성 텍스트를 `debpe` 로 정규화한 뒤 학습과 **같은 순서로** 토크나이즈해 이어 붙이고
     (`tok(chat_prompt).ids + tok(sol, add_special_tokens=False).ids[:2048]`) **prefill forward 1회**.
     층 `L//2` 의 **풀이 토큰 구간 평균 풀링** = 특징.
  4. 모델별 가중치(StandardScaler → PCA 24 → 로지스틱) 적용 → **모델별 절대 임계 `thr`** 로 판정.

## 왜 창 2048 인가
#DS-RELABEL 재평가에서 창 2048 만 교집합판·보충판 **둘 다** 통과 기준을 넘었다(모델평균 0.871 /
0.872, 기준선 셋 전부보다 높음). 사전등록 주창 512 는 둘 다 미달이라 판정은 '신호 미입증' 이고,
2048 은 '사전등록 반복이 필요한 유일한 미해결 후보' 로 남았다. 이 제출물이 그 반복을
**비공개 검증셋**에서 하는 것에 해당한다. 2048 창이 풀이의 16~37% 를 보므로 답 누출을 배제할 수
없다는 한계도 그때 적어 뒀다 — 리더보드 점수는 그 한계를 포함한 값으로 읽어야 한다.

## 판정이 중앙값 분할이 아닌 이유 (계측으로 고름, 리더보드 점수는 안 봤다)
train 라벨이 robust 로 쏠려 있어(26/4 · 21/10 · 29/3) 로지스틱의 자연 경계(0)는 True 를 평균
**81%** 찍는다 = 상수 예측에 가까워 균형 셋에서 분해능이 없다. `thr` = **train 점수 중앙값**이면
True 비율이 0.48~0.50 이 되고, **절대 임계라 호출 그룹 크기에 무관**하다(리더보드는 모델당 3~5문제뿐이라
중앙값 분할이 거칠다). train 정확도로 고르지 않았다 — 30문제·수천 차원이 PCA 24 에서도 완전
분리되므로 그 1.00 은 과적합의 증거다.

## 폴백 (지휘 지정: h15C)
가중치 없는 `model_id`(Qwen3.5-4B · gpt-oss-120b · 미지) · 시간 예산 부족 · 생성/특징 실패 ·
예외 — 전부 **h15C**(= −log(단어수), 임계 −4.0840, train 고정)로 떨어진다. h15C 는 우리 제출 중
최고(0.57)이고 생성이 0 이라 시간을 쓰지 않는다. `SUB2048_FALLBACK=nsym` 으로 hprobe 식
n_sym 순위 폴백으로 바꿀 수 있다(기본 꺼짐).

환경변수: `SUB2048_SOFT`(기본 3300) · `SUB2048_QUANT=4bit`(로컬 8GB 스모크) ·
`SUB2048_MAXNEW`(기본 2048 — **바꾸면 학습 창과 어긋난다**) · `SUB2048_CALLS`(기본 4) ·
`SUB2048_THR0=1`(임계를 0 으로, 비교용) · `SUB2048_STUB=1`(모델 없이 폴백 경로만 시험).
"""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

T0 = time.time()
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts" / "sub2048.json"
SOFT = float(os.environ.get("SUB2048_SOFT", 3300))
MAXNEW = int(os.environ.get("SUB2048_MAXNEW", 2048))
EXPECTED_CALLS = int(os.environ.get("SUB2048_CALLS", 4))
CACHE = Path(os.environ.get("SUB2048_CACHE", "/tmp/sub2048_cache.json"))
STATE = Path(os.environ.get("SUB2048_STATE", "/tmp/sub2048_state.json"))
H15C_THR = -4.084030533093263          # solutions/h15C/artifacts/h15.json 의 thr["C"], train 고정
BPE_MARKS = {"Ġ": " ", "Ċ": "\n", "ĉ": "\t"}     # Ġ Ċ ĉ

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)

# 토큰당 KV 바이트 (CLAUDE.md 실측): Olmo 는 MHA 라 9배 크다 → 배치를 이걸로 쪼갠다.
KV_PER_TOK = {"Skywork/Skywork-OR1-Math-7B": 56 << 10,
              "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 144 << 10,
              "allenai/Olmo-3-7B-Think": 512 << 10,
              "Qwen/Qwen3.5-4B": 128 << 10}
_m = {"id": None, "tok": None, "model": None, "failed": set()}


def log(msg):
    print(f"[sub2048 {time.time() - T0:7.1f}s] {msg}", flush=True)


def remaining():
    return SOFT - (time.time() - T0)


def jload(p, d):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return d


def jsave(p, o):
    try:
        Path(p).write_text(json.dumps(o), encoding="utf-8")
    except Exception:
        pass


def art():
    return jload(ART, {"per_model": {}})


def weights_for(model_id):
    pm = art().get("per_model", {})
    w = pm.get(str(model_id or "").strip())
    if w is None and ":" in str(model_id or ""):
        w = pm.get(str(model_id).split(":")[0])
    return w


def debpe(t):
    """vLLM 디토큰화 손상(Ġ/Ċ) 2차 방어. 워커는 HF 경로라 안 생기지만, 생기면 특징이 통째로
    어긋나므로 가드를 켠 채로 둔다 (#TOK-AUDIT: DeepSeek 41/41 전행이 손상된 적이 있다)."""
    for k, v in BPE_MARKS.items():
        t = t.replace(k, v)
    return t


# ---------- 폴백 ----------
def log_words(text):
    return math.log(max(1, len(str(text).split())))


def h15c_decide(texts):
    """h15C: score = −log(단어수), 임계는 train 에서 고정. 짧으면 robust."""
    return [bool(-log_words(t) >= H15C_THR) for t in texts]


_RE_MATH = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^]]*\\]")
_RE_SYM = re.compile(r"(?<![A-Za-z])[a-zA-Z](?![A-Za-z])")


def n_sym(text):
    """h15A 와 같은 양(수식 구간 안 홀로 선 한 글자 수). nsym 폴백을 쓸 때만 쓴다."""
    return sum(len(_RE_SYM.findall(m.group(0))) for m in _RE_MATH.finditer(str(text)))


def fallback(texts):
    if os.environ.get("SUB2048_FALLBACK") == "nsym":
        vals = [float(n_sym(t)) for t in texts]
        n = len(vals)
        if n == 1:
            return [True]
        order = sorted(range(n), key=lambda i: (-vals[i], -i))       # n_sym 작음 = robust
        out = [True] * n
        for i in order[:n // 2]:
            out[i] = False
        return out
    return h15c_decide(texts)


# ---------- 모델 ----------
def get_model(model_id):
    """판정 대상 모델 자신을 bf16 으로 연다. 가중치가 없으면 부르지 않는다(호출부가 막는다)."""
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
    if os.environ.get("SUB2048_QUANT") == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    for name in [want] + ([want.split(":")[0]] if ":" in want else []):
        t = time.time()
        try:
            try:        # DeepSeek 은 AutoTokenizer 가 옛 Llama 로 열어 인코딩이 달라진다 (9/23 함정)
                tok = PreTrainedTokenizerFast.from_pretrained(name, local_files_only=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            tok.padding_side = "left"               # 생성 배치는 왼쪽 패딩이어야 출력 뒤쪽이 정렬된다
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
    """라벨 생성·특징 추출과 **같은 프롬프트**(solve_common.SOLVE_PROMPT == h21_uncertainty 의 그것)."""
    return tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{problem}"}],
                                   tokenize=False, add_generation_prompt=True)


def plan_batch(model_id, n, prompt_tok):
    """VRAM 여유로 한 배치 시퀀스 수를 정한다. Olmo 는 토큰당 KV 512 KB 라 같은 배치가 9배 무겁다
    (h22 에서 이 계획을 빼먹어 40시퀀스를 한 배치로 잡고 OOM 을 냈다)."""
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return max(1, min(n, 8))
    per = KV_PER_TOK.get(str(model_id), 160 << 10) * (prompt_tok + MAXNEW)
    b = int((free * 0.6) // max(1, per))
    return max(1, min(n, b))


def generate(tok, model, problems, model_id, budget_s):
    """greedy 최대 MAXNEW 토큰. 반환: 문제별 풀이 텍스트(실패는 None).
    **시간이 모자라면 생성하지 않고 전부 None** — 중간에 끊으면 특징 길이가 학습과 어긋난다."""
    import torch
    prompts = [chat_prompt(tok, p) for p in problems]
    ptok = max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    B = plan_batch(model_id, len(problems), ptok)
    # ms(B,L) = 1.15 × (15 + 0.375·B + 7.21e-4·B·L) — 워커 실측 적합(analysis/ms_per_step_model.py)
    L = ptok + MAXNEW / 2
    ms = 1.15 * (15 + 0.375 * B + 7.21e-4 * B * L)
    groups = (len(problems) + B - 1) // B
    est = groups * MAXNEW * ms / 1000.0
    log(f"  생성 계획: {len(problems)}문제 · 배치 {B} · 그룹 {groups} · 프롬프트 {ptok}토큰 · "
        f"{ms:.1f} ms/step → 추정 {est:.0f}s (예산 {budget_s:.0f}s)")
    if est > budget_s:
        log("  !! 추정이 예산을 넘는다 → 생성 포기, 폴백으로 간다")
        return [None] * len(problems)
    out = [None] * len(problems)
    for i in range(0, len(problems), B):
        idx = list(range(i, min(i + B, len(problems))))
        if remaining() < 60:
            log("  !! 남은 시간 60s 미만 → 나머지 그룹 포기")
            break
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        t = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=MAXNEW,
                                 pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for k, j in enumerate(idx):
            txt = debpe(tok.decode(new[k], skip_special_tokens=True))
            out[j] = txt if txt.strip() else None
        log(f"  생성 그룹 {i // B + 1}/{groups} · {len(idx)}문제 · {time.time() - t:.0f}s")
        del enc, gen, new
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return out


def feature(tok, model, problem, solution, layer):
    """**experiments/probe_gen.py 와 같은 양**: 프롬프트 뒤에 풀이를 붙여 forward 1회,
    풀이 토큰 앞 MAXNEW 개 평균 풀링. 토크나이즈는 학습과 같은 순서로 **따로 세어 이어 붙인다**
    (한 번에 세면 경계 토큰이 합쳐져 풀이 시작 위치가 밀린다)."""
    import torch
    pid = tok(chat_prompt(tok, problem), add_special_tokens=False)["input_ids"]
    sid = tok(solution, add_special_tokens=False)["input_ids"][:MAXNEW]
    if not sid:
        return None
    ids = torch.tensor([pid + sid], device=model.device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    s = len(pid)
    h = out.hidden_states[layer][0, s:s + len(sid), :].float().mean(0).cpu().numpy()
    del out
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return h


def probe_score(w, h):
    """score = ((h − mean)/scale − pca_mean) @ comps.T @ coef + intercept. 높을수록 robust."""
    import numpy as np
    z = (np.asarray(h, dtype=np.float64) - np.asarray(w["mean"])) / np.asarray(w["scale"])
    p = (z - np.asarray(w["pca_mean"])) @ np.asarray(w["comps"]).T
    return float(p @ np.asarray(w["coef"]) + w["intercept"])


def key_of(t, model_id):
    return hashlib.sha1(f"{model_id}|{MAXNEW}|{str(t).strip()}".encode("utf-8")).hexdigest()[:16]


def decide(problems, model_id="", effort="default"):
    n = len(problems)
    w = weights_for(model_id)
    state = jload(STATE, {"calls": 0})
    state["calls"] = int(state.get("calls", 0)) + 1
    jsave(STATE, state)
    budget = max(0.0, remaining()) / max(1, EXPECTED_CALLS - (state["calls"] - 1))
    log(f"호출 {state['calls']} · model_id {model_id} · effort {effort} · 문제 {n} · "
        f"남은 {remaining():.0f}s · 이 호출 예산 {budget:.0f}s")
    if w is None:
        log("  가중치 없는 model_id → 모델을 열지 않고 h15C 폴백")
        return fallback(problems)
    if os.environ.get("SUB2048_STUB") == "1":
        log("  STUB=1 → 폴백 경로만 시험")
        return fallback(problems)

    cache = jload(CACHE, {})
    scores = [None] * n
    todo = []
    for i, p in enumerate(problems):
        c = cache.get(key_of(p, model_id))
        if isinstance(c, (int, float)):
            scores[i] = float(c)
        else:
            todo.append(i)
    if todo:
        tok, model = get_model(model_id)
        if tok is None:
            log("  모델을 못 열었다 → h15C 폴백")
            return fallback(problems)
        sols = generate(tok, model, [problems[i] for i in todo], model_id, budget * 0.8)
        got = 0
        for i, sol in zip(todo, sols):
            if sol is None:
                continue
            try:
                h = feature(tok, model, problems[i], sol, int(w["layer"]))
            except Exception as e:
                log(f"  특징 실패 (문제 {i}): {type(e).__name__} {str(e)[:100]}")
                h = None
            if h is None:
                continue
            scores[i] = probe_score(w, h)
            cache[key_of(problems[i], model_id)] = scores[i]
            got += 1
        jsave(CACHE, cache)
        log(f"  프로브 점수 확보 {got}/{len(todo)} (캐시 적중 {n - len(todo)})")

    ok = [i for i in range(n) if scores[i] is not None]
    if not ok:
        log("  점수 0개 → h15C 폴백 전량")
        return fallback(problems)
    thr = 0.0 if os.environ.get("SUB2048_THR0") == "1" else float(w.get("thr", 0.0))
    out = fallback(problems)                      # 점수 없는 문제는 폴백 값을 그대로 둔다
    for i in ok:
        out[i] = bool(scores[i] > thr)
    log(f"  임계 {thr:+.3f} · 프로브 판정 {len(ok)}문제 · True {sum(out)}/{n}")
    return [bool(x) for x in out]
