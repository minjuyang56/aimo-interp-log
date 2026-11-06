"""hprobe — 판정 대상 모델의 **prefill 1회** hidden state 에 모델별 선형 프로브를 걸고 호출 내 중앙값 분할.

사전등록된 최종 프로토콜(CLAUDE.md "최종 프로브 평가 프로토콜", 2026-10-08 커밋 e749b2e)을 그대로 구현한 **골격**이다.
설계는 고정이고 바뀌는 것은 `artifacts/hprobe.json` 의 가중치뿐이다 — DeepSeek·Olmo 자체 라벨이 도착하면
`analysis/build_hprobe_artifacts.py` 를 다시 돌려 **가중치만 교체**하고 이 파일은 손대지 않는다.

점수 경로 (특징 추출은 experiments/probe_pilot.py 의 extract() 와 같게 유지한다 — 어긋나면 train 가중치가 무의미해진다)
  1. SOLVE_PROMPT + 문제를 채팅 템플릿에 넣고 **생성 없이 forward 1회**
  2. 문제 본문이 시작하는 토큰 위치부터 끝까지 hidden_states[층] **평균 풀링**
  3. z = (x - mean)/scale -> p = (z - pca_mean) @ comps.T -> score = p·coef + intercept   (**높을수록 robust**)
  4. 호출 안에서 **중앙값 분할**(상위 절반 True). 절대 임계값은 쓰지 않는다.

가중치가 없는 model_id (= 아직 라벨을 만들지 못한 모델)
  기본은 **텍스트 규칙 폴백**: n_sym(= edit_rename 이 바꾸는 문자 수)의 호출 내 순위, 적을수록 robust (h15A 와 같은 양).
  다른 모델의 프로브를 빌려오지 않는 이유는 둘이다 — (a) hidden 차원이 모델마다 달라(2560/3584/4096) 가중치를
  그대로 쓸 수 없고, (b) 남의 모델을 계측기로 쓰는 설계(h20f)는 문제 단위 예측이라 모델별 균형 검증셋에서
  구조적으로 0.50 이다. 그래도 비교하고 싶으면 `HPROBE_SHARED=<model_id>` 로 계측기를 지정할 수 있다(기본 꺼짐).

**현재 가중치의 상태를 숨기지 않는다**: Skywork 88문제 라벨로 적합한 가중치의 내부 CV AUC 는 **0.454** 로
우연 아래다(파일럿의 nested 0.367 과 같은 결론). 즉 이 제출물은 **파이프라인이 완성된 빈 그릇**이고,
점수는 3모델 라벨이 모이고 사전등록 평가가 통과할 때만 기대할 수 있다.
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

T0 = time.time()
ART = Path(__file__).resolve().parent / "artifacts" / "hprobe.json"
SOFT = float(os.environ.get("HPROBE_SOFT", 3300))
EXPECTED_CALLS = int(os.environ.get("HPROBE_CALLS", 4))
RESERVE = float(os.environ.get("HPROBE_RESERVE", 15))
CACHE = Path(os.environ.get("HPROBE_CACHE", "/tmp/hprobe_cache.json"))
STATE = Path(os.environ.get("HPROBE_STATE", "/tmp/hprobe_state.json"))
SHARED = os.environ.get("HPROBE_SHARED", "").strip()      # 기본 꺼짐 (위 주석 참고)
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
_m = {"model": None, "tok": None, "id": None, "failed": set()}
_art = {"v": None}


def log(msg):
    print(f"[hprobe {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


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


def art():
    if _art["v"] is None:
        _art["v"] = jload(ART, {"per_model": {}})
    return _art["v"]


def weights_for(model_id):
    """**미지 model_id 에 KeyError 가 나면 그룹 전체가 날아간다**(절대 규칙 4) -> 항상 .get 으로만 꺼낸다."""
    pm = art().get("per_model", {})
    w = pm.get(str(model_id))
    if w is None and ":" in str(model_id):                  # 'qwen3-8b:low' 류 접미사를 떼고 한 번 더
        w = pm.get(str(model_id).split(":")[0])
    return w


# ---- 텍스트 폴백 (h15A 와 같은 n_sym) ----
def _rename(text):
    def chunk(c):
        c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + UPPER_SHIFT) % 26 + 65), c)
        return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: LOWER_PAIRS.get(m.group(1), m.group(1)), c)

    def span(m):
        s, out, pos = m.group(0), [], 0
        for cmd in LATEX_CMD.finditer(s):
            out.append(chunk(s[pos:cmd.start()]))
            out.append(cmd.group(0))
            pos = cmd.end()
        out.append(chunk(s[pos:]))
        return "".join(out)
    return MATH_SPAN.sub(span, text)


def n_sym(text):
    rn = _rename(text)
    return sum(1 for a, b in zip(text, rn) if a != b) if len(rn) == len(text) else 0


def key_of(t, model_id):
    """특징은 모델마다 다르다 -> 캐시 키에 model_id 를 넣는다(effort 는 프로브 값에 영향이 없다)."""
    return hashlib.sha1(f"{model_id}|{t.strip()}".encode("utf-8")).hexdigest()[:16]


def get_model(model_id):
    """판정 대상 모델 자신을 bf16 으로 연다. 다른 모델이 오면 앞 모델을 해제한다. 못 열면 (None, None)."""
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
    if os.environ.get("HPROBE_QUANT") == "4bit":            # 로컬 8 GB 스모크용
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    for name in [want] + ([want.split(":")[0]] if ":" in want else []):
        t = time.time()
        try:
            try:                                            # DeepSeek 은 AutoTokenizer 가 옛 Llama 로 열어 인코딩이 달라진다
                tok = PreTrainedTokenizerFast.from_pretrained(name, local_files_only=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
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


def feature(tok, model, text, layer):
    """**experiments/probe_pilot.py 의 extract() 와 같은 양**: 문제 본문 토큰 구간 평균 풀링, 생성 없음."""
    import torch
    full = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}],
                                   tokenize=False, add_generation_prompt=True)
    cpos = full.rfind(text[:40])
    pre = tok(full[:cpos] if cpos > 0 else full, return_tensors="pt", add_special_tokens=False)["input_ids"]
    enc = tok(full, return_tensors="pt", add_special_tokens=False).to(model.device)
    s = int(pre.shape[1]) if cpos > 0 else 0
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[layer][0, s:, :].float().mean(0).cpu().numpy()
    del out
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return h


def probe_score(w, h):
    """score = ((h - mean)/scale - pca_mean) @ comps.T @ coef + intercept. 높을수록 robust."""
    import numpy as np
    z = (np.asarray(h, dtype=np.float64) - np.asarray(w["mean"])) / np.asarray(w["scale"])
    p = (z - np.asarray(w["pca_mean"])) @ np.asarray(w["comps"]).T
    return float(p @ np.asarray(w["coef"]) + w["intercept"])


def rank_split(values, higher_is_robust=True):
    """호출 내 중앙값 분할: 좋은 절반을 True. 동점은 입력 순서가 앞인 쪽을 True. 1개면 True."""
    n = len(values)
    if n == 1:
        return [True]
    order = sorted(range(n), key=lambda i: (values[i] if higher_is_robust else -values[i], -i))
    out = [True] * n
    for i in order[:n // 2]:
        out[i] = False
    return out


def two_tier(n, probe, texts):
    """프로브 점수가 있는 층과 없는 층을 **각각 따로** 중앙값 분할한다.
    두 점수는 단위가 달라 한 줄로 이어 붙이면 층 사이 순서가 뒤섞인다(h20e 에서 계측해 이 방식으로 고정)."""
    out = [True] * n
    a = [i for i in range(n) if probe[i] is not None]
    b = [i for i in range(n) if probe[i] is None]
    for idx, vals, hi in ((a, [probe[i] for i in a], True),                    # 프로브 점수 큼 = robust
                          (b, [float(n_sym(texts[i])) for i in b], False)):    # n_sym 작음 = robust
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
    scores = [None] * n
    w = weights_for(model_id)
    instrument = str(model_id)
    if w is None and SHARED:                                  # 선택 사항: 남의 모델을 계측기로 쓰기(기본 꺼짐)
        w2 = weights_for(SHARED)
        if w2 is not None:
            w, instrument = w2, SHARED
            log(f"  {model_id} 가중치 없음 -> 공용 계측기 {SHARED} 사용 (HPROBE_SHARED)")
    have = f"있음(층 {w['layer']}, C {w['C']})" if w else "없음"
    log(f"호출 {state['calls']}/{EXPECTED_CALLS} · {str(model_id).split('/')[-1]} · effort {effort} · "
        f"문제 {n} · 가중치 {have} · 남은 {remaining():.0f}s · 호출예산 {budget:.0f}s")
    if w is None:
        log("  -> 텍스트 규칙(n_sym 순위)으로 전량 판정")
        return [bool(x) for x in two_tier(n, scores, list(problems))]
    todo = []
    for i, p in enumerate(problems):
        c = cache.get(key_of(p, instrument))
        if c is not None and c.get("s") is not None:
            scores[i] = float(c["s"])
        else:
            todo.append(i)
    if todo:
        tok = model = None
        try:
            t_load = time.time()
            tok, model = get_model(instrument)
            if time.time() - t_load > 1:
                budget = max(0.0, budget - (time.time() - t_load))
                log(f"  로드에 {time.time() - t_load:.0f}s -> 남은 호출예산 {budget:.0f}s")
        except Exception as e:
            log(f"  로드 실패: {type(e).__name__} {str(e)[:120]}")
        if model is None:
            log(f"  {instrument} 를 열 수 없음 -> 텍스트 규칙 폴백")
        else:
            end = time.time() + max(5.0, min(budget, remaining()) - RESERVE)
            done, t_fw = 0, time.time()
            for i in todo:
                if time.time() > end:
                    log(f"  예산 소진 -> 남은 {len(todo) - done}문제는 텍스트 층으로")
                    break
                try:
                    h = feature(tok, model, problems[i], int(w["layer"]))
                    scores[i] = probe_score(w, h)
                    cache[key_of(problems[i], instrument)] = {"s": scores[i]}
                except Exception as e:
                    log(f"  문제 {i} 특징 추출 실패: {type(e).__name__} {str(e)[:100]}")
                done += 1
            if done:
                log(f"  forward {done}회 · {(time.time() - t_fw) / done * 1000:.0f} ms/문제")
            jsave(CACHE, cache)
    k = sum(1 for s in scores if s is not None)
    out = two_tier(n, scores, list(problems))
    sv = [s for s in scores if s is not None]
    log(f"판정: 프로브 {k}문제" + (f"({min(sv):+.2f}~{max(sv):+.2f})" if sv else "")
        + f" + 텍스트 {n - k}문제 · True {sum(out)}/{n}")
    return [bool(o) for o in out]
