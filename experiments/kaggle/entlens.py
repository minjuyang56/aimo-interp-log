"""#LIT-PLAN P2·P3 커널 본체 — 토큰 엔트로피 궤적 + logit lens '답 확정 층'.

사전등록은 CLAUDE.md `### P2 — 토큰 엔트로피 궤적` · `### P3 — logit lens` 절이다.
이 파일은 **측정만** 한다 — 특징 3개·스칼라 1개의 계산과 평가는 전부 로컬
`analysis/entlens_eval.py` 가 저장된 원시값에서 다시 낸다(집계만 저장하면 집계의 버그가 안 보인다,
상설 규칙 ①).

저장하는 원시값
  * `ent` : 토큰별 엔트로피 `H_t = −Σ_v p_v ln p_v` (**전 어휘** · raw logits softmax · 자연로그)
  * `text`: 생성 텍스트 전문 · `n_tok` · `finished`(EOS 도달) · BPE 마커 수
  * `pcurve`: 층별 `p_ℓ` = softmax(lm_head(final_norm(h_ℓ)))[강제 즉답의 첫 토큰]
  * `ans` : `</think>\\nANSWER:` 강제 8토큰 greedy 의 첫 줄 · 거기서 뽑은 정수

왜 `output_scores=True` 를 쓰지 않나: 1024스텝 × 33행 × V(151k) float 를 전부 들고 있으면
**약 20 GB** 다. `LogitsProcessor` 로 스텝마다 엔트로피만 재서 쌓으면 메모리가 O(1) 이다.

환경 변수 (헤더가 채우고, 로컬 스모크는 직접 준다)
  ENTLENS_MODEL   판정 모델 id
  ENTLENS_MAXNEW  생성 상한 (기본 1024)
  ENTLENS_BATCH   한 배치 시퀀스 수 (0 = 전부 한 배치. Olmo 는 KV 때문에 분할)
  ENTLENS_NPROB   앞 N 문제만 (0 = 33 전부. 스모크용)
  ENTLENS_QUANT   "fp16"(Kaggle T4) | "4bit"(로컬 8 GB)
  ENTLENS_LOCAL   "1" 이면 local_files_only
  ENTLENS_OUT     출력 디렉터리
"""

import base64
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,  # noqa: F401
                          LogitsProcessorList, PreTrainedTokenizerFast)

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
FORCE = "</think>\nANSWER:"
INT_RE = re.compile(r"-?\d+")
MARKERS = ("Ġ", "Ċ")                      # Ġ · Ċ — vLLM 디코드 사고의 흔적 (HF 경로엔 없어야 한다)

MODEL = os.environ.get("ENTLENS_MODEL", "Qwen/Qwen3.5-4B")
MAXNEW = int(os.environ.get("ENTLENS_MAXNEW", "1024"))
BATCH = int(os.environ.get("ENTLENS_BATCH", "0"))
NPROB = int(os.environ.get("ENTLENS_NPROB", "0"))
QUANT = os.environ.get("ENTLENS_QUANT", "fp16")
LOCAL = os.environ.get("ENTLENS_LOCAL", "0") == "1"
OUTDIR = Path(os.environ.get("ENTLENS_OUT", "/kaggle/working/entlens"))


# ───────────────────────── 문제 33개 (헤더가 PROBS_B64 를 먼저 정의한다) ─────────────────────────

def load_problems():
    """base64(JSON) 로 실어 온다 — LaTeX 백슬래시가 어느 단계에서도 해석되지 않는다.
    이 저장소에서 heredoc·정규식 백슬래시로 파일을 깨뜨린 적이 **9번** 있다."""
    blob = globals().get("PROBS_B64")
    if not blob:
        raise SystemExit("PROBS_B64 가 없다 — --body entlens_problems.py,entlens.py 로 올려야 한다")
    rows = json.loads(base64.b64decode(blob).decode("utf-8"))
    #  #FLIP-ENT: 팔이 모델마다 다른 문제집합이라 dict 로도 실어 온다. 리스트면 기존 동작 그대로.
    if isinstance(rows, dict):
        if MODEL not in rows:
            raise SystemExit(f"PROBS_B64 dict 에 {MODEL} 키가 없다 — 키 {sorted(rows)}")
        rows = rows[MODEL]
    return rows[:NPROB] if NPROB else rows


# ───────────────────────── 모델 ─────────────────────────

def load_model():
    t = time.time()
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(MODEL, local_files_only=LOCAL)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=LOCAL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(local_files_only=LOCAL, low_cpu_mem_usage=True, device_map="auto",
              attn_implementation="sdpa")
    if QUANT == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["dtype"] = torch.float16                 # T4 는 Turing — **bf16 을 못 쓴다**
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    except Exception:
        kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    print(f"[entlens] {MODEL} 로드 {time.time() - t:.0f}s · quant {QUANT}", flush=True)
    return tok, model


def chat(tok, text, force=False):
    s = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}],
                                tokenize=False, add_generation_prompt=True)
    return s + FORCE if force else s


def final_norm(model):
    """logit lens 는 **마지막 정규화를 거친 뒤** 어휘로 투영한다. 생략하면 스케일이 어긋나
    확률이 의미를 잃는다. 모델마다 경로가 다르므로 찾아서 돌려준다."""
    for path in ("model.norm", "model.language_model.norm", "model.model.norm",
                 "transformer.ln_f", "model.final_layernorm"):
        obj, ok = model, True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and callable(obj):
            print(f"[entlens] final_norm = {path}", flush=True)
            return obj
    raise SystemExit("final_norm 을 못 찾았다 — 모델 구조를 확인하라")


# ───────────────────────── P2: 엔트로피 궤적 ─────────────────────────

class EntProc(LogitsProcessor):
    """스텝마다 전 어휘 엔트로피만 재서 쌓고 logits 는 **그대로** 돌려준다(생성을 바꾸지 않는다)."""

    def __init__(self, n):
        self.H = [[] for _ in range(n)]

    def __call__(self, input_ids, scores):
        lp = torch.log_softmax(scores.float(), dim=-1)
        h = -(lp.exp() * lp).sum(-1)
        for i, v in enumerate(h.tolist()):
            if i < len(self.H):
                self.H[i].append(float(v))
        return scores


def gen_chunk(tok, model, texts):
    """greedy 생성 + 토큰별 엔트로피. 반환: 행별 dict 목록."""
    prompts = [chat(tok, t) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    proc = EntProc(len(texts))
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                             logits_processor=LogitsProcessorList([proc]),
                             pad_token_id=tok.pad_token_id,
                             return_dict_in_generate=True)
    plen = enc["input_ids"].shape[1]
    seq = out.sequences[:, plen:]
    dt = time.time() - t0
    eos = set(x for x in [tok.eos_token_id] if x is not None)
    if getattr(tok, "eos_token_id", None) is not None:
        eos.add(tok.eos_token_id)
    gcfg = getattr(model, "generation_config", None)
    if gcfg is not None and gcfg.eos_token_id is not None:
        e = gcfg.eos_token_id
        eos.update(e if isinstance(e, (list, tuple)) else [e])
    rows = []
    for i in range(len(texts)):
        ids = seq[i].tolist()
        cut, fin = len(ids), False
        for j, v in enumerate(ids):
            if v in eos:
                cut, fin = j + 1, True
                break
        txt = tok.decode(ids[:cut], skip_special_tokens=True)
        rows.append(dict(n_tok=cut, finished=bool(fin), text=txt,
                         ent=[round(x, 6) for x in proc.H[i][:cut]],
                         n_marks=sum(txt.count(m) for m in MARKERS)))
    print(f"[entlens] 생성 {len(texts)}행 · {dt:.0f}s · {MAXNEW}스텝 · "
          f"{dt / max(1, MAXNEW) * 1000:.0f} ms/step · 완주 "
          f"{sum(r['finished'] for r in rows)}/{len(rows)}", flush=True)
    return rows


# ───────────────────────── P3: 강제 즉답 + 층별 logit lens ─────────────────────────

def lens_one(tok, model, norm, head, text):
    """반환 dict: ans · ans_int · pcurve(층별 확률) · gate(마지막층 vs 실제 로짓 차).

    **답의 첫 토큰은 모델이 실제로 낸 토큰 id 를 쓴다.** 로컬 스모크가 잡은 결함: 즉답이
    `" 12"` 인데 `strip()` 한 `"12"` 를 다시 토크나이즈하면 **다른 토큰**(선행 공백 유무)이 되고,
    그러면 '모델이 내지 않은 토큰' 의 확률을 층별로 추적하게 된다. 생성 id 를 그대로 쓰면
    재토크나이즈가 끼어들 자리가 없다.
    """
    prompt = chat(tok, text, force=True)
    pi = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        g = model.generate(pi, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    gen = g[0, pi.shape[1]:].tolist()
    ans_raw = tok.decode(gen, skip_special_tokens=True)
    #  첫 줄까지만 — 그 뒤는 풀이로 넘어간 부분이다
    cut = len(gen)
    for j, t in enumerate(gen):
        if "\n" in tok.decode([t]):
            cut = j
            break
    line = gen[:cut]
    ans = tok.decode(line, skip_special_tokens=True).strip()
    m = INT_RE.search(ans.replace(",", ""))
    ans_int = int(m.group()) if m else None
    #  답의 첫 토큰 = 숫자나 부호를 처음 담은 **생성 토큰**
    k = next((j for j, t in enumerate(line)
              if any(c.isdigit() or c == "-" for c in tok.decode([t]))), None)
    if k is None:
        return dict(ans=ans_raw, ans_int=ans_int, pcurve=None, gate=None,
                    note="첫 줄에 숫자 토큰이 없다")
    tgt = int(line[k])
    ids = torch.cat([pi, torch.tensor([line[:k + 1]], device=pi.device)], dim=1)
    pos = pi.shape[1] + k - 1              # 이 위치의 **다음** 토큰이 답의 첫 토큰이다
    with torch.no_grad():
        o = model(ids, output_hidden_states=True)
    hs = o.hidden_states
    L = len(hs) - 1
    pc = []
    #  **no_grad 안에서 돈다** — `head.weight` 가 requires_grad=True 라 밖에서 돌면 층마다
    #  (V=151k) 그래프가 쌓여 T4 에서 터질 수 있다(값은 같고 메모리만 다르다).
    with torch.no_grad():
        for li in range(L + 1):
            h = hs[li][0, pos, :]
            #  `hidden_states[-1]` 은 HF 에서 **이미 final_norm 을 거친** 값이다(LlamaModel.forward
            #  가 루프 뒤 self.norm 을 적용하고 그 결과를 마지막으로 넣는다) → 그 층만 다시
            #  정규화하지 않는다. 이 가정은 아래 `real` 과의 차이로 **계측해서** 확인한다
            #  (로컬 스모크 실측 최대차 **1.0e-10** → 가정이 맞다).
            hv = h if li == L else norm(h.unsqueeze(0)).squeeze(0)
            lg = head(hv.to(head.weight.dtype)).float()
            pc.append(float(torch.softmax(lg, dim=-1)[tgt]))
        real = float(torch.softmax(o.logits[0, pos, :].float(), dim=-1)[tgt])
    return dict(ans=ans_raw, ans_int=ans_int, tok_first=tgt, tok_text=tok.decode([tgt]),
                pcurve=[round(x, 8) for x in pc],
                gate=dict(p_last=round(pc[L], 8), p_real=round(real, 8),
                          diff=round(abs(pc[L] - real), 10)), note="")


# ───────────────────────── 본 실행 ─────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    probs = load_problems()
    tok, model = load_model()
    norm, head = final_norm(model), model.get_output_embeddings()
    alias = MODEL.split("/")[-1].replace(".", "_")
    nl = int(getattr(model.config, "num_hidden_layers", 0))
    print(f"[entlens] 문제 {len(probs)} · 층 {nl} · maxnew {MAXNEW} · batch {BATCH or len(probs)}",
          flush=True)

    # P2 — 배치 생성 (BATCH 로 쪼갠다. Olmo 는 KV 512 KB/토큰이라 한 배치가 안 들어간다)
    size = BATCH if BATCH else len(probs)
    rows = []
    for s in range(0, len(probs), size):
        part = probs[s:s + size]
        try:
            got = gen_chunk(tok, model, [p["problem"] for p in part])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[entlens] OOM → 배치 반감 재시도 ({len(part)} → {max(1, len(part) // 2)})",
                  flush=True)
            got = []
            h = max(1, len(part) // 2)
            for s2 in range(0, len(part), h):
                got += gen_chunk(tok, model, [p["problem"] for p in part[s2:s2 + h]])
        for p, r in zip(part, got):
            rows.append(dict(pidx=p["pidx"], problem=p["problem"], **r))
        (OUTDIR / f"{alias}_p2.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8")   # 중간 저장 (끊겨도 남는다)

    # P3 — 문제마다 강제 즉답 + 층별 렌즈
    t0, lens = time.time(), []
    for n, p in enumerate(probs, 1):
        try:
            d = lens_one(tok, model, norm, head, p["problem"])
        except Exception as e:                                       # noqa: BLE001
            d = dict(ans="", ans_int=None, pcurve=None, gate=None, note=f"예외 {type(e).__name__}")
        lens.append(dict(pidx=p["pidx"], **d))
        if n % 5 == 0 or n == len(probs):
            print(f"[entlens] 렌즈 {n}/{len(probs)} · {time.time() - t0:.0f}s", flush=True)
    (OUTDIR / f"{alias}_p3.json").write_text(json.dumps(lens, ensure_ascii=False), encoding="utf-8")

    gates = [r["gate"]["diff"] for r in lens if r.get("gate")]
    meta = dict(model=MODEL, n_layers=nl, maxnew=MAXNEW, batch=size, n_prob=len(probs),
                quant=QUANT, n_marks=sum(r["n_marks"] for r in rows),
                finished=sum(r["finished"] for r in rows),
                lens_ok=len(gates), lens_gate_max=(max(gates) if gates else None))
    (OUTDIR / f"{alias}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"[entlens] 저장 {OUTDIR} · meta {json.dumps(meta, ensure_ascii=False)}", flush=True)
    #  **게이트**: hidden_states[-1] 이 이미 정규화된 값이라는 가정을 계측으로 확인한다.
    #  틀렸으면 p_last 가 실제 로짓 확률과 어긋난다 → pcurve 전체를 믿을 수 없다.
    if gates and max(gates) > 1e-3:
        print(f"[entlens] **경고** 마지막층 렌즈가 실제 로짓과 {max(gates):.3e} 어긋난다 — "
              f"final_norm 가정을 다시 봐야 한다", flush=True)
    else:
        print(f"[entlens] 렌즈 게이트 통과 · 마지막층 vs 실제 로짓 최대차 "
              f"{(max(gates) if gates else float('nan')):.3e}", flush=True)
    #  상설 규칙 ② — 원문 5개를 로그에 찍는다. 이 로그를 읽은 뒤에만 평가로 넘어간다.
    for r in rows[:5]:
        print(f"\n===== pidx {r['pidx']} · n_tok {r['n_tok']} · 완주 {r['finished']} · "
              f"마커 {r['n_marks']} =====\n{r['text'][:900]}\n", flush=True)
    for r in lens[:5]:
        print(f"[즉답] pidx {r['pidx']} · ans={r['ans']!r} · int={r['ans_int']} · "
              f"note={r['note']}", flush=True)


if __name__ == "__main__":
    main()
