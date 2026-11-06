"""제출 compass (Small 트랙) — 풀이 생성 궤적 나침반, 모델별 고정 임계값. 사전등록 2026-09-29 (지휘 #26-2).

무엇을 하나
  판정 대상 model_id를 그대로 로드(bf16) → 문제들을 배치 greedy로 MAX_NEW 토큰 생성하며 L//2층 마지막 토큰 벡터를
  CHUNK 토큰 청크 평균으로 기록 → 문제마다 (compass_mean, traced_ratio, curvature) →
  combo = z(compass_mean) + z(traced_ratio) − z(curvature) (z 통계는 train v2에서 고정, artifacts/)
  → combo ≥ 모델별 고정 임계값(train v2 모델별 combo 분포의 사전지정 분위수 Q, 아티팩트) 이면 True.
  배치 내 상대 비교(중앙값 분할) 없음 — 최종셋이 불균형일 수 있다는 주최측 답변(9/28).

안전장치
  전역 시계(import 시점) · 모델당 예산 PER_MODEL_BUDGET · 예산 부족 시 남은 문제 전부 True · OOM 시 배치 반감 ·
  미지 model_id(아티팩트 없음) → 전부 True · 모든 예외 → True.
런타임(추정): 워커 RTX PRO 6000, 7–8B bf16 배치 8 × 1024토큰 ≈ 60–90 s/배치. 배치 ≈ 5–10문제/모델이면 모델당 1–3분.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
TOTAL_BUDGET = 3600.0
MARGIN = 120.0
PER_MODEL_BUDGET = float(os.environ.get("COMPASS_MODEL_BUDGET", 600))
MAX_NEW = int(os.environ.get("COMPASS_MAX_NEW", 1024))
CHUNK = 32
BATCH = int(os.environ.get("COMPASS_BATCH", 8))
FALLBACK = True
QUANT = os.environ.get("COMPASS_QUANT", "none")     # 실전은 none(bf16). 로컬 8GB 측정용으로만 4bit.


def log(msg):
    """채점기는 stdout을 쓰므로 stderr로만 남긴다."""
    print(f"[compass {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)
ART = Path(__file__).resolve().parent / "artifacts" / "compass_thresholds.npz"
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")


def remaining():
    return TOTAL_BUDGET - MARGIN - (time.time() - T0)


def load_artifacts(model_id):
    """아티팩트: 모델별 mu(청크 중심, D) · z 통계(cm/tr/cv 평균·표준편차) · 임계값. 없으면 None."""
    d = np.load(ART, allow_pickle=True)
    table = d["table"].item()
    return table.get(model_id)


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto" if cuda else None,
              dtype=torch.bfloat16 if cuda else torch.float32)
    if QUANT == "4bit" and cuda:                      # 로컬 측정 전용 (VRAM 8GB)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    return tok, model


def metrics(C):
    n = len(C)
    if n < 3:
        return None
    nrm = np.linalg.norm(C, axis=1) + 1e-8
    cos = (C @ C[-1]) / (nrm * nrm[-1])
    d = np.diff(C, axis=0); dn = np.linalg.norm(d, axis=1)
    ratio = float(np.linalg.norm(C[-1] - C[0]) / (dn.sum() + 1e-8))
    cc = (d[:-1] * d[1:]).sum(1) / (dn[:-1] * dn[1:] + 1e-8)
    return float(cos.mean()), ratio, float((1 - cc).mean())


def gen_chunks(tok, model, texts, max_new):
    """배치 greedy 생성. 반환: 문제별 (n_chunks, D) L//2 청크 평균 벡터 리스트."""
    import torch
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}], tokenize=False, add_generation_prompt=True) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(model.device); am = enc["attention_mask"].to(model.device)
    pos = (am.cumsum(-1) - 1).clamp(min=0); B = ids.shape[0]
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
    L = model.config.num_hidden_layers + 1 if hasattr(model.config, "num_hidden_layers") else None
    past = out.past_key_values; next_ids = out.logits[:, -1].argmax(-1)
    done = torch.zeros(B, dtype=torch.bool, device=ids.device); cur = pos[:, -1].clone()
    eos = {tok.eos_token_id, tok.pad_token_id}
    bufs = [[] for _ in range(B)]; chunks = [[] for _ in range(B)]
    with torch.no_grad():
        for _ in range(max_new):
            done = done | torch.tensor([t in eos for t in next_ids.tolist()], device=ids.device)
            if bool(done.all()) or remaining() < 30:
                break
            am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device=am.device)], 1); cur = cur + 1
            o = model(input_ids=next_ids[:, None], attention_mask=am, position_ids=cur[:, None], past_key_values=past, use_cache=True,
                      output_hidden_states=True, logits_to_keep=1)
            past = o.past_key_values; mid = len(o.hidden_states) // 2
            hm = o.hidden_states[mid][:, -1].float().cpu()
            for b in range(B):
                if not done[b]:
                    bufs[b].append(hm[b])
                    if len(bufs[b]) == CHUNK:
                        chunks[b].append(torch.stack(bufs[b]).mean(0)); bufs[b] = []
            next_ids = o.logits[:, -1].argmax(-1)
    res = []
    for b in range(B):
        if bufs[b]:
            chunks[b].append(torch.stack(bufs[b]).mean(0))
        res.append(torch.stack(chunks[b]).numpy() if chunks[b] else np.zeros((0, hm.shape[-1]), np.float32))
    return res


def decide(model_id, problems):
    """반환: list[bool]. 아티팩트 없는 모델·예산 초과·예외 → FALLBACK."""
    import torch
    art = load_artifacts(model_id)
    if art is None:
        return [FALLBACK] * len(problems)
    budget = min(PER_MODEL_BUDGET, remaining())
    if budget < 60:
        return [FALLBACK] * len(problems)
    t_start = time.time()
    log(f"{model_id} 로드 시작 (quant={QUANT}, 예산 {budget:.0f}s, 문제 {len(problems)})")
    tok, model = load_model(model_id)
    t_load = time.time() - t_start
    log(f"로드 완료 {t_load:.1f}s")
    mu = np.asarray(art["mu"], np.float32); zs = art["z"]; thr = float(art["threshold"])
    out = [FALLBACK] * len(problems); batch = BATCH; i = 0
    while i < len(problems):
        if time.time() - t_start > budget or remaining() < 60:
            break                                   # 남은 문제는 FALLBACK
        texts = problems[i:i + batch]
        try:
            ch = gen_chunks(tok, model, texts, MAX_NEW)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch > 1:
                batch = max(1, batch // 2); continue
            i += 1; continue
        log(f"배치 {i}~{i + len(texts) - 1} 생성 완료 ({(time.time() - t_start - t_load) / max(1, i + len(texts)):.1f}s/문제)")
        for j, C in enumerate(ch):
            m = metrics(C - mu) if len(C) >= 3 else None
            if m is None:
                continue
            cm, tr, cv = m
            combo = (cm - zs["cm"][0]) / zs["cm"][1] + (tr - zs["tr"][0]) / zs["tr"][1] - (cv - zs["cv"][0]) / zs["cv"][1]
            out[i + j] = bool(combo >= thr)
        i += len(texts)
    n_done = sum(1 for x in out if x is not FALLBACK) if False else i
    log(f"판정 {n_done}/{len(problems)} · 로드 {t_load:.1f}s · 총 {time.time() - t_start:.1f}s "
        f"· 문제당 {(time.time() - t_start - t_load) / max(1, n_done):.1f}s · 남은 문제는 기본값 {FALLBACK}")
    del model; torch.cuda.empty_cache()
    return out
