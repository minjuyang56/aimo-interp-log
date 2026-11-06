"""H10-B — 나침반, 풀이 생성 궤적 (Kaggle T4 fp16용 단일 스크립트). 준비만, 실행은 지휘 결정 후.

규칙 적합성: 라벨 생성이 아니다 — 변형 없음, 원문제만, 모델 본인이 풀이를 생성하는 동안의 내부 상태를 기록할 뿐.
             추가 라벨 데이터도, 검증셋과 같은 변형 유형도 만들지 않는다. (CLAUDE.md 메모)

무엇을 하나 (v2 36문제 × 4모델):
  1. 원문제를 solve 프롬프트로 넣고 greedy로 첫 MAX_NEW(=1024) 토큰 생성 (fp16, 배치 4 left-padding·명시적 position_ids;
     커널 시작 시 순차 vs 배치 토큰 일치율 자체검사 ≥0.9 아니면 순차 폴백. 문제마다 체크포인트 저장, --resume 으로 이어감)
  2. 매 생성 토큰의 L//2층·마지막층 벡터를 32토큰 청크 평균으로 저장 → (청크수, 2, D) + 청크 텍스트
  3. 프롬프트 forward에서 단어 4통(h9 규칙, MATH137로 만든 목록) 단위의 층별 벡터(균등 가중)와 문제토큰 층 평균도 저장
  4. 나침반-B = 청크 c마다 cos(구조액기스_{L//2}, 청크c 상태_{L//2}) − cos(표면액기스, 청크c) 를 마지막 청크 방향 기준으로;
     지표: 마지막 청크에서의 값(끝에서 구조를 향하나), 청크별 평균, 청크별 기울기(점점 구조로 도는가). 부호: 큼 = robust.
     계산은 로컬 h10b_eval.py 에서 (여기선 저장만).

출력: /kaggle/working/h10b/<모델별칭>_v2.npz  (store: pidx·y·chunks(C,2,D)·unit_layers(U,L+1,D)·buckets·layer_mean·T)

Kaggle: !pip install -U transformers==5.13.0 tokenizers accelerate safetensors pandas pyarrow
        !python h10b_trajectory.py   (모델당 커널 하나, kaggle_h10b.py 가 올림. VRAM 총량이 모자라면 load()가 8bit로 자동 전환)
"""

import argparse
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
HF_PARQUET = "https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/data/train-00000-of-00001.parquet"
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")
MAX_NEW, CHUNK = globals().get("RUN_MAX_NEW", 1024), globals().get("RUN_CHUNK", 32)   # 지휘 #19: 1024/32 · 전체 궤적판(#26-4)은 헤더가 8192/128로 덮음
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
# --- 단어 4통 (h9_essence.py 와 동일 목록; MATH137만 보고 작성) ---
V_REL = set("""divides divided divisible exceeds exceed equals equal satisfy satisfies satisfying intersect intersects lies lie
varies vary chosen choose take taken put shake shakes add added weigh weighs obtain rearrange express expressed compute
evaluate simplify simplified find solve solved defined define represented represent participate enter measure averaging
reversed pointing turns cubed inscribed having combined contains contain formed form placed place arranged arrange
selected select removed remove leaves leave depart departs differ differs multiply multiplied subtract subtracted
holds hold gives give receives receive tangent touches meets meet passes pass""".split())
N_MATH = set("""number numbers value values sum sums integer integers digit digits remainder square squares prime primes
multiple multiples root roots area areas length lengths triangle triangles circle circles circular radius diameter polygon
octagon rectangle rectangles cone cones sphere angle angles side sides point points line lines vertical asymptotes graph
base bases sequence sequences term terms expression equation equations solution solutions divisor divisors factor factors
units expansion difference product quotient ratio decimal fraction fractions constant symmetry axes axis interior pair
pairs distance width height leg legs parentheses placement operation addition subtraction multiplication congruent
equivalent perimeter vertex vertices midpoint segment parallel perpendicular circumradius inradius polynomial coefficient
function probability set subset element power exponent modulo floor ceiling gcd lcm coprime arithmetic geometric series
mean median mode average matrix vector determinant complex real rational irrational""".split())
Q_WORDS = set("""one two three four five six seven eight nine ten eleven twelve twenty hundred thousand million twice thrice once
each every all both single per exactly least most more less than greater smaller maximum minimum largest smallest greatest
half third quarter double triple total first second last consecutive distinct different possible many much some several
few none any positive negative odd even whole nonzero""".split())
STOP = set("""the a an of to in on at by for with and or but if then is are was were be been being this that these those it
its as from into over under between among which who whom whose where when while there here such so not no nor can could
would should may might will shall do does did have has had he she they we you i his her their our your let suppose assume
given denote consider note also only just own same other another what how why whether about after before during until
your answer enter express simplest""".split())


def units(text):
    out, covered = [], []
    for m in MATH_SPAN.finditer(text):
        out.append((m.start(), m.end(), "Q")); covered.append((m.start(), m.end()))
    inmath = lambda i: any(a <= i < b for a, b in covered)
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        if not inmath(m.start()): out.append((m.start(), m.end(), "Q"))
    for m in re.finditer(r"[A-Za-z]+", text):
        if inmath(m.start()): continue
        w = m.group(0).lower()
        if w in STOP or len(w) < 3: continue
        out.append((m.start(), m.end(), "V_rel" if w in V_REL else "N_math" if w in N_MATH else "Q" if w in Q_WORDS else "N_topic"))
    return sorted(out)


def problems():
    import pandas as pd
    cache = Path("train-main-v2.parquet")
    if not cache.is_file():
        cache.write_bytes(urllib.request.urlopen(urllib.request.Request(HF_PARQUET, headers={"User-Agent": "Mozilla/5.0"}), timeout=180).read())
    df = pd.read_parquet(cache); rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna(); y = -1 if len(lab) == 0 else int(lab.astype(bool).mean() >= 0.5)
        rows.append((i, p.rstrip(), y))      # 끝 공백 제거: Qwen3.5 템플릿이 본문 끝 \n을 지워 index 실패 (v2 2문제)
    return rows


SIZE_GB = {"Qwen/Qwen3.5-4B": 9.3, "Skywork/Skywork-OR1-Math-7B": 15.2, "allenai/Olmo-3-7B-Think": 14.6, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 16.4}


def load(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9 if torch.cuda.is_available() else 0
    need = SIZE_GB.get(model_id, 16.4) * 1.15 + 2.0          # 가중치 + KV·활성 여유
    kw = dict(device_map="auto", low_cpu_mem_usage=True)
    if total >= need:
        quant = "fp16/bf16"; kw["dtype"] = dtype
    else:                                                       # T4 16GB 하나에 7–8B fp16은 안 들어감 → 8bit
        from transformers import BitsAndBytesConfig
        quant = "8bit"; kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True); kw["dtype"] = dtype
    print(f"[load] {model_id}: VRAM 총 {total:.1f} GB, 필요 {need:.1f} GB → {quant}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    return tok, model


def locate(p0, text):
    """프롬프트 안 문제 본문 위치. 못 찾으면 None (크래시 대신 skip)."""
    for t in (text, text.rstrip(), text.strip()):
        i = p0.find(t)
        if i >= 0:
            return i, t
    return None, None


def gen_batch(tok, model, items, max_new, want_hidden=True):
    """items: [(pidx, text, y)] 를 한 배치로 greedy 생성 (left padding, 명시적 position_ids).
    반환: per-item dict(tokens, chunks(C,2,D), chunk_texts, unit_layers, buckets, T, span_sum(L+1,D), span_cnt, eos) 또는 None(locate 실패)."""
    import torch
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompts, metas = [], []
    for pidx, text, y in items:
        p0 = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}], tokenize=False, add_generation_prompt=True)
        base, t_used = locate(p0, text)
        if base is None:
            print(f"  [skip] pidx={pidx}: 프롬프트에서 본문 못 찾음", flush=True); metas.append(None); continue
        prompts.append(p0); metas.append((pidx, t_used, y, base))
    if not prompts:
        return [None] * len(items)
    enc = tok(prompts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs_all = enc.pop("offset_mapping").tolist()
    ids = enc["input_ids"].to(model.device); am = enc["attention_mask"].to(model.device)
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am, position_ids=pos, output_hidden_states=True, use_cache=True)
    hs = out.hidden_states; L = len(hs); mid = L // 2; B = ids.shape[0]
    results = []
    if want_hidden:
        ALL = torch.stack([h for h in hs]).float().cpu()          # (L, B, T, D)
    for bi, meta in enumerate([m for m in metas if m is not None]):
        pidx, text, y, base = meta; offs = offs_all[bi]; us = units(text)
        real = [i for i in range(len(offs)) if am[bi, i].item() == 1]
        r = {"pidx": pidx, "y": y, "T": len(real)}
        if want_hidden:
            uidx = [[i for i in real if offs[i][1] > base + s and offs[i][0] < base + e] for (s, e, _) in us]
            keep = [j for j, idx in enumerate(uidx) if idx]
            r["unit_layers"] = torch.stack([ALL[:, bi, uidx[j]].mean(1) for j in keep]).half().numpy() if keep else np.zeros((0, L, ALL.shape[3]), np.float16)
            r["buckets"] = np.array([us[j][2] for j in keep])
            span = [i for i in real if offs[i][1] > base and offs[i][0] < base + len(text)]
            r["span_sum"] = ALL[:, bi, span].sum(1).numpy() if span else np.zeros((L, ALL.shape[3]), np.float32); r["span_cnt"] = len(span)
        r["_bufs"] = []; r["tokens"] = []; r["eos"] = False
        results.append(r)
    past = out.past_key_values; next_ids = out.logits[:, -1].argmax(-1)
    done = torch.zeros(B, dtype=torch.bool, device=ids.device); cur_pos = pos[:, -1].clone()
    eos_ids = {tok.eos_token_id} | ({tok.pad_token_id} if tok.pad_token_id is not None else set())
    with torch.no_grad():
        for step in range(max_new):
            for bi in range(B):
                if not done[bi]:
                    results[bi]["tokens"].append(next_ids[bi].item())
            done = done | torch.tensor([t in eos_ids for t in next_ids.tolist()], device=ids.device)
            for bi in range(B):
                if done[bi] and not results[bi]["eos"] and results[bi]["tokens"] and results[bi]["tokens"][-1] in eos_ids:
                    results[bi]["eos"] = True; results[bi]["tokens"].pop()   # EOS 자체는 궤적에서 제외
            if bool(done.all()):
                break
            am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device=am.device)], 1); cur_pos = cur_pos + 1
            o = model(input_ids=next_ids[:, None], attention_mask=am, position_ids=cur_pos[:, None], past_key_values=past, use_cache=True, output_hidden_states=want_hidden)
            past = o.past_key_values
            if want_hidden:
                hm = o.hidden_states[mid][:, -1].float().cpu(); hl = o.hidden_states[-1][:, -1].float().cpu()
                for bi in range(B):
                    if not done[bi]:
                        results[bi]["_bufs"].append(torch.stack([hm[bi], hl[bi]]))
            next_ids = o.logits[:, -1].argmax(-1)
    # 청크 집계: _bufs[t] 는 tokens[t] 를 입력으로 넣었을 때의 상태 (tokens[t] 생성 직후)
    for r in results:
        bufs, toks = r.pop("_bufs"), r["tokens"]
        if want_hidden:
            n = min(len(bufs), len(toks)); chunks, texts_ = [], []
            for i in range(0, n, CHUNK):
                chunks.append(torch.stack(bufs[i:i + CHUNK]).mean(0)); texts_.append(tok.decode(toks[i:i + CHUNK]))
            r["chunks"] = torch.stack(chunks).half().numpy() if chunks else np.zeros((0, 2, hs[0].shape[-1]), np.float16)
            r["chunk_texts"] = np.array(texts_, dtype=object); r["gen_tokens"] = len(toks)
    it = iter(results)
    return [next(it) if m is not None else None for m in metas]


def batch_self_check(tok, model, items, n=24, thresh=0.9):
    """같은 두 문제를 순차(B=1)와 배치(B=2)로 n토큰 생성해 토큰 일치율 비교. 둘 다 ≥ thresh 면 배치 OK."""
    seq = [gen_batch(tok, model, [it], n, want_hidden=False)[0]["tokens"] for it in items]
    bat = [r["tokens"] for r in gen_batch(tok, model, items, n, want_hidden=False)]
    rates = [sum(a == b for a, b in zip(s, b_)) / max(1, len(s)) for s, b_ in zip(seq, bat)]
    ok = all(r >= thresh for r in rates)
    print(f"[batch self-check] 순차 vs 배치 토큰 일치율 {[round(r, 2) for r in rates]} → {'배치 사용' if ok else '순차로 폴백'}", flush=True)
    return ok


def run(model_id, out_dir, batch=4, resume_glob=None):
    import glob
    import torch
    tok, model = load(model_id); alias = model_id.split("/")[-1].replace(".", "_")
    out_path = Path(out_dir) / f"{alias}_v2.npz"
    store, layer_sum, layer_cnt, n_old, old_mean, t0 = [], None, 0, 0, None, time.time()
    if resume_glob:
        hits = sorted(glob.glob(resume_glob, recursive=True))
        if hits:
            d = np.load(hits[-1], allow_pickle=True); store = list(d["store"]); n_old = len(store)
            if "layer_sum" in d and d["layer_sum"].ndim == 2:
                layer_sum = torch.tensor(d["layer_sum"]); layer_cnt = int(d["layer_cnt"])
            else:
                old_mean = torch.tensor(d["layer_mean"]).float()   # 옛 파일: 문제 수 가중으로 근사 병합
            print(f"[resume] {hits[-1]}: 문제 {n_old}개 재사용 (pidx {[int(r['pidx']) for r in store][:5]}…)", flush=True)
    have = {int(r["pidx"]) for r in store}
    todo = [it for it in problems() if it[0] not in have]
    print(f"[run] {alias}: 남은 문제 {len(todo)}/36, 요청 배치 {batch}", flush=True)
    if batch > 1 and len(todo) >= 2:
        batch = batch if batch_self_check(tok, model, todo[:2]) else 1

    def save():
        lm = layer_sum / max(1, layer_cnt) if layer_sum is not None else None
        if old_mean is not None:
            n_new = len(store) - n_old
            lm = old_mean if lm is None else (old_mean * n_old + lm * n_new) / max(1, n_old + n_new)
        np.savez_compressed(out_path, store=np.array(store, dtype=object), layer_mean=lm.half().numpy() if lm is not None else np.zeros(1, np.float16),
                            layer_sum=layer_sum.numpy() if layer_sum is not None else np.zeros(1, np.float32), layer_cnt=layer_cnt, model=model_id, batch=batch)
    k = n_old
    for i in range(0, len(todo), batch):
        items = todo[i:i + batch]
        try:
            res = gen_batch(tok, model, items, MAX_NEW)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print("  [OOM] 배치 → 순차", flush=True); batch = 1
            res = [gen_batch(tok, model, [it], MAX_NEW)[0] for it in items]
        for it, r in zip(items, res):
            k += 1
            if r is None:
                continue
            ss = torch.tensor(r.pop("span_sum")); sc = r.pop("span_cnt")
            layer_sum = ss if layer_sum is None else layer_sum + ss; layer_cnt += sc
            store.append(r)
            print(f"  [{alias}] {k}/36 gen={r['gen_tokens']} chunks={len(r['chunks'])} units={len(r['buckets'])} eos={r['eos']}  {time.time() - t0:.0f}s", flush=True)
        save()
    save(); print(f"[done] {alias}: 저장 {len(store)}문제 {time.time() - t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=globals().get("RUN_MODELS", MODELS)); ap.add_argument("--out", default="/kaggle/working/h10b")
    ap.add_argument("--batch", type=int, default=globals().get("RUN_BATCH", 4)); ap.add_argument("--resume", default=globals().get("RUN_RESUME", None))
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    for m in a.models:
        run(m, a.out, a.batch, a.resume)
