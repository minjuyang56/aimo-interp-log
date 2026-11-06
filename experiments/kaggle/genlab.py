"""자체 라벨 생성 (Kaggle 본체) — 공개 MATH level 4~5 정수답 문제에 라벨을 붙인다. (지휘 #28 → #29 개정)

규칙 근거 (지휘 #29): 공식 사이트 Competition Rules → Data, [Clarification 19 Sep]
  "We retired the rule that teams may not use extra labeled data. We disincentivise large-scale test-time
   perturbation generation through the inference time limit, but such approaches are not explicitly disallowed."
→ 추가 라벨 데이터 금지 규칙은 폐지. 금지는 평가 데이터 유출뿐. 그래서 **공식 유형 변형(rename·typos·distract)을 주 라벨**로,
  비공식(reorder·notation·format)은 보조 라벨로 따로 만든다. 두 라벨의 일치도도 같이 기록한다.

러너가 [genlab_kaggle.py(헤더) + variants_official.py + variants_nonofficial.py + 이 파일] 을 이어 붙여 kernel.py 로 올린다.

절차 (모델당 커널 하나)
  1. 문제: MATH test level 4~5, 해답 \\boxed 가 정수 → v2 36문제와 중복 제거 → seed 0 셔플 → N개
  2. 원문제 greedy 생성(max MAX_NEW) + L//2층 32토큰 청크 궤적 저장(H10-B 재사용)
  3. 원문제를 맞힌 문제만 라벨 대상. 공식 변형 N_OFF개 · 비공식 변형 N_UNOFF개 각 1회 생성
  4. label_official  = 공식 변형 전부 답 유지 → True, 하나라도 붕괴/추출 실패 → False
     label_unofficial= 비공식 변형 전부 답 유지 → True, …
     label           = label_official (주 라벨)
  5. 배치마다 체크포인트 저장, --resume 으로 이어감
"""

import argparse
import io
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

MATH_SUBJECTS = ["algebra", "counting_and_probability", "intermediate_algebra", "number_theory", "prealgebra", "precalculus", "geometry"]
MATH_URL = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main/{s}/test-00000-of-00001.parquet"
V2_URL = "https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/data/train-00000-of-00001.parquet"
BOXED_INT = re.compile(chr(92) * 2 + r"boxed\s*\{\s*(-?[\d,]+)\s*\}")
SIZE_GB = {"Qwen/Qwen3.5-4B": 9.3, "Skywork/Skywork-OR1-Math-7B": 15.2, "allenai/Olmo-3-7B-Think": 14.6, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 16.4}
CHUNK = 32
OFFICIAL_ORDER = ["rename", "typos", "distract"]


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read()


def load_public(n_problems, seed=0, levels=("Level 4", "Level 5")):
    import pandas as pd
    rows = []
    for s in MATH_SUBJECTS:
        try:
            df = pd.read_parquet(io.BytesIO(_get(MATH_URL.format(s=s))))
        except Exception as e:
            print(f"  [skip] {s}: {type(e).__name__}", flush=True); continue
        for r in df.itertuples():
            if r.level not in levels:
                continue
            m = BOXED_INT.search(r.solution)
            if not m:
                continue
            rows.append({"problem": r.problem.strip(), "gold": int(m.group(1).replace(",", "")), "level": r.level, "subject": s})
    cache = Path("train-main-v2.parquet")
    if not cache.is_file():
        cache.write_bytes(_get(V2_URL))
    v2 = set(p.strip() for p, _ in pd.read_parquet(cache).groupby("problem", sort=False))
    rows = [r for r in rows if r["problem"] not in v2]
    idx = np.random.default_rng(seed).permutation(len(rows))[:n_problems]
    out = [rows[i] for i in sorted(idx)]
    for i, r in enumerate(out):
        r["pidx"] = i
    print(f"문제 {len(out)}개 (후보 {len(rows)} · v2 중복 제거 후)", flush=True)
    return out


def plan_variants(text, pidx, n_off, n_unoff):
    """반환: [(kind, text, alt, group)] — 공식 n_off개 + 비공식 n_unoff개."""
    jobs = []
    for kind, v in official_variants(text, seed=pidx)[:n_off]:                    # noqa: F821 variants_official
        jobs.append((kind, v, 0, "official"))
    for kind, v, alt in variants_for(text)[:n_unoff]:                             # noqa: F821 variants_nonofficial
        jobs.append((kind, v, 2 if kind == "format2" else (1 if alt else 0), "unofficial"))
    return jobs


def finalize(s):
    """공식·비공식 라벨을 계획 개수가 다 채워졌을 때만 확정."""
    for group, key in (("official", "label_official"), ("unofficial", "label_unofficial")):
        vs = [v for v in s["variants"] if v["group"] == group]
        s[key] = bool(all(v["ok"] for v in vs)) if len(vs) == s["plan"][group] and vs else None
    s["label"] = s["label_official"]


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9 if torch.cuda.is_available() else 0
    need = SIZE_GB.get(model_id, 16.4) * 1.15 + 3.0
    kw = dict(device_map="auto", low_cpu_mem_usage=True, dtype=dtype)
    if total < need:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print(f"[load] VRAM {total:.1f} GB < 필요 {need:.1f} GB → 8bit", flush=True)
    else:
        print(f"[load] VRAM {total:.1f} GB → fp16/bf16", flush=True)
    return tok, AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()


def generate(tok, model, prompts, max_new, want_traj, deadline=None):
    """배치 greedy. want_traj면 L//2층 마지막 토큰의 CHUNK 평균만 CPU로 옮긴다(전송 32배 절약)."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(model.device); am0 = enc["attention_mask"].to(model.device)
    B, T0 = ids.shape
    pos = (am0.cumsum(-1) - 1).clamp(min=0)
    am = torch.zeros(B, T0 + max_new, dtype=am0.dtype, device=am0.device); am[:, :T0] = am0
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am0, position_ids=pos, use_cache=True, logits_to_keep=1)
    past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1)
    cur = pos[:, -1].clone()
    eos_ids = {tok.eos_token_id} | ({tok.pad_token_id} if tok.pad_token_id is not None else set())
    alive = torch.ones(B, dtype=torch.bool, device=ids.device)
    toks = [[] for _ in range(B)]; got_eos = [False] * B
    chunks = [[] for _ in range(B)]; acc = None; cnt = 0
    with torch.no_grad():
        for step in range(max_new):
            hit = torch.tensor([t in eos_ids for t in nxt.tolist()], device=ids.device)
            for b in range(B):
                if alive[b] and hit[b]:
                    got_eos[b] = True
                elif alive[b]:
                    toks[b].append(int(nxt[b]))
            alive = alive & ~hit
            if not bool(alive.any()) or (deadline and time.time() > deadline):
                break
            am[:, T0 + step] = 1; cur = cur + 1
            o = model(input_ids=nxt[:, None], attention_mask=am[:, :T0 + step + 1], position_ids=cur[:, None],
                      past_key_values=past, use_cache=True, output_hidden_states=want_traj, logits_to_keep=1)
            past = o.past_key_values
            if want_traj:
                h = o.hidden_states[len(o.hidden_states) // 2][:, -1] * alive[:, None]
                acc = h.float() if acc is None else acc + h.float()
                cnt += 1
                if cnt == CHUNK:
                    c = (acc / CHUNK).cpu()
                    for b in range(B):
                        chunks[b].append(c[b])
                    acc = None; cnt = 0
            nxt = o.logits[:, -1].argmax(-1)
    if want_traj and cnt:
        c = (acc / cnt).cpu()
        for b in range(B):
            chunks[b].append(c[b])
    res = []
    for b in range(B):
        ch = torch.stack(chunks[b]).numpy().astype(np.float16) if want_traj and chunks[b] else None
        res.append((tok.decode(toks[b], skip_special_tokens=True), len(toks[b]), got_eos[b], ch))
    return res


def prompt_of(tok, text, alt):
    sysmsg = SOLVE_PROMPT_ALT2 if alt == 2 else (SOLVE_PROMPT_ALT if alt else SOLVE_PROMPT)   # noqa: F821
    return tok.apply_chat_template([{"role": "user", "content": f"{sysmsg}\n\n{text}"}], tokenize=False, add_generation_prompt=True)


def run(model_id, out_dir, n_problems, max_new, batch, resume=None, hours=None, n_off=3, n_unoff=2):
    import torch
    alias = model_id.split("/")[-1].replace(".", "_")
    out_path = Path(out_dir) / f"{alias}.npz"
    deadline = time.time() + hours * 3600 if hours else None
    probs = load_public(n_problems)
    store = []
    if resume:
        import glob
        hits = sorted(glob.glob(resume, recursive=True))
        if hits:
            store = list(np.load(hits[-1], allow_pickle=True)["store"]); print(f"[resume] {hits[-1]}: {len(store)}문제", flush=True)
    have = {int(r["pidx"]) for r in store}
    tok, model = load_model(model_id); t0 = time.time()

    def save():
        np.savez_compressed(out_path, store=np.array(store, dtype=object), model=model_id, n=n_problems,
                            max_new=max_new, n_off=n_off, n_unoff=n_unoff)

    # ---- 1단계: 원문제 + 궤적
    todo = [p for p in probs if p["pidx"] not in have]
    print(f"[1단계] 원문제 {len(todo)}개 · 배치 {batch} · max_new {max_new} · 변형 공식 {n_off} + 비공식 {n_unoff}", flush=True)
    for i in range(0, len(todo), batch):
        if deadline and time.time() > deadline:
            print("  [예산] 1단계 중단", flush=True); break
        chunk = todo[i:i + batch]
        res = generate(tok, model, [prompt_of(tok, p["problem"], 0) for p in chunk], max_new, True, deadline)
        for p, (text, ntok, eos, ch) in zip(chunk, res):
            ans = extract_answer(text)                                                   # noqa: F821
            plan = plan_variants(p["problem"], p["pidx"], n_off, n_unoff)
            store.append({"pidx": p["pidx"], "gold": p["gold"], "level": p["level"], "subject": p["subject"],
                          "problem": p["problem"], "base_ans": ans, "correct": ans is not None and ans == p["gold"],
                          "chunks": ch if ch is not None else np.zeros((0, 1), np.float16), "gen_tokens": ntok, "eos": eos,
                          "plan": {"official": sum(1 for j in plan if j[3] == "official"), "unofficial": sum(1 for j in plan if j[3] == "unofficial")},
                          "variants": [], "label": None, "label_official": None, "label_unofficial": None})
        save()
        el = time.time() - t0
        print(f"  {i + len(chunk)}/{len(todo)} · 맞힘 {sum(s['correct'] for s in store)}/{len(store)} · "
              f"{el:.0f}s · {sum(s['gen_tokens'] for s in store) / max(1, el):.1f} tok/s · eos {sum(s['eos'] for s in store)}/{len(store)}", flush=True)

    # ---- 2단계: 변형
    jobs = []
    for s in store:
        if not s["correct"]:
            continue
        done = {(v["kind"]) for v in s["variants"]}
        for kind, text, alt, group in plan_variants(s["problem"], s["pidx"], n_off, n_unoff):
            if kind not in done:
                jobs.append((s, kind, text, alt, group))
    print(f"[2단계] 변형 생성 {len(jobs)}건", flush=True)
    tok_v = 0; t_v = time.time()
    for i in range(0, len(jobs), batch):
        if deadline and time.time() > deadline:
            print("  [예산] 2단계 중단", flush=True); break
        chunk = jobs[i:i + batch]
        res = generate(tok, model, [prompt_of(tok, t, alt) for _, _, t, alt, _ in chunk], max_new, False, deadline)
        for (s, kind, _, alt, group), (text, ntok, eos, _) in zip(chunk, res):
            ans = extract_answer(text, alt=bool(alt))                                     # noqa: F821
            s["variants"].append({"kind": kind, "group": group, "ans": ans, "ok": ans is not None and ans == s["gold"], "tokens": ntok, "eos": eos})
            finalize(s); tok_v += ntok
        save()
        print(f"  {i + len(chunk)}/{len(jobs)} · {time.time() - t0:.0f}s · 변형 {tok_v / max(1, time.time() - t_v):.1f} tok/s", flush=True)
    for s in store:
        finalize(s)
    save()
    lo = [s for s in store if s["label_official"] is not None]
    lu = [s for s in store if s["label_unofficial"] is not None]
    both = [s for s in store if s["label_official"] is not None and s["label_unofficial"] is not None]
    agree = sum(s["label_official"] == s["label_unofficial"] for s in both)
    print(f"[done] {alias}: 문제 {len(store)} · 맞힘 {sum(s['correct'] for s in store)} · "
          f"공식 라벨 {len(lo)} (robust {sum(bool(s['label_official']) for s in lo)}) · "
          f"비공식 라벨 {len(lu)} (robust {sum(bool(s['label_unofficial']) for s in lu)}) · "
          f"둘 다 있는 {len(both)}건 중 일치 {agree} ({agree / max(1, len(both)):.0%}) · {time.time() - t0:.0f}s", flush=True)
    for s in lo[:5]:
        print("   ", s["pidx"], "gold", s["gold"], [(v["kind"], v["ans"]) for v in s["variants"]],
              "→ 공식", s["label_official"], "비공식", s["label_unofficial"], flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=globals().get("RUN_MODELS", ["Qwen/Qwen3.5-4B"]))
    ap.add_argument("--out", default="/kaggle/working/genlab")
    ap.add_argument("--n", type=int, default=globals().get("RUN_N", 100))
    ap.add_argument("--max-new", type=int, default=globals().get("RUN_MAX_NEW", 4096))
    ap.add_argument("--batch", type=int, default=globals().get("RUN_BATCH", 8))
    ap.add_argument("--resume", default=globals().get("RUN_RESUME", None))
    ap.add_argument("--hours", type=float, default=globals().get("RUN_HOURS", None))
    ap.add_argument("--n-official", type=int, default=globals().get("RUN_N_OFF", 3))
    ap.add_argument("--n-unofficial", type=int, default=globals().get("RUN_N_UNOFF", 2))
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    for m in a.models:
        run(m, a.out, a.n, a.max_new, a.batch, a.resume, a.hours, a.n_official, a.n_unofficial)
