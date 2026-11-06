"""자체 라벨 생성 v2 — vLLM 생성 + HF 궤적 1회 forward. (지휘 #30, 2026-09-30)

왜 바꿨나: HF 토큰별 파이썬 루프가 병목이었다. 실측 Qwen3.5-4B 배치8 20.2 tok/s · Skywork 배치16 **16.1 tok/s** —
T4 2장 7B fp16이면 합산 수백 tok/s가 나와야 하므로 코드 병목이 확실. 그래서
  1단계 **vLLM**으로 텍스트만 생성(연속 배칭, 매 토큰 hidden 추출 없음)
  2단계 vLLM 종료 후 **HF로 '프롬프트+생성문' 전체를 teacher forcing forward 1회** 해서 L//2층 32토큰 청크 평균을 뽑는다.
        인과 마스크라 생성 시점의 hidden과 **같은 벡터**이고 비용은 forward 한 번.
  VRAM 충돌을 막으려 두 단계를 완전히 분리(vLLM 죽이고 → HF 로드).

문제 난이도: MATH **level 3~4**, max_new 6144. **EOS 없이 상한에 닿은 생성은 '틀림'이 아니라 '제외'** (지휘 #30).
변형: 공식 rename·typos·distract(주 라벨) + 비공식 reorder·notation·format(보조 라벨).
라벨: 그룹의 변형이 전부 정답 유지 -> robust · 하나라도 오답 -> spurious · 하나라도 미완주 -> 그 그룹은 None(제외).

러너: kaggle_h10b.py --header genlab_vllm_kaggle.py --body variants_official.py,variants_nonofficial.py,genlab_vllm.py
"""

import argparse
import io
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

MATH_SUBJECTS = ["algebra", "counting_and_probability", "intermediate_algebra", "number_theory", "prealgebra", "precalculus", "geometry"]
MATH_URL = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main/{s}/test-00000-of-00001.parquet"
V2_URL = "https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/data/train-00000-of-00001.parquet"
BOXED_INT = re.compile(chr(92) * 2 + r"boxed\s*\{\s*(-?[\d,]+)\s*\}")
CHUNK = 32


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read()


def load_public(n_problems, seed=0, levels=("Level 3", "Level 4")):
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
    print(f"문제 {len(out)}개 (후보 {len(rows)} · {'/'.join(levels)} · v2 중복 제거 후)", flush=True)
    return out


def plan_variants(text, pidx, n_off, n_unoff):
    jobs = []
    for kind, v in official_variants(text, seed=pidx)[:n_off]:                    # noqa: F821
        jobs.append((kind, v, 0, "official"))
    for kind, v, alt in variants_for(text)[:n_unoff]:                             # noqa: F821
        jobs.append((kind, v, 2 if kind == "format2" else (1 if alt else 0), "unofficial"))
    return jobs


def chat(tok, text, alt):
    sysmsg = SOLVE_PROMPT_ALT2 if alt == 2 else (SOLVE_PROMPT_ALT if alt else SOLVE_PROMPT)   # noqa: F821
    return tok.apply_chat_template([{"role": "user", "content": f"{sysmsg}\n\n{text}"}], tokenize=False, add_generation_prompt=True)


def finalize(s):
    """그룹별 라벨. 미완주 변형이 있으면 그 그룹은 None(제외)."""
    for group, key in (("official", "label_official"), ("unofficial", "label_unofficial")):
        vs = [v for v in s["variants"] if v["group"] == group]
        if len(vs) != s["plan"].get(group, 0) or not vs:      # plan 이 빈 문제(오답·미완주)는 라벨 없음
            s[key] = None
        elif any(v["ok"] is None for v in vs):
            s[key] = None
        else:
            s[key] = bool(all(v["ok"] for v in vs))
    s["label"] = s["label_official"]


# ---------------------------------------------------------------- 1단계: vLLM 생성
def vllm_stage(model_id, probs, max_new, n_off, n_unoff, max_seqs, out_json):
    import torch
    from vllm import LLM, SamplingParams
    tp = max(1, torch.cuda.device_count())
    print(f"[vLLM] {model_id} · tensor_parallel={tp} · max_num_seqs={max_seqs} · max_new={max_new}", flush=True)
    t0 = time.time()
    llm = LLM(model=model_id, dtype="half", tensor_parallel_size=tp, max_num_seqs=max_seqs,
              gpu_memory_utilization=0.90, enforce_eager=False, trust_remote_code=True, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    print(f"[vLLM] 로드 {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    outs = llm.generate([chat(tok, p["problem"], 0) for p in probs], sp)
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[1단계] 원문제 {len(probs)}개 · {time.time() - t1:.0f}s · **{ntok / (time.time() - t1):.1f} tok/s** · 토큰 {ntok}", flush=True)

    store = []
    for p, o in zip(probs, outs):
        g = o.outputs[0]
        fin = g.finish_reason == "stop"
        ans = extract_answer(g.text)                                              # noqa: F821
        store.append({"pidx": p["pidx"], "gold": p["gold"], "level": p["level"], "subject": p["subject"], "problem": p["problem"],
                      "base_ans": ans, "finished": fin, "correct": bool(fin and ans is not None and ans == p["gold"]),
                      "gen_tokens": len(g.token_ids), "gen_token_ids": list(g.token_ids), "prompt_tokens": len(o.prompt_token_ids),
                      "plan": {}, "variants": [], "label": None, "label_official": None, "label_unofficial": None})
    n_fin = sum(s["finished"] for s in store)
    print(f"  완주 {n_fin}/{len(store)} · 정답 {sum(s['correct'] for s in store)}/{len(store)} "
          f"(완주한 것 중 {sum(s['correct'] for s in store)}/{max(1, n_fin)})", flush=True)

    jobs, prompts = [], []
    for s in store:
        if not s["correct"]:
            continue
        plan = plan_variants(s["problem"], s["pidx"], n_off, n_unoff)
        s["plan"] = {"official": sum(1 for j in plan if j[3] == "official"), "unofficial": sum(1 for j in plan if j[3] == "unofficial")}
        for kind, text, alt, group in plan:
            jobs.append((s, kind, alt, group)); prompts.append(chat(tok, text, alt))
    if prompts:
        t2 = time.time()
        vouts = llm.generate(prompts, sp)
        vtok = sum(len(o.outputs[0].token_ids) for o in vouts)
        print(f"[2단계] 변형 {len(prompts)}건 · {time.time() - t2:.0f}s · **{vtok / (time.time() - t2):.1f} tok/s**", flush=True)
        for (s, kind, alt, group), o in zip(jobs, vouts):
            g = o.outputs[0]; fin = g.finish_reason == "stop"
            ans = extract_answer(g.text, alt=bool(alt))                           # noqa: F821
            s["variants"].append({"kind": kind, "group": group, "ans": ans, "finished": fin,
                                  "ok": None if not fin else bool(ans is not None and ans == s["gold"]), "tokens": len(g.token_ids)})
    for s in store:
        finalize(s)
    Path(out_json).write_text(json.dumps(store), encoding="utf-8")
    print(f"[vLLM] 저장 {out_json} · 총 {time.time() - t0:.0f}s", flush=True)
    del llm
    return store


# ---------------------------------------------------------------- 2단계: HF 궤적 1회 forward
def traj_stage(model_id, store, out_npz):
    """생성문을 teacher forcing 으로 한 번 통과시켜 L//2층 32토큰 청크 평균을 뽑는다 (생성 당시와 같은 벡터)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True).eval()
    layers = max((m for m in model.modules() if isinstance(m, torch.nn.ModuleList)), key=len)
    # 옛 커널은 hidden_states[len(hidden_states)//2] 를 썼다. hidden_states[i] = i-1번 층의 출력이므로
    # 같은 벡터를 얻으려면 layers[mid-1] 에 훅을 건다. (mid를 그대로 쓰면 한 층 어긋난다 — 로컬 검증에서 코사인 0.83)
    mid = (len(layers) + 1) // 2
    grab = {}
    h = layers[mid - 1].register_forward_hook(lambda m, a, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    t0 = time.time(); n = 0
    for s in store:
        gid = s.get("gen_token_ids") or []
        if not gid:
            s["chunks"] = np.zeros((0, 1), np.float16); continue
        try:
            pid = tok(chat(tok, s["problem"], 0), return_tensors="pt")["input_ids"][0].tolist()
            ids = torch.tensor([pid + list(gid)], device=model.device)
            with torch.no_grad():
                model(input_ids=ids, logits_to_keep=1)
            H = grab["h"][0, len(pid):].float()                   # 생성 토큰 자리만
            C = [H[i:i + CHUNK].mean(0) for i in range(0, len(H), CHUNK)]
            s["chunks"] = torch.stack(C).cpu().numpy().astype(np.float16) if C else np.zeros((0, H.shape[-1]), np.float16)
        except Exception as e:                                    # 한 문제가 터져도 나머지는 살린다
            print(f"  [궤적 실패] pidx={s['pidx']}: {type(e).__name__} {e}", flush=True)
            s["chunks"] = np.zeros((0, 1), np.float16)
        s.pop("gen_token_ids", None); n += 1
        if n % 20 == 0:
            print(f"  궤적 {n}/{len(store)} · {time.time() - t0:.0f}s", flush=True)
    h.remove()
    np.savez_compressed(out_npz, store=np.array(store, dtype=object), model=model_id, mid=mid)
    print(f"[궤적] {n}건 · {time.time() - t0:.0f}s · 저장 {out_npz}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=globals().get("RUN_MODELS", ["Skywork/Skywork-OR1-Math-7B"]))
    ap.add_argument("--out", default="/kaggle/working/genlab")
    ap.add_argument("--n", type=int, default=globals().get("RUN_N", 16))
    ap.add_argument("--max-new", type=int, default=globals().get("RUN_MAX_NEW", 6144))
    ap.add_argument("--max-seqs", type=int, default=globals().get("RUN_MAX_SEQS", 48))
    ap.add_argument("--n-official", type=int, default=globals().get("RUN_N_OFF", 3))
    ap.add_argument("--n-unofficial", type=int, default=globals().get("RUN_N_UNOFF", 2))
    ap.add_argument("--skip-traj", action="store_true")
    ap.add_argument("--levels", default=globals().get("RUN_LEVELS", "Level 3,Level 4"))
    ap.add_argument("--tag", default=globals().get("RUN_TAG", ""))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    probs = load_public(a.n, levels=tuple(x.strip() for x in re.split(r"[,|]", a.levels) if x.strip()))   # | 도 구분자 (러너 --extra 가 쉼표를 쓴다)
    for m in a.models:
        alias = m.split("/")[-1].replace(".", "_")
        js = out / f"{alias}{a.tag}_gen.json"
        store = vllm_stage(m, probs, a.max_new, a.n_official, a.n_unofficial, a.max_seqs, js)
        lo = [s for s in store if s["label_official"] is not None]
        lu = [s for s in store if s["label_unofficial"] is not None]
        both = [s for s in store if s["label_official"] is not None and s["label_unofficial"] is not None]
        ag = sum(s["label_official"] == s["label_unofficial"] for s in both)
        print(f"[라벨] {alias}: 공식 {len(lo)} (robust {sum(bool(s['label_official']) for s in lo)}) · "
              f"비공식 {len(lu)} (robust {sum(bool(s['label_unofficial']) for s in lu)}) · 겹치는 {len(both)}건 일치 {ag}", flush=True)
        if not a.skip_traj:
            print("[궤적] vLLM 정리 후 HF 로드", flush=True)
            os.system("nvidia-smi --query-gpu=memory.used --format=csv,noheader")
            traj_stage(m, store, out / f"{alias}{a.tag}.npz")


if __name__ == "__main__":
    main()
