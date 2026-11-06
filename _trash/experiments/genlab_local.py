"""genlab v2 로컬판 — Qwen3.5-4B 를 노트북 GPU(8GB, 4bit)에서 돌린다. (지휘 #35)

왜 로컬인가: Qwen3.5 는 하이브리드 어텐션이라 Kaggle 에서 vLLM 이 불안정하고 HF 폴백은 느려 캐글 GPU 시간을 낭비한다.
노트북 GPU 는 놀고 있으니 여기서 감독 없이 오래 돌린다.

설계는 genlab v2 사전등록과 동일:
  1. 선별: 원본 k회 샘플(T=0.6, top_p=0.95) -> base_acc, **완주(EOS)한 샘플만** 집계. 0.25 <= base_acc <= 0.75 채택
  2. 변형: 공식 3종(rename·typos·distract) 각 1개, 변형마다 k회 -> var_acc
  3. 라벨: max_drop = base_acc − min(var_acc) -> <=0.10 robust · >=0.25 spurious · 중간 제외
  4. 궤적: 채택 문제 greedy 1회 -> teacher forcing forward 1회로 L//2 32토큰 청크 평균 (layers[mid-1] 훅)

체크포인트: 작업 하나(문제 또는 변형) 끝날 때마다 state.json 저장, --resume 으로 이어감. 진행률은 progress 파일에 한 줄.
VRAM 부족(공유메모리 스필) 시 --batch 를 줄여 재시작하면 state 를 이어받는다.

사용:  .venv/Scripts/python.exe experiments/genlab_local.py --probe          (속도 측정만)
       .venv/Scripts/python.exe experiments/genlab_local.py --n 40 --k 4 --max-new 4096 --batch 2 --resume
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from variants_nonofficial import SOLVE_PROMPT, extract_answer  # noqa: E402
from variants_official import official_variants  # noqa: E402

RES = HERE / "results"
STATE = RES / "genlab_local_state.json"
PROG = RES / "genlab_local_progress.txt"
MODEL = "Qwen/Qwen3.5-4B"
AIME_URL = "https://huggingface.co/datasets/AI-MO/aimo-validation-aime/resolve/main/data/train-00000-of-00001.parquet"
MATH_URL = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main/{s}/test-00000-of-00001.parquet"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "intermediate_algebra", "number_theory", "prealgebra", "precalculus", "geometry"]
BOXED_INT = re.compile(chr(92) * 2 + r"boxed\s*\{\s*(-?[\d,]+)\s*\}")
CHUNK = 32
BAND = (0.25, 0.75)
DROP_ROBUST, DROP_SPUR = 0.10, 0.25


def prog(msg):
    line = f"[{time.strftime('%m-%d %H:%M')}] {msg}"
    print(line, flush=True)
    with PROG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read()


def load_pool(n, seed=0):
    """AIME + MATH level 5, 정수 정답, v2 36문제 제외. Kaggle 판과 같은 시드·순서."""
    import pandas as pd
    rows = []
    df = pd.read_parquet(io.BytesIO(_get(AIME_URL)))
    for r in df.itertuples():
        try:
            rows.append({"problem": r.problem.strip(), "gold": int(str(r.answer).strip()), "level": "AIME"})
        except Exception:
            pass
    for s in MATH_SUBJECTS:
        try:
            d = pd.read_parquet(io.BytesIO(_get(MATH_URL.format(s=s))))
        except Exception:
            continue
        for r in d.itertuples():
            if r.level != "Level 5":
                continue
            m = BOXED_INT.search(r.solution)
            if m:
                rows.append({"problem": r.problem.strip(), "gold": int(m.group(1).replace(",", "")), "level": "Level 5"})
    v2p = RES / "train-main-v2.parquet"
    v2 = set(p.strip() for p, _ in pd.read_parquet(v2p).groupby("problem", sort=False)) if v2p.is_file() else set()
    rows = [r for r in rows if r["problem"] not in v2]
    idx = np.random.default_rng(seed).permutation(len(rows))[:n]
    out = [rows[i] for i in sorted(idx)]
    for i, r in enumerate(out):
        r["pidx"] = i
    return out


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(MODEL)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=qc, device_map="auto", low_cpu_mem_usage=True).eval()
    return tok, model


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}], tokenize=False, add_generation_prompt=True)


def gen(tok, model, prompts, k, temp, max_new):
    """반환: 프롬프트별 [(텍스트, 완주여부)] * k, 생성 토큰 수 합."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=temp > 0, temperature=temp if temp > 0 else None,
                             top_p=0.95 if temp > 0 else None, num_return_sequences=k,
                             max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    res, ntok = [], 0
    for j in range(len(prompts)):
        grp = []
        for r in new[j * k:(j + 1) * k]:
            nz = int((r != tok.pad_token_id).sum()); ntok += nz
            grp.append((tok.decode(r, skip_special_tokens=True), bool((r == tok.eos_token_id).any())))
        res.append(grp)
    return res, ntok


def acc_of(grp, gold):
    fin = [t for t, f in grp if f]
    if not fin:
        return None
    return sum(1 for t in fin if extract_answer(t) == gold) / len(fin)


def save(state):
    STATE.write_text(json.dumps(state), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true", help="1문제 k=2 max_new=384 로 속도만 측정")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    tok, model = load_model()
    prog(f"모델 로드 완료 ({MODEL}, 4bit)")

    if a.probe:
        pool = load_pool(2)
        t0 = time.time()
        _, ntok = gen(tok, model, [chat(tok, p["problem"]) for p in pool[:1]], 2, 0.6, 384)
        el = time.time() - t0
        prog(f"[probe] 1문제 k=2 max_new=384 · {el:.0f}s · {ntok} 토큰 · {ntok / el:.1f} tok/s")
        for n, k, mx in ((40, 4, 4096), (30, 4, 4096), (40, 4, 3072)):
            est = (n * k + n * 0.3 * 3 * k) * mx / max(1.0, ntok / el) / 3600
            prog(f"[probe] 예상: N={n} k={k} max_new={mx} → 약 {est:.1f}h (채택률 30% 가정, 상한까지 생성 가정)")
        return

    state = json.loads(STATE.read_text(encoding="utf-8")) if (a.resume and STATE.is_file()) else {"screen": {}, "var": {}, "cfg": vars(a)}
    pool = load_pool(a.n)
    prog(f"문제풀 {len(pool)}개 · 선별 완료 {len(state['screen'])} · 변형 완료 {len(state['var'])}")

    # 1. 선별
    todo = [p for p in pool if str(p["pidx"]) not in state["screen"]]
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        try:
            groups, ntok = gen(tok, model, [chat(tok, p["problem"]) for p in chunk], a.k, 0.6, a.max_new)
        except Exception as e:
            prog(f"[선별 실패] pidx={[p['pidx'] for p in chunk]}: {type(e).__name__} {str(e)[:90]}")
            for p in chunk:
                state["screen"][str(p["pidx"])] = {"base_acc": None, "err": type(e).__name__}
            save(state); continue
        for p, grp in zip(chunk, groups):
            acc = acc_of(grp, p["gold"])
            state["screen"][str(p["pidx"])] = {"base_acc": acc, "n_fin": sum(1 for _, f in grp if f), "gold": p["gold"], "level": p["level"]}
        save(state)
        done = len(state["screen"]); el = time.time() - t0
        prog(f"선별 {done}/{len(pool)} · {el / 60:.0f}분 · {ntok / max(1, el):.1f} tok/s · "
             f"대역 채택 {sum(1 for v in state['screen'].values() if v['base_acc'] is not None and BAND[0] <= v['base_acc'] <= BAND[1])}")

    kept = [p for p in pool if (v := state["screen"].get(str(p["pidx"]))) and v["base_acc"] is not None and BAND[0] <= v["base_acc"] <= BAND[1]]
    prog(f"선별 끝 · 채택 {len(kept)}/{len(pool)}")

    # 2. 변형
    for p in kept:
        for kind, text in official_variants(p["problem"], seed=p["pidx"]):
            key = f"{p['pidx']}|{kind}"
            if key in state["var"]:
                continue
            try:
                groups, _ = gen(tok, model, [chat(tok, text)], a.k, 0.6, a.max_new)
                state["var"][key] = acc_of(groups[0], p["gold"])
            except Exception as e:
                prog(f"[변형 실패] {key}: {type(e).__name__}")
                state["var"][key] = None
            save(state)
        prog(f"변형 진행 · 문제 {p['pidx']} 완료 ({len(state['var'])}건 누적)")

    # 3. 라벨
    labeled = []
    for p in kept:
        base = state["screen"][str(p["pidx"])]["base_acc"]
        va = [v for k2, v in state["var"].items() if k2.startswith(f"{p['pidx']}|") and v is not None]
        if not va:
            continue
        drop = base - min(va)
        lab = True if drop <= DROP_ROBUST else (False if drop >= DROP_SPUR else None)
        labeled.append({**p, "base_acc": base, "max_drop": drop, "label": lab})
    prog(f"라벨 {sum(1 for s in labeled if s['label'] is not None)}건 "
         f"(robust {sum(1 for s in labeled if s['label'] is True)} / spurious {sum(1 for s in labeled if s['label'] is False)}) · 중간 제외 {sum(1 for s in labeled if s['label'] is None)}")

    # 4. 궤적 (greedy 1회 + teacher forcing)
    import torch
    for s in labeled:
        g, _ = gen(tok, model, [chat(tok, s["problem"])], 1, 0.0, a.max_new)
        s["greedy_text"] = g[0][0][0]
    layers = max((m for m in model.modules() if isinstance(m, torch.nn.ModuleList)), key=len)
    mid = (len(layers) + 1) // 2
    grab = {}
    h = layers[mid - 1].register_forward_hook(lambda m, ar, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    for s in labeled:
        try:
            pid = tok(chat(tok, s["problem"]), return_tensors="pt")["input_ids"][0].tolist()
            gid = tok(s["greedy_text"], add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                model(input_ids=torch.tensor([pid + gid], device=model.device))
            H = grab["h"][0, len(pid):].float()
            C = [H[j:j + CHUNK].mean(0) for j in range(0, len(H), CHUNK)]
            s["chunks"] = torch.stack(C).cpu().numpy().astype(np.float16) if C else np.zeros((0, 1), np.float16)
            s["prompt_tokens"] = len(pid); s["gen_tokens"] = len(gid)
        except Exception as e:
            prog(f"[궤적 실패] pidx={s['pidx']}: {type(e).__name__}")
            s["chunks"] = np.zeros((0, 1), np.float16)
        s["label_official"] = s["label"]; s["label_unofficial"] = None
    h.remove()
    out = RES / "genlab" / "Qwen3_5-4B_v2local.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, store=np.array(labeled, dtype=object), model=MODEL, mid=mid)
    prog(f"완료 · 저장 {out}")


if __name__ == "__main__":
    main()
