"""H21 — 짧은 예산으로 재는 **모델 자신의 불확실성** 3종. 로컬 4bit, 모델 하나씩. (지휘 #65 사전등록)

왜: 라벨은 "이 문제가 어려운가" 가 아니라 "이 모델이 이 문제를 진짜 아는가" 다(모델×문제).
    그러면 신호는 문제 텍스트가 아니라 그 모델 자신의 확신이어야 한다. H20 의 생성 길이는 그 대리지표였고
    값을 얻는 데 만 토큰이 들어 예산을 못 샀다. 여기서는 **1024토큰 또는 forward 몇 번**으로 끝낸다.

사전등록 부호 (결과 보고 뒤집지 않는다)
  H21a mean_entropy  : 생성 토큰 엔트로피 평균이 **클수록 spurious**        (점수 = −entropy)
  H21b ans_logprob   : 강제 즉답 토큰의 평균 logprob 이 **낮을수록 spurious** (점수 = +logprob)
  H21c drop          : logp(즉답|원본) − logp(즉답|상수변형) 이 **클수록 robust** (점수 = +drop)
      근거: 상수가 바뀌면 답도 바뀌어야 한다. 진짜 푸는 모델은 옛 답을 버리고(drop 큼),
            암기·운이면 상수와 무관하게 같은 답을 계속 높게 본다(drop 작음).

사용:
  .venv/Scripts/python.exe experiments/h21_uncertainty.py --stage conf --model Qwen/Qwen3.5-4B
  .venv/Scripts/python.exe experiments/h21_uncertainty.py --stage gen  --model Qwen/Qwen3.5-4B
결과: experiments/results/h21/<별칭>_<stage>.json
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "h21"
V2 = ROOT / "experiments" / "results" / "train-main-v2.parquet"
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
FORCE = "</think>\nANSWER:"
NUM_RE = re.compile(r"(?<![\w.])([2-9]|[1-9]\d{1,3})(?![\w.])")      # 0·1 과 긴 수는 건드리지 않는다


def problems():
    """문제 단위 라벨(다수결) + 모델별 라벨을 같이 돌려준다."""
    import pandas as pd
    df = pd.read_parquet(V2)
    rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna()
        if len(lab) == 0:
            continue
        per = {str(r.model_id): bool(r.is_robust) for r in g.itertuples()
               if r.is_robust is not None and str(r.is_robust) != "nan"}
        rows.append({"pidx": i, "problem": str(p).rstrip(),
                     "y": int(bool(lab.astype(bool).mean() >= 0.5)), "per_model": per})
    return rows


def perturb_constant(text):
    """상수 하나만 결정적으로 바꾼다(주최측 섭동 성격과 같은 종류). 바꿀 수 없으면 None.
    규칙: 마지막에 등장하는 2~4자리 수(또는 2~9)를 +1 한다. 0·1 과 5자리 이상은 건드리지 않는다
    (0/1 은 수식 구조를 바꾸기 쉽고, 긴 수는 연도·식별자일 확률이 높다)."""
    ms = list(NUM_RE.finditer(text))
    if not ms:
        return None
    m = ms[-1]
    return text[:m.start()] + str(int(m.group(1)) + 1) + text[m.end():]


def load(model_id, quant):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    t = time.time()
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa")
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    except Exception:
        kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    print(f"[h21] {model_id} 로드 {time.time() - t:.0f}s", flush=True)
    return tok, model


def chat(tok, text, force=False):
    s = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}],
                                tokenize=False, add_generation_prompt=True)
    return s + FORCE if force else s


def score_continuation(tok, model, prompt, cont):
    """logp(cont | prompt) 의 **토큰 평균**. teacher forcing forward 1회."""
    import torch
    pi = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ci = tok(cont, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ci.shape[1] == 0:
        return None
    ids = torch.cat([pi, ci], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, pi.shape[1] - 1:-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    tgt = ci[0].to(lp.device)
    return float(lp[torch.arange(len(tgt)), tgt].mean())


def stage_conf(tok, model, rows, args):
    """H21b 즉답 확신 + H21c 상수변형에서의 옛 답 하락."""
    import torch
    out = []
    for n, r in enumerate(rows, 1):
        rec = {"pidx": r["pidx"], "y": r["y"], "per_model": r["per_model"], "problem": r["problem"]}
        p0 = chat(tok, r["problem"], force=True)
        enc = tok(p0, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():                                  # 즉답 8토큰 greedy
            gen = model.generate(**enc, do_sample=False, max_new_tokens=args.ans_tokens,
                                 pad_token_id=tok.pad_token_id)
        ans_ids = gen[0, enc["input_ids"].shape[1]:]
        raw = tok.decode(ans_ids, skip_special_tokens=True)
        # **첫 줄만 답으로 쓴다.** 강제 종료를 뚫고 계속 떠드는 모델이 있다(답 뒤에 "Let's denote ..." 가 붙는다).
        # 그 수다가 평균 logprob 을 지배하면 '자기 답에 대한 확신' 이 아니라 '수다의 확신' 을 재게 된다.
        ans = raw.splitlines()[0].strip() if raw.strip() else ""
        rec["answer_raw"], rec["answer"] = raw.strip(), ans
        rec["ans_logprob"] = score_continuation(tok, model, p0, ans) if ans.strip() else None
        pert = perturb_constant(r["problem"])
        rec["perturbed"] = pert is not None
        if pert is not None and ans.strip():
            lp_pert = score_continuation(tok, model, chat(tok, pert, force=True), ans)
            rec["ans_logprob_pert"] = lp_pert
            rec["drop"] = None if (lp_pert is None or rec["ans_logprob"] is None) else rec["ans_logprob"] - lp_pert
        else:
            rec["ans_logprob_pert"], rec["drop"] = None, None
        out.append(rec)
        print(f"  [{n}/{len(rows)}] y={r['y']} 답 {rec['answer'][:18]!r} "
              f"logp {rec['ans_logprob']} drop {rec['drop']}", flush=True)
    return out


def stage_gen(tok, model, rows, args):
    """H21a 짧은 생성(1024토큰) 구간의 엔트로피·logprob.

    **주의: `output_scores=True` 를 쓰면 안 된다.** HF 가 스텝마다 (batch, vocab) 로짓을 전부 들고 있어
    1024스텝 × 배치8 × 152k 어휘 ≈ 5 GB 가 GPU 에 쌓인다(8 GB 카드에서 OOM). 대신 LogitsProcessor 로
    **스텝마다 스칼라로 줄여** 받는다. greedy 라 선택 토큰은 argmax 이므로 프로세서 안에서 바로 알 수 있다."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class Reduce(LogitsProcessor):
        def __init__(self, n):
            self.ent, self.lp, self.n = [], [], n

        def __call__(self, input_ids, scores):
            lp = torch.log_softmax(scores.float(), dim=-1)
            self.ent.append((-(lp.exp() * lp).sum(-1)).cpu())
            self.lp.append(lp.gather(1, lp.argmax(-1, keepdim=True)).squeeze(1).cpu())
            return scores

    out = []
    for st in range(0, len(rows), args.batch):
        chunk = rows[st:st + args.batch]
        prompts = [chat(tok, r["problem"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        red = Reduce(len(chunk))
        t0 = time.time()
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new,
                               pad_token_id=tok.pad_token_id, logits_processor=LogitsProcessorList([red]))
        new_tok = g[:, enc["input_ids"].shape[1]:]
        ent = torch.stack(red.ent, dim=1)                      # (batch, steps)
        lps = torch.stack(red.lp, dim=1)
        for i, r in enumerate(chunk):
            live = (new_tok[i] != tok.pad_token_id).cpu()[:ent.shape[1]]
            if live.sum() == 0:
                live[:] = True
            e, l = ent[i][live], lps[i][live]
            k = max(1, int(0.1 * len(l)))
            out.append({"pidx": r["pidx"], "y": r["y"], "per_model": r["per_model"],
                        "problem": r["problem"], "n_tok": int(live.sum()),
                        "mean_entropy": float(e.mean()), "mean_logprob": float(l.mean()),
                        "mink_logprob": float(l.sort().values[:k].mean()),
                        "hi_ent_frac": float((e > 1.0).float().mean())})
        print(f"  [{st + len(chunk)}/{len(rows)}] {time.time() - t0:.0f}s · 스텝 {ent.shape[1]} · "
              f"엔트로피 {out[-1]['mean_entropy']:.3f} · 토큰 {out[-1]['n_tok']}", flush=True)
        del red, ent, lps
        torch.cuda.empty_cache()
    return out


def stage_sens(tok, model, rows, args):
    """H21d·H21p — **무관 편집**(답 불변)에서 즉답 logp 가 얼마나 흔들리는가.

    [author]의 민감도 프로파일 가설: robust 하면 관련 편집(상수 변경)엔 민감하고 **무관 편집엔 둔감**해야 한다.
    H21c 가 앞쪽 절반, 여기가 뒤쪽 절반이다. 9/22 버전은 내부 표현 코사인으로 쟀다가 도구 쪽에서 기각됐고,
    이번에는 출력 확률로 잰다.

    conf 단계에서 저장한 **그 모델의 즉답 문자열을 그대로 쓴다** — 답을 다시 뽑으면 두 단계가 다른 답을
    비교하게 되어 H21p 가 무의미해진다."""
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import numpy as np
    from variants_official import edit_rename, edit_typos        # 결정적 규칙, 시드 고정
    alias = args.model.split("/")[-1].replace(".", "_")
    prev = OUT / f"{alias}_conf.json"
    if not prev.is_file():
        raise SystemExit(f"conf 결과가 없다: {prev} (먼저 --stage conf 를 돌릴 것)")
    conf = {r["pidx"]: r for r in json.loads(prev.read_text(encoding="utf-8"))}
    out = []
    for n, r in enumerate(rows, 1):
        c = conf.get(r["pidx"])
        if not c or not c.get("answer") or c.get("ans_logprob") is None:
            continue
        ans, lp0 = c["answer"], float(c["ans_logprob"])
        rec = {"pidx": r["pidx"], "y": r["y"], "per_model": r["per_model"], "problem": r["problem"],
               "answer": ans, "ans_logprob": lp0, "drop": c.get("drop")}
        deltas = {}
        for name, txt in (("rename", edit_rename(r["problem"])),
                          ("typos", edit_typos(r["problem"], np.random.default_rng(r["pidx"]), k=2))):
            if txt is None or txt == r["problem"]:
                deltas[name] = None
                continue
            lp = score_continuation(tok, model, chat(tok, txt, force=True), ans)
            deltas[name] = None if lp is None else lp0 - lp
        rec["d_rename"], rec["d_typos"] = deltas["rename"], deltas["typos"]
        got = [v for v in deltas.values() if v is not None]
        rec["d_irrel"] = float(np.mean([abs(v) for v in got])) if got else None       # 주지표용(둔감도)
        rec["d_irrel_signed"] = float(np.mean(got)) if got else None                  # 탐색용
        rec["h21p"] = (None if (rec["d_irrel"] is None or rec["drop"] is None)
                       else float(rec["drop"]) - rec["d_irrel"])
        out.append(rec)
        print(f"  [{n}/{len(rows)}] y={r['y']} 답 {ans[:12]!r} drop {rec['drop']} "
              f"d_irrel {rec['d_irrel']} h21p {rec['h21p']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["conf", "gen", "sens"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--ans-tokens", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = problems()
    if a.limit:
        rows = rows[:a.limit]
    print(f"[h21] {a.stage} · {a.model} · 문제 {len(rows)} "
          f"(robust {sum(r['y'] for r in rows)}/spurious {sum(1 - r['y'] for r in rows)})", flush=True)
    tok, model = load(a.model, a.quant)
    t0 = time.time()
    res = {"conf": stage_conf, "gen": stage_gen, "sens": stage_sens}[a.stage](tok, model, rows, a)
    alias = a.model.split("/")[-1].replace(".", "_")
    f = OUT / f"{alias}_{a.stage}.json"
    f.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"[h21] 저장 {f} · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
