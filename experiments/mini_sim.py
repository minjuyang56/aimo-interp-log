"""미니 시뮬레이션 — 판정 대상 모델을 직접 돌려 변형 시험을 하면 시간 예산 안에 정확도가 나오나.

train-main-v2의 한 모델 행을 골라, 문제마다
    원본 1회 + 프로그램 변형 2종(rename · typos) 각 1회 = 생성 3회
를 돌리고 케이스당 시간·답 일치·라벨 일치·max_drop 상관을 기록한다.
변형은 전부 규칙 기반(모델 없이 생성) — 런타임에도 합법인 종류만 쓴다.

보고 항목:
  a. 케이스당 평균 소요 시간 (예산 180초 대비)
  b. "변형에서 답 이탈" 신호 vs is_robust 라벨 일치율
  c. 이탈 신호(원본과 다른 변형 수) vs max_drop 상관

저장: results/mini_sim_<모델>.jsonl  (한 케이스 한 줄, 끊기면 --resume)

사용:
  uv run python experiments/mini_sim.py --limit 10 --max-new-tokens 2048          # GPU
  uv run python experiments/mini_sim.py --limit 2  --max-new-tokens 128           # CPU 시간 스케일만
  uv run python experiments/mini_sim.py --report-only
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensitivity import edit_typos, MATH_SPAN  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
TRAIN = HERE / "results" / "train-main-v2.parquet"
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.M)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
# 대문자(점 이름)는 알파벳 전체를 한 번에 돌리는 치환(A→P, B→Q, …, 전단사)이라 $ABC$, $\overline{AB}$처럼
# 붙어 있어도 글자 단위로 일관되게 바뀐다. 소문자 변수는 붙어 있으면 sin·log 같은 함수일 수 있어
# 홀로 선 글자만 바꾸고, 서로 맞바꾸는 쌍(x↔t …)이라 동시에 적용하면 충돌이 없다.
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
LATEX_CMD = re.compile(r"\\[A-Za-z]+")


def edit_rename(text, rng=None):
    """수식 안의 변수·점 이름을 일관되게 바꾼다. LaTeX 명령과 수식 밖 본문은 그대로."""
    def rename_chunk(chunk):
        chunk = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + UPPER_SHIFT) % 26 + 65), chunk)
        return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: LOWER_PAIRS.get(m.group(1), m.group(1)), chunk)
    def sub_span(m):
        span, out, pos = m.group(0), [], 0
        for cmd in LATEX_CMD.finditer(span):          # 명령어는 건너뛰고 그 사이만 바꾼다
            out.append(rename_chunk(span[pos:cmd.start()])); out.append(cmd.group(0)); pos = cmd.end()
        out.append(rename_chunk(span[pos:]))
        return "".join(out)
    return MATH_SPAN.sub(sub_span, text)


def clean_bytes(text):
    """디코드에 남은 바이트 BPE 표식(Ġ=공백, Ċ=줄바꿈)을 되돌린다. 정상 디코드면 무변화."""
    return text.replace("Ġ", " ").replace("Ċ", "\n")


def extract_answer(text):
    """ANSWER: 뒤 첫 줄(또는 마지막 \\boxed{})만 답으로 본다. 그 뒤에 이어지는 설명·재개된 생각은 버린다."""
    text = THINK_RE.sub("", clean_bytes(text))
    m = re.findall(r"ANSWER:\s*([^\n<]+)", text)
    hit = [m[-1]] if m else BOXED_RE.findall(text)
    if not hit:
        return None
    return hit[-1].strip().rstrip(".").replace(",", "").replace("$", "").strip() or None


def load_rows(limit):
    df = pd.read_parquet(TRAIN)
    df = df[(df.model_id == MODEL) & df.is_robust.notna()].copy()
    df["y"] = df.is_robust.astype(bool)
    # robust/spurious 섞이게: 라벨별로 번갈아 뽑는다
    t, f = df[df.y].to_dict("records"), df[~df.y].to_dict("records")
    rows, i = [], 0
    while len(rows) < limit and (i < len(t) or i < len(f)):
        if i < len(f): rows.append(f[i])
        if i < len(t) and len(rows) < limit: rows.append(t[i])
        i += 1
    return rows


FORCE_SUFFIX = "\n</think>\n\nBased on the reasoning so far, the final answer is:\nANSWER:"


def generate(tok, model, problem, max_new_tokens, force_answer=False, force_tokens=24):
    """greedy 생성. force_answer=True면 예산 안에 생각이 안 끝났을 때 </think>를 붙이고
    답만 짧게 더 뽑는다 (budget forcing). 반환 dict에 생각이 자연히 끝났는지도 담는다."""
    import torch
    text = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{problem}"}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new = out[0, ids["input_ids"].shape[1]:]
    hit_eos = bool((new == tok.eos_token_id).any())
    body = tok.decode(new, skip_special_tokens=True)
    think_closed = "</think>" in body
    forced = False
    if force_answer and not hit_eos and extract_answer(body) is None:
        forced = True
        ids2 = tok(text + body + FORCE_SUFFIX, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out2 = model.generate(**ids2, max_new_tokens=force_tokens, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        tail = tok.decode(out2[0, ids2["input_ids"].shape[1]:], skip_special_tokens=True)
        body = body + FORCE_SUFFIX + tail
    sec = time.time() - t0
    return {"text": body, "tokens": int(new.numel()), "seconds": sec, "eos": hit_eos,
            "think_closed": think_closed, "forced": forced}


def report(path, budget=180.0):
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        print("기록 없음"); return
    print(f"케이스 {len(recs)} · max_new_tokens {recs[0]['max_new_tokens']} · device {recs[0]['device']}")
    secs = np.array([r["case_seconds"] for r in recs]); toks = np.array([r["case_tokens"] for r in recs])
    rate = toks.sum() / secs.sum()
    print(f"\n[a] 케이스당 시간  평균 {secs.mean():.0f}s (min {secs.min():.0f}, max {secs.max():.0f})"
          f"   예산 180s 대비 {secs.mean() / budget:.1f}×   생성 속도 {rate:.2f} tok/s")
    mx = recs[0]["max_new_tokens"]
    if mx < 2048:
        print(f"    → 2048 토큰 × 3회로 환산: {3 * 2048 / rate:.0f}s / 케이스 = 예산의 {3 * 2048 / rate / budget:.0f}×")
    answered = [r for r in recs if r["answers"]["original"] is not None]
    n_gen = 3 * len(recs)
    closed = sum(g.get("think_closed", False) for r in recs for g in r["gen"].values())
    forced = sum(g.get("forced", False) for r in recs for g in r["gen"].values())
    print(f"\n[b] 답이 파싱된 케이스 {len(answered)}/{len(recs)}  (eos 도달 {sum(r['hit_eos_all'] for r in recs)}/{len(recs)} 케이스"
          f" · 생각 자연 종료 {closed}/{n_gen} 생성 · 강제 종료 {forced}/{n_gen} 생성)")
    if answered:
        agree = 0
        for r in answered:
            r["deviated"] = sum(1 for k in ("rename", "typos") if r["answers"][k] != r["answers"]["original"])
            pred_robust = r["deviated"] == 0
            agree += pred_robust == r["is_robust"]
        print(f"    이탈 0 → robust 예측 · 라벨 일치 {agree}/{len(answered)} = {agree / len(answered):.2f}")
        dev = np.array([r["deviated"] for r in answered]); md = np.array([r["max_drop"] for r in answered])
        if len(answered) >= 3 and dev.std() > 0:
            from scipy.stats import spearmanr
            print(f"\n[c] 이탈 수 vs max_drop  Spearman ρ = {spearmanr(dev, md).correlation:+.2f}  (n={len(answered)})")
        else:
            print("\n[c] 이탈 수에 분산이 없거나 n<3 — 상관 계산 불가")
    else:
        print("    → 답이 하나도 안 나옴. [b][c] 측정 불가 (생각 토큰이 max_new_tokens를 다 먹음)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                   help="VRAM 8 GB 노트북 GPU용. 4bit(nf4)면 8B가 약 5 GB. 서버(96 GB)에선 none")
    p.add_argument("--tag", default="", help="결과 파일 이름 꼬리표 (예: cpu128, gpu4bit). 실행 조건별로 파일을 가른다")
    p.add_argument("--force-answer", action="store_true",
                   help="예산 안에 생각이 안 끝나면 </think>를 붙여 답만 짧게 더 뽑는다 (budget forcing)")
    a = p.parse_args()
    out = HERE / "results" / f"mini_sim_{MODEL.split('/')[-1]}{'_' + a.tag if a.tag else ''}.jsonl"
    if a.report_only:
        report(out); return

    rows = load_rows(a.limit)
    done = set()
    if a.resume and out.is_file():
        done = {json.loads(l)["problem"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    print(f"모델 {MODEL} · 케이스 {len(rows)} (완료 {len(done)}) · max_new_tokens {a.max_new_tokens}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    # 업스트림 extraction.py의 우회: AutoTokenizer가 이 체크포인트를 legacy Llama로 변환해
    # 바이트 BPE를 못 되돌린다(디코드에 Ġ/Ċ가 남음). PreTrainedTokenizerFast로 직접 연다.
    from transformers import PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(MODEL, local_files_only=True)
    except (OSError, TypeError, ValueError):
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    kw = dict(dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
              device_map="auto" if device == "cuda" else None)
    if a.quant != "none" and device == "cuda":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16)
                                     if a.quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        kw.pop("dtype")
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    if device == "cpu": model.to("cpu")
    vram = f" vram={torch.cuda.memory_allocated() / 1e9:.1f}GB" if device == "cuda" else ""
    print(f"load {time.time() - t0:.0f}s device={device} quant={a.quant}{vram}", flush=True)

    rng = np.random.default_rng(0)
    with out.open("a", encoding="utf-8") as sink:
        for i, r in enumerate(rows, 1):
            if r["problem"] in done: continue
            variants = {"original": r["problem"], "rename": edit_rename(r["problem"]), "typos": edit_typos(r["problem"], rng)}
            answers, gen, tc, ts, eos_all = {}, {}, 0, 0.0, True
            for k, v in variants.items():
                g = generate(tok, model, v, a.max_new_tokens, force_answer=a.force_answer)
                answers[k] = extract_answer(g["text"])
                gen[k] = {"tokens": g["tokens"], "seconds": round(g["seconds"], 1), "eos": g["eos"],
                          "think_closed": g["think_closed"], "forced": g["forced"], "tail": g["text"][-160:]}
                tc += g["tokens"]; ts += g["seconds"]; eos_all &= g["eos"]
                print(f"  [{i}/{len(rows)}] {k:8s} {g['tokens']:5d} tok  {g['seconds']:6.0f}s  {g['tokens'] / g['seconds']:.2f} tok/s"
                      f"  ans={answers[k]!r:>8}  eos={g['eos']}  think_closed={g['think_closed']}  forced={g['forced']}", flush=True)
            rec = {"problem": r["problem"], "is_robust": bool(r["y"]), "max_drop": float(r["max_drop"]),
                   "reasoning_effort": r["reasoning_effort"], "variants": {k: v for k, v in variants.items() if k != "original"},
                   "answers": answers, "gen": gen, "case_tokens": tc, "case_seconds": round(ts, 1),
                   "hit_eos_all": eos_all, "max_new_tokens": a.max_new_tokens, "device": device}
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n"); sink.flush()
            print(f"  [{i}/{len(rows)}] 케이스 {ts:.0f}s  robust={r['y']}  max_drop={r['max_drop']:.2f}  answers={answers}", flush=True)
    report(out)


if __name__ == "__main__":
    main()
