"""gen_labels.py — 클라우드 GPU(Kaggle T4 ×2)에서 자체 라벨을 만드는 단일 스크립트.

무엇을 하나
  1. train-main-v2 문제(HF에서 직접 받음)마다 규칙 기반 변형본을 만든다:
       rename (변수·점 이름 일관 치환) · typos (단어 3개 글자 순서) · distract (무관 문장 삽입)
       reorder (앞 두 문장 맞바꿈, 보수적 조건) — 네 가지 모두 정답 불변
       number (숫자 n+1, 답이 바뀜) 는 --variants 에 명시할 때만 대조군으로. max_drop 계산에서 항상 제외
  2. Qwen3.5-4B로 원본 + 변형본 각각 k회 샘플링 생성 (transformers 배치, vLLM 없음)
  3. 원본 k회의 최빈 답 = 의사 정답. 변형별 정답유지율 = 답이 의사 정답과 같은 비율
  4. my_max_drop = base_acc − min(variant_acc)  (number 제외)
  5. 주최측 max_drop(v2 컬럼)과 스피어만 상관 → 이 절차가 주최측 라벨과 얼마나 맞나

체크포인트  results/gen_labels/<run>/generations.jsonl 에 생성 1건 = 1줄. 재실행하면 있는 건 건너뜀.
재개       같은 인자로 다시 실행하면 이어서 한다.

Kaggle 사용 (인터넷 ON, GPU T4 ×2):
  !pip -q install "transformers>=4.51" accelerate sentencepiece scipy pandas pyarrow
  !python gen_labels.py --out /kaggle/working/gen --k 4 --max-new-tokens 6144 --batch 8
  → 끝나면 /kaggle/working/gen/summary.csv + report.txt 를 내려받는다.
  중간에 끊기면 같은 명령을 다시 실행 (generations.jsonl 기준 재개).

노트북 스모크:
  .venv/Scripts/python.exe experiments/gen_labels.py --limit 1 --variants rename,typos --k 1 --max-new-tokens 512 --quant 4bit --out experiments/results/gen_smoke

T4는 bf16이 안 되므로 dtype은 fp16(자동). 8 GB 노트북 GPU에선 --quant 4bit.
"""

import argparse
import io
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen3.5-4B"
HF_PARQUET = "https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/data/train-00000-of-00001.parquet"
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
ANSWER_RE = re.compile(r"ANSWER:\s*([^\n<]+)")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
NUMBER = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
DISTRACTORS = [
    "Note that the problem was originally posed at a regional competition in 2019.",
    "A friend of the author suggested this problem after a long hike.",
    "The committee that wrote this problem also enjoys chess on weekends.",
]


# ---------------------------------------------------------------- 변형 (규칙 기반, 모델 없음)
def edit_rename(text, rng):
    def chunk(c):
        c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + UPPER_SHIFT) % 26 + 65), c)
        return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: LOWER_PAIRS.get(m.group(1), m.group(1)), c)
    def span(m):
        s, out, pos = m.group(0), [], 0
        for cmd in LATEX_CMD.finditer(s):
            out.append(chunk(s[pos:cmd.start()])); out.append(cmd.group(0)); pos = cmd.end()
        out.append(chunk(s[pos:])); return "".join(out)
    return MATH_SPAN.sub(span, text)


def edit_typos(text, rng, k=3):
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    words = [m for m in re.finditer(r"[A-Za-z]{5,}", text) if not any(a <= m.start() < b for a, b in spans)]
    if not words:
        return text
    out = list(text)
    for i in sorted(rng.choice(len(words), size=min(k, len(words)), replace=False), reverse=True):
        m = words[i]; j = m.start() + 1 + int(rng.integers(m.end() - m.start() - 3))
        out[j], out[j + 1] = out[j + 1], out[j]
    return "".join(out)


def edit_distract(text, rng):
    return text.rstrip() + " " + DISTRACTORS[int(rng.integers(len(DISTRACTORS)))]


def edit_number(text, rng):
    nums = [m for m in NUMBER.finditer(text) if int(m.group(1)) >= 2]
    if not nums:
        return None
    m = nums[int(rng.integers(len(nums)))]
    return text[:m.start(1)] + str(int(m.group(1)) + 1) + text[m.end(1):]


def edit_reorder(text, rng):
    """문장 재배열 — 보수적으로: 문장 3개 이상이고 앞 두 문장이 질문이 아닐 때, 1·2번째 문장만 맞바꾼다.
    (정의 순서가 뒤집히면 뜻이 바뀔 수 있어 그 이상은 안 건드린다. 조건이 안 맞으면 None.)"""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) < 3 or parts[0].endswith("?") or parts[1].endswith("?"):
        return None
    if re.search(r"(Let|Define|Suppose|Denote)", parts[0] + " " + parts[1]):
        return None
    return " ".join([parts[1], parts[0]] + parts[2:])


EDITS = {"rename": edit_rename, "typos": edit_typos, "distract": edit_distract, "reorder": edit_reorder, "number": edit_number}
ANSWER_CHANGING = {"number"}


def extract_answer(text):
    text = THINK_RE.sub("", text)
    m = ANSWER_RE.findall(text); hit = [m[-1]] if m else BOXED_RE.findall(text)
    if not hit:
        return None
    a = hit[-1].strip().rstrip(".").replace(",", "").replace("$", "").strip()
    return a or None


# ---------------------------------------------------------------- 데이터
def load_problems(limit=None):
    import pandas as pd
    cache = Path("train-main-v2.parquet")
    if not cache.is_file():
        cache.write_bytes(urllib.request.urlopen(urllib.request.Request(HF_PARQUET, headers={"User-Agent": "Mozilla/5.0"}), timeout=180).read())
    df = pd.read_parquet(cache)
    rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna()
        y = None if len(lab) == 0 else bool(lab.astype(bool).mean() >= 0.5)
        rows.append({"pidx": i, "problem": p, "is_robust": y, "max_drop": float(g.max_drop.mean())})
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- 모델
def load_model(quant):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    kw = dict(device_map="auto" if cuda else None, low_cpu_mem_usage=True)
    if quant == "4bit" and cuda:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    else:
        kw["dtype"] = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    if not cuda:
        model.to("cpu")
    return tok, model


def generate_batch(tok, model, texts, max_new_tokens, temperature, seed):
    import torch
    torch.manual_seed(seed)
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{t}"}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                             temperature=temperature if temperature > 0 else None, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    res = []
    for row in new:
        n = int((row != tok.pad_token_id).sum())
        res.append((tok.decode(row, skip_special_tokens=True), n, bool((row == tok.eos_token_id).any())))
    return res


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gen_labels/run1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--balanced", type=int, help="라벨 있는 문제에서 robust N + spurious N만 (첫 Kaggle 실행용)")
    ap.add_argument("--variants", default="rename,typos,distract,reorder",
                    help="정답 불변 변형만 기본. number(답 바뀜)는 명시할 때만 대조군으로")
    ap.add_argument("--k", type=int, default=4, help="텍스트당 샘플 수")
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gen_path = out / "generations.jsonl"
    variants = [v for v in a.variants.split(",") if v]
    rng = np.random.default_rng(0)
    problems = load_problems(a.limit)
    if a.balanced:
        rob = [r for r in problems if r["is_robust"] is True][:a.balanced]
        spu = [r for r in problems if r["is_robust"] is False][:a.balanced]
        problems = sorted(rob + spu, key=lambda r: r["pidx"])

    # 작업 목록: (pidx, kind, sample_idx, text)
    jobs, texts = [], {}
    for r in problems:
        texts[(r["pidx"], "original")] = r["problem"]
        for v in variants:
            t = EDITS[v](r["problem"], rng)
            if t is not None and t != r["problem"]:
                texts[(r["pidx"], v)] = t
    for (pidx, kind), t in texts.items():
        for s in range(a.k):
            jobs.append((pidx, kind, s, t))
    done = set()
    if gen_path.is_file():
        for line in gen_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line); done.add((d["pidx"], d["kind"], d["sample"]))
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"문제 {len(problems)} · 텍스트 {len(texts)} · 생성 총 {len(jobs)} · 완료 {len(done)} · 남음 {len(todo)}", flush=True)

    if not a.report_only and todo:
        tok, model = load_model(a.quant)
        t0 = time.time(); n_done = 0
        with gen_path.open("a", encoding="utf-8") as sink:
            for i in range(0, len(todo), a.batch):
                chunk = todo[i:i + a.batch]
                res = generate_batch(tok, model, [j[3] for j in chunk], a.max_new_tokens, a.temperature, seed=1000 + i)
                for (pidx, kind, s, _), (text, ntok, eos) in zip(chunk, res):
                    sink.write(json.dumps({"pidx": pidx, "kind": kind, "sample": s, "answer": extract_answer(text),
                                           "tokens": ntok, "eos": eos, "tail": text[-120:]}, ensure_ascii=False) + "\n")
                sink.flush(); n_done += len(chunk)
                el = time.time() - t0
                print(f"  {n_done}/{len(todo)}  {el:.0f}s  ({el / n_done:.0f}s/gen, 남은 예상 {el / n_done * (len(todo) - n_done) / 60:.0f}분)  "
                      f"last: {[(j[1], r[1], r[2], r[0] and extract_answer(r[0])) for j, r in zip(chunk[-2:], res[-2:])]}", flush=True)

    # ---------------------------------------------------------- 집계
    recs = [json.loads(l) for l in gen_path.read_text(encoding="utf-8").splitlines() if l.strip()] if gen_path.is_file() else []
    by = {}
    for d in recs:
        by.setdefault((d["pidx"], d["kind"]), []).append(d)
    rows = []
    for r in problems:
        orig = by.get((r["pidx"], "original"), [])
        answers = [d["answer"] for d in orig if d["answer"] is not None]
        gold = Counter(answers).most_common(1)[0][0] if answers else None
        base_acc = np.mean([d["answer"] == gold for d in orig]) if orig and gold else np.nan
        row = {"pidx": r["pidx"], "is_robust": r["is_robust"], "org_max_drop": r["max_drop"], "gold": gold,
               "base_acc": base_acc, "n_orig": len(orig), "eos_rate": np.mean([d["eos"] for d in orig]) if orig else np.nan}
        accs = []
        for v in variants:
            g = by.get((r["pidx"], v), [])
            acc = np.mean([d["answer"] == gold for d in g]) if g and gold else np.nan
            row[f"acc_{v}"] = acc
            if v not in ANSWER_CHANGING and not np.isnan(acc):
                accs.append(acc)
        row["my_max_drop"] = (base_acc - min(accs)) if accs and not np.isnan(base_acc) else np.nan
        row["number_insensitivity"] = row.get("acc_number", np.nan)   # 숫자 바꿨는데 같은 답 = 외움 의심
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False)
    lines = [f"문제 {len(df)} · 의사정답 있음 {df.gold.notna().sum()} · eos 도달율 평균 {df.eos_rate.mean():.2f}"]
    ok = df.dropna(subset=["my_max_drop"])
    if len(ok) >= 3:
        from scipy.stats import spearmanr
        rho = spearmanr(ok.my_max_drop, ok.org_max_drop).correlation
        lines.append(f"my_max_drop vs 주최측 max_drop  Spearman ρ = {rho:+.2f}  (n={len(ok)})")
        lab = ok.dropna(subset=["is_robust"])
        if len(lab) >= 3:
            from sklearn.metrics import roc_auc_score
            lines.append(f"my_max_drop → is_robust  AUC = {roc_auc_score(lab.is_robust.astype(int), -lab.my_max_drop):.3f}  (n={len(lab)})")
    lines.append(f"변형별 평균 정답유지율: " + "  ".join(f"{v}={df[f'acc_{v}'].mean():.2f}" for v in variants if f"acc_{v}" in df))
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print(f"저장 {out / 'summary.csv'} · {out / 'report.txt'}")


if __name__ == "__main__":
    main()
