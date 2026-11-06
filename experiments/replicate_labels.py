"""대리 모델(8B) 하나로 라벨 생성 절차를 소규모 재현하고, 시간을 잰다.

이 실험 하나가 "변형 생성 → k회 풀기 → 이탈률" 전략의 생사를 결정한다.
두 숫자를 얻는 게 목적이다:

  1. 라벨 재현율 — 8B가 낸 robust/spurious 판정이 주최측 라벨과 얼마나 맞나
  2. 문제당 소요 시간 — 3600초 예산 안에 몇 문제를 볼 수 있나

절차 (논문 §1.4와 baselines CSV 컬럼을 그대로 따라감):

  원본을 k회 풀린다                       → base_accuracy
  유형별로 변형본을 만들어 각각 k회 풀린다   → variant_accuracy
  변형본 중 과반이 무너지면 False, 아니면 True

정직하게 밝혀둘 두 가지:

  - CSV에 정답 컬럼이 없다. 그래서 원본 k회 풀이의 최빈 답을 의사 정답으로 쓴다.
    base_accuracy가 675행 중 662행에서 1.0이라 이 근사는 거의 항상 맞는다.
    (원본조차 못 맞히는 문제는 애초에 라벨이 안 붙는다 — 논문 "not clear-cut" 제외)
  - 주최측은 변형본을 gpt-oss-120b로 만들었다. 여기선 같은 8B로 만든다.
    변형본 품질이 다르니 재현율은 상한이 아니라 하한으로 읽어야 한다.

CPU에서 돌리면 문제당 시간은 GPU의 수십 배다. 로컬 실행은 파이프라인이 도는지
확인하는 용도이고, 전략을 결정하는 숫자는 GPU에서 다시 재야 한다.

사용:
  # 파이프라인이 도는지만 확인 (1문제, 1회, 짧게)
  uv run python experiments/replicate_labels.py --n-problems 1 --k 1 --max-new-tokens 512

  # 본 실험 (GPU)
  uv run python experiments/replicate_labels.py --n-problems 10 --k 5

  # 주최측이 만든 변형본이 있는 행은 그걸 쓴다 (False 행에만 있음)
  uv run python experiments/replicate_labels.py --use-organizer-variants
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
from _paths import AGG_CSV, BASELINES, require  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
SOLVE_PROMPT_PATH = BASELINES / "prompts" / "solve.txt"

# 논문 Appendix B의 정의를 그대로 지시문으로 옮겼다.
VARIANT_PROMPTS = {
    "rephrase": "Rewrite the following math problem using different wording while keeping "
                "its meaning and its answer exactly the same.",
    "rename": "Rewrite the following math problem, renaming every variable, person, or object "
              "to different names while preserving the underlying structure and the answer.",
    "domain": "Rewrite the following math problem so that the same mathematical structure is "
              "set in a completely different real-world domain. The answer must stay the same.",
    "distract": "Rewrite the following math problem, inserting one or two sentences of "
                "irrelevant but plausible information that do not change the answer.",
    "typos": "Rewrite the following math problem, introducing a few harmless typographical "
             "errors (misspellings, doubled letters, odd spacing) without changing any number "
             "or the answer.",
}
VARIANT_SUFFIX = ("\n\nOutput only the rewritten problem, nothing else.\n\nProblem:\n{problem}")

ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.M)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-problems", type=int, default=10, help="True/False 균형 맞춰 뽑는다")
    p.add_argument("--k", type=int, default=5, help="원본·변형본 각각 몇 번 풀리나")
    p.add_argument("--types", default="rephrase,rename,domain,distract,typos")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-organizer-variants", action="store_true",
                   help="permutations_causing_decay의 변형본이 있으면 생성 대신 그걸 쓴다")
    p.add_argument("--out", type=Path, default=HERE / "results" / "replicate_labels.jsonl")
    p.add_argument("--dry-run", action="store_true", help="모델 없이 문제 선택과 프롬프트만 출력")
    return p.parse_args()


def as_bool(text):
    return text.strip().lower() == "true"


def pick_problems(n, seed):
    """agg CSV에서 True/False 반반. 같은 seed면 같은 문제."""
    with AGG_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rng = random.Random(seed)
    trues = [r for r in rows if as_bool(r["model_is_robust"])]
    falses = [r for r in rows if not as_bool(r["model_is_robust"])]
    rng.shuffle(trues)
    rng.shuffle(falses)
    half = n // 2
    chosen = trues[:n - half] + falses[:half]
    rng.shuffle(chosen)
    return chosen


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"[load] {MODEL_ID}  device={device} dtype={dtype}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
    ).to(device).eval()
    print(f"[load] {time.time() - t0:.0f}s", flush=True)
    return tok, model, device


def chat(tok, model, device, user_text, max_new_tokens, temperature):
    """한 번 생성. (본문, 생성 토큰 수, 초) 를 돌려준다. <think> 블록은 떼어낸다."""
    import torch

    messages = [{"role": "user", "content": user_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    seconds = time.time() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    text = THINK_RE.sub("", text).strip()
    return text, int(new_tokens.numel()), seconds


def extract_answer(text):
    hit = ANSWER_RE.findall(text)
    if not hit:
        return None
    return hit[-1].strip().rstrip(".").replace(",", "")


def solve_many(tok, model, device, solve_prompt, problem, k, args):
    answers, tokens, seconds = [], 0, 0.0
    for _ in range(k):
        text, n_tok, sec = chat(tok, model, device, f"{solve_prompt}\n\n{problem}",
                                args.max_new_tokens, args.temperature)
        answers.append(extract_answer(text))
        tokens += n_tok
        seconds += sec
    return answers, tokens, seconds


def main():
    args = parse_args()
    require(AGG_CSV, SOLVE_PROMPT_PATH)
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    solve_prompt = SOLVE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    problems = pick_problems(args.n_problems, args.seed)

    print(f"문제 {len(problems)}개 / k={args.k} / 유형 {types} / max_new_tokens={args.max_new_tokens}")
    for r in problems:
        print(f"  {r['problem_id']}  robust={r['model_is_robust']:5s}  {r['original_problem'][:70]!r}")
    if args.dry_run:
        print("\n[dry-run] 변형 생성 프롬프트 예시:")
        print(VARIANT_PROMPTS[types[0]] + VARIANT_SUFFIX.format(problem=problems[0]["original_problem"]))
        return

    tok, model, device = load_model()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    agree = 0

    with args.out.open("a", encoding="utf-8") as sink:
        for idx, row in enumerate(problems, 1):
            t_problem = time.time()
            original = row["original_problem"]
            label = as_bool(row["model_is_robust"])
            print(f"\n[{idx}/{len(problems)}] {row['problem_id']}  label={label}", flush=True)

            # 1. 원본 k회 → 의사 정답 + base_accuracy
            base_answers, base_tok, base_sec = solve_many(tok, model, device, solve_prompt, original, args.k, args)
            counted = Counter(a for a in base_answers if a is not None)
            gold = counted.most_common(1)[0][0] if counted else None
            base_acc = sum(a == gold for a in base_answers) / args.k if gold else 0.0
            print(f"    원본     answers={base_answers}  gold={gold}  base_acc={base_acc:.2f}"
                  f"  ({base_tok} tok, {base_sec:.0f}s)", flush=True)

            # 2. 변형본 만들기 — 주최측 것이 있으면 그걸, 없으면 생성
            variants = []
            organizer = json.loads(row["permutations_causing_decay"]) if args.use_organizer_variants else []
            for item in organizer:
                if item["permutation_type"] in types:
                    variants.append((item["permutation_type"], item["question"], "organizer"))
            covered = {v[0] for v in variants}
            gen_tok = gen_sec = 0
            for ptype in types:
                if ptype in covered:
                    continue
                text, n_tok, sec = chat(tok, model, device,
                                        VARIANT_PROMPTS[ptype] + VARIANT_SUFFIX.format(problem=original),
                                        max_new_tokens=min(args.max_new_tokens, 1024),
                                        temperature=args.temperature)
                gen_tok += n_tok
                gen_sec += sec
                variants.append((ptype, text, "generated"))
            print(f"    변형본   {len(variants)}개 (생성 {gen_tok} tok, {gen_sec:.0f}s)", flush=True)

            # 3. 변형본 각각 k회
            variant_results = []
            var_tok = var_sec = 0
            for ptype, text, source in variants:
                answers, n_tok, sec = solve_many(tok, model, device, solve_prompt, text, args.k, args)
                var_tok += n_tok
                var_sec += sec
                acc = sum(a == gold for a in answers) / args.k if gold else 0.0
                broken = acc < 0.5
                variant_results.append({"type": ptype, "source": source, "question": text,
                                        "answers": answers, "accuracy": acc, "broken": broken})
                print(f"      {ptype:9s} acc={acc:.2f} {'BROKEN' if broken else 'ok':6s} {answers}", flush=True)

            # 4. 과반 규칙
            n_broken = sum(v["broken"] for v in variant_results)
            predicted = not (n_broken > len(variant_results) / 2)
            decay = base_acc - (sum(v["accuracy"] for v in variant_results) / len(variant_results))
            agree += predicted == label
            elapsed = time.time() - t_problem
            total_tok = base_tok + gen_tok + var_tok
            print(f"    판정     broken {n_broken}/{len(variant_results)} → predicted={predicted}"
                  f"  label={label}  {'MATCH' if predicted == label else 'MISS'}"
                  f"  decay={decay:.2f}", flush=True)
            print(f"    시간     {elapsed:.0f}s  ({total_tok} tok, {total_tok / elapsed:.1f} tok/s)", flush=True)

            sink.write(json.dumps({
                "problem_id": row["problem_id"], "label": label, "predicted": predicted,
                "base_answers": base_answers, "gold": gold, "base_accuracy": base_acc,
                "variants": variant_results, "n_broken": n_broken, "decay": decay,
                "seconds": elapsed, "tokens": total_tok, "k": args.k, "device": device,
                "max_new_tokens": args.max_new_tokens,
            }, ensure_ascii=False) + "\n")
            sink.flush()

    print(f"\n라벨 재현율: {agree}/{len(problems)} = {agree / len(problems):.2f}")
    print(f"결과: {args.out}")


if __name__ == "__main__":
    main()
