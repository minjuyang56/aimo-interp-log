"""라벨 있는 137문제를 모델에 넣고 모든 층의 마지막-토큰 hidden state를 저장한다.

프로브 학습의 재료. 문제마다 forward 1회 — 생성은 안 한다.
저장: results/hidden_<모델별칭>.npz
    H        (문제수, 층수, 차원)  float16   마지막 프롬프트 토큰의 hidden state
    y        (문제수,)             bool      model_is_robust
    pid      (문제수,)             str       problem_id
    decay    (문제수,)             float32   absolute_accuracy_decay

프롬프트는 baselines/prompts/solve.txt + 채팅 템플릿. 모델이 "이제 풀기 시작할" 상태의
표현을 잡는다.

사용:
  uv run python experiments/extract_hidden.py                  # 전체 137문제
  uv run python experiments/extract_hidden.py --limit 2        # 타이밍 확인
  uv run python experiments/extract_hidden.py --model Qwen/Qwen3.5-4B
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
from _paths import AGG_CSV, BASELINES, require  # noqa: E402

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
SOLVE_PROMPT = BASELINES / "prompts" / "solve.txt"


def alias(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "_")


def load_tokenizer(model_id: str):
    """PreTrainedTokenizerFast 우선. AutoTokenizer는 DeepSeek-R1-0528-Qwen3-8B를 옛 LlamaTokenizer로
    열어서 인코딩이 달라지고(같은 문장 25 vs 30 토큰) 디코드에 Ġ/Ċ가 남는다 (2026-09-23 확인).
    업스트림 uncertainty-profiling/extraction.py와 같은 우회."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except (OSError, TypeError, ValueError):
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)


def load_rows(limit: int | None):
    with AGG_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # problem_id 기준 중복 제거 (141행 → 137문제). 라벨은 문제 단위라 첫 행이면 충분.
    seen, unique = set(), []
    for r in rows:
        if r["problem_id"] in seen:
            continue
        seen.add(r["problem_id"])
        unique.append(r)
    return unique[:limit] if limit else unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    require(AGG_CSV, SOLVE_PROMPT)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(args.limit)
    out = args.out or HERE / "results" / f"hidden_{alias(args.model)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt = SOLVE_PROMPT.read_text(encoding="utf-8").strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={args.model} device={device} 문제 {len(rows)}개 → {out}", flush=True)
    t0 = time.time()
    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    print(f"load {time.time() - t0:.1f}s", flush=True)

    H, y, pid, decay = [], [], [], []
    for i, r in enumerate(rows, 1):
        text = tok.apply_chat_template(
            [{"role": "user", "content": f"{prompt}\n\n{r['original_problem']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(text, return_tensors="pt").to(device)
        t1 = time.time()
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states   # (층수+1) × (1, T, D)
        H.append(np.stack([h[0, -1].float().cpu().numpy() for h in hs]).astype(np.float16))
        y.append(r["model_is_robust"].strip().lower() == "true")
        pid.append(r["problem_id"])
        decay.append(float(r["absolute_accuracy_decay"]))
        print(f"  [{i:3d}/{len(rows)}] {r['problem_id']}  T={ids['input_ids'].shape[1]:4d}"
              f"  {time.time() - t1:5.1f}s  robust={y[-1]}", flush=True)
        if i % 10 == 0:   # 중간 저장 — 죽어도 여기까지는 남는다
            np.savez_compressed(out, H=np.stack(H), y=np.array(y), pid=np.array(pid),
                                decay=np.array(decay, dtype=np.float32), model=args.model)

    np.savez_compressed(out, H=np.stack(H), y=np.array(y), pid=np.array(pid),
                        decay=np.array(decay, dtype=np.float32), model=args.model)
    Hs = np.stack(H)
    print(f"\n저장 {out}  H{Hs.shape}  robust {int(sum(y))}/{len(y)}  총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
