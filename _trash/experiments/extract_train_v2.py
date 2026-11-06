"""train-main-v2의 36문제를 로컬 4모델 각각에 읽혀 모든 층의 마지막-토큰 hidden state를 저장한다.

핵심 전제(9/22 확인): 라벨은 (모델, 문제)가 아니라 문제 단위다 — 같은 문제는 모델이 달라도
라벨이 같았다(11/11). 그래서 문제 라벨을 4모델에 공유하면 모델당 3~12개였던 학습 벡터가
모델당 36개가 된다. 라벨 없는 문제(None)도 벡터는 뽑아둔다 — 경계 실험용.

저장: results/trainv2_hidden_<모델별칭>.npz
    H       (문제수, 층수+1, 차원) float16
    y       (문제수,)  int8   1=robust 0=spurious -1=None
    drop    (문제수,)  float32 max_drop (여러 행이면 평균)
    pidx    (문제수,)  int    train-main-v2 안의 문제 순번
    problems (문제수,) str

사용:  .venv/Scripts/python.exe experiments/extract_train_v2.py [--models ...] [--quant 4bit]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias, load_tokenizer  # noqa: E402

TRAIN = HERE / "results" / "train-main-v2.parquet"
LOCAL_MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
                "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)


def problem_table():
    """문제 단위로 접는다. 라벨이 모델 따라 갈리는 문제는 다수결(동률이면 None)."""
    df = pd.read_parquet(TRAIN)
    rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna()
        if len(lab) == 0:
            y = -1
        else:
            t, f = int(lab.astype(bool).sum()), int((~lab.astype(bool)).sum())
            y = 1 if t > f else 0 if f > t else -1
        rows.append({"pidx": i, "problem": p, "y": y, "drop": float(g.max_drop.mean()),
                     "n_models": g.model_id.nunique()})
    return pd.DataFrame(rows)


def load_model(model_id, quant):
    import torch
    from transformers import AutoModelForCausalLM
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto")
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=LOCAL_MODELS)
    ap.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    a = ap.parse_args()

    import torch
    tbl = problem_table()
    print(f"문제 {len(tbl)} · robust {(tbl.y == 1).sum()} · spurious {(tbl.y == 0).sum()} · None {(tbl.y == -1).sum()}", flush=True)
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)

    for m in a.models:
        out = out_dir / f"trainv2_hidden_{alias(m)}.npz"
        t0 = time.time()
        tok = load_tokenizer(m)
        model = load_model(m, a.quant)
        print(f"\n== {m}  load {time.time() - t0:.0f}s  device={model.device}", flush=True)
        H = []
        for r in tbl.itertuples():
            text = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{r.problem}"}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            t1 = time.time()
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states
            H.append(np.stack([h[0, -1].float().cpu().numpy() for h in hs]).astype(np.float16))
            print(f"  [{r.pidx + 1:2d}/{len(tbl)}] T={ids['input_ids'].shape[1]:4d} {time.time() - t1:5.1f}s y={r.y:+d}", flush=True)
        np.savez_compressed(out, H=np.stack(H), y=tbl.y.values.astype(np.int8), drop=tbl["drop"].values.astype(np.float32),
                            pidx=tbl.pidx.values, problems=tbl.problem.values, model=m)
        print(f"  저장 {out}  H{np.stack(H).shape}  총 {time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
