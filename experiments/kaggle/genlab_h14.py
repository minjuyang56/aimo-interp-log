"""H14 — rename 변형 시 생성길이비가 클수록 spurious. **공식 라벨로 직접 검증.** (지휘 #43, 사전등록 2026-09-30)

왜: 합성 라벨을 더 늘리는 대신, 이미 있는 공식 학습셋(train-main-v2, 문제 단위 라벨 33개)에 바로 물어본다.
    근거는 p13b 판별 실험 — 상한을 16384로 올려 양쪽 모두 완주시킨 뒤에도 **진짜 spurious 4건의 길이비 평균 3.25 vs
    절단 아티팩트 9건 1.17**. 진짜로 무너진 4건은 전부 rename 이었다. (n=4, 가설 수준)

절차 (Skywork)
  문제 33개 × (원본 1 + rename 2종) × k 회 샘플(T=0.6, top_p=0.95), max_new 16384
  문제별 길이비 = mean(rename 두 종의 평균 생성 토큰) / mean(원본 평균 생성 토큰)
출력 /kaggle/working/h14/<별칭>_h14.json — 문제별 라벨·원본/변형 토큰 수·완주 수·정답 수·길이비

분석(로컬 `analysis/h14_length_ratio.py`)은 사전등록대로: **부호 +(길이비 큼 = spurious)**,
주지표 AUC, 문제 단위 permutation null 1000, 원본 생성길이 잔차화 AUC 병기.
"""

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

V2_URL = "https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/data/train-00000-of-00001.parquet"


def official_problems():
    """train-main-v2 문제 단위 라벨 (모델 간 다수결). 반환: [{pidx, problem, y}] — y 없는 문제는 제외."""
    import pandas as pd
    cache = Path("train-main-v2.parquet")
    if not cache.is_file():
        cache.write_bytes(urllib.request.urlopen(urllib.request.Request(V2_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read())
    df = pd.read_parquet(cache)
    rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna()
        if len(lab) == 0:
            continue
        rows.append({"pidx": i, "problem": p.rstrip(), "y": int(bool(lab.astype(bool).mean() >= 0.5))})
    print(f"공식 문제 {len(rows)}개 (robust {sum(r['y'] for r in rows)} / spurious {sum(1 - r['y'] for r in rows)})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=globals().get("RUN_MODEL", "Skywork/Skywork-OR1-Math-7B"))
    ap.add_argument("--k", type=int, default=globals().get("RUN_K", 2))
    ap.add_argument("--max-new", type=int, default=globals().get("RUN_MAX_NEW", 16384))
    ap.add_argument("--max-seqs", type=int, default=globals().get("RUN_MAX_SEQS", 16))
    ap.add_argument("--out", default="/kaggle/working/h14")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    alias = a.model.split("/")[-1].replace(".", "_")
    probs = official_problems()

    import torch
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, dtype="half", tensor_parallel_size=max(1, torch.cuda.device_count()),
              max_num_seqs=a.max_seqs, max_model_len=a.max_new + 2048, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=a.k, temperature=0.6, top_p=0.95, max_tokens=a.max_new, seed=0)
    print(f"[H14] vLLM 로드 {time.time() - t0:.0f}s · 문제 {len(probs)} × (원본+rename 2종) × k={a.k}", flush=True)

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}],   # noqa: F821
                                       tokenize=False, add_generation_prompt=True)

    import re as _re
    kinds = [k.strip() for k in _re.split(r"[,|]", globals().get("RUN_KINDS", "orig,rename,rename2")) if k.strip()]
    plan = []
    for p in probs:
        import numpy as _np
        _rng = _np.random.default_rng(p["pidx"])
        # **요청한 종류만 만든다.** 전부 미리 만들면 kinds=orig 로 줄여 돌릴 때도 edit_* 가 필요해져
        # 변형 코드를 inline 하지 않으면 NameError 로 죽는다(H20-M 1차 push 에서 vLLM 로드까지 끝낸 뒤 터졌다).
        makers = {"orig": lambda t, r: t,
                  "rename": lambda t, r: edit_rename(t),                 # noqa: F821
                  "rename2": lambda t, r: edit_rename2(t),               # noqa: F821
                  "typos": lambda t, r: edit_typos(t, r),                # noqa: F821
                  "distract": lambda t, r: edit_distract(t, r)}          # noqa: F821
        for kd in kinds:
            txt = makers[kd](p["problem"], _rng)
            plan.append((p, kd, txt or p["problem"], txt is None))
    prompts = [chat(t) for _, _, t, _ in plan]
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    ntok = sum(len(o.token_ids) for out in outs for o in out.outputs)
    print(f"[H14] 생성 {len(prompts)}프롬프트 × k={a.k} · {time.time() - t1:.0f}s · {ntok / max(1, time.time() - t1):.1f} tok/s", flush=True)

    rec = {p["pidx"]: {"pidx": p["pidx"], "y": p["y"], "problem": p["problem"], "by_kind": {}} for p in probs}
    for (p, kd, _, degenerate), out in zip(plan, outs):
        toks = [len(o.token_ids) for o in out.outputs]
        fin = [o for o in out.outputs if o.finish_reason == "stop"]
        rec[p["pidx"]]["by_kind"][kd] = {"tokens": toks, "mean_tokens": sum(toks) / max(1, len(toks)),
                                         "n_fin": len(fin), "degenerate": bool(degenerate),
                                         "answers": [extract_answer(o.text) for o in out.outputs]}      # noqa: F821
    for r in rec.values():
        bk = r["by_kind"]
        base = bk["orig"]["mean_tokens"] if "orig" in bk else None
        # RUN_KINDS 를 줄여 돌릴 수 있다(H20 재측정은 orig 하나). 없는 종류를 참조해 죽지 않게 한다.
        var = [bk[kd]["mean_tokens"] for kd in ("rename", "rename2") if kd in bk and not bk[kd]["degenerate"]]
        r["length_ratio"] = (sum(var) / len(var)) / max(1.0, base) if (var and base) else None
        r["n_variant_used"] = len(var)
    ok = [r for r in rec.values() if r["length_ratio"] is not None]
    print(f"[H14] 길이비 산출 {len(ok)}/{len(rec)} (rename 이 원문과 같아 못 쓴 문제 {len(rec) - len(ok)})", flush=True)
    print(f"  robust 평균 길이비 {sum(r['length_ratio'] for r in ok if r['y']) / max(1, sum(1 for r in ok if r['y'])):.2f} · "
          f"spurious 평균 {sum(r['length_ratio'] for r in ok if not r['y']) / max(1, sum(1 for r in ok if not r['y'])):.2f}", flush=True)
    Path(a.out, f"{alias}_h14.json").write_text(json.dumps(list(rec.values())), encoding="utf-8")
    print("[H14] 저장 완료", flush=True)


if __name__ == "__main__":
    main()
