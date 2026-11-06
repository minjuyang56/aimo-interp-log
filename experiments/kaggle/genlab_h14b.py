"""H14b — 저비용 대리지표: 앞 1024토큰의 불확실성 차이. (지휘 #44, 사전등록 2026-09-30)

왜: H14(길이비)는 원본·변형을 끝까지 생성해야 해서 제출에 못 쓴다 — 공식 FAQ 8번이 3600초를 **배치 전체 합산**으로
    못박았고, 문제당 3회 × 16384토큰은 240 인스턴스에서 한도를 넘는다. 그래서 **짧게 끊어 보는 대리지표**가 필요하다.

절차: 공식 train-main-v2 문제 단위 라벨 33개 × (원본 1 + rename 1) × **1회** 생성, **max_new 1024**.
      vLLM 에 logprobs=20 을 요청해 토큰별 상위 20 로그확률을 받는다.
지표 (모두 rename − 원본, **부호: rename 이 더 불확실할수록 spurious**)
  (a) d_logprob = 평균 토큰 logprob 차이            -> 변형에서 더 낮으면(음수) spurious 방향
  (b) d_entropy = 평균 엔트로피 차이(상위 20 절단)   -> 변형에서 더 높으면(양수) spurious 방향
  (c) d_first_math = 첫 수식 등장까지 토큰 수 차이    -> 변형에서 더 늦으면(양수) spurious 방향
  주지표는 (a). 부호는 위대로 고정하고, 분석에서 (b)(c)는 같은 방향으로 맞춰 보고한다.
출력 /kaggle/working/h14b/<별칭>_h14b.json — 문제별 라벨·지표·생성 텍스트 앞부분(수식 위치 재계산용)

주의: 엔트로피는 상위 20개만으로 계산한 **절단 엔트로피**다. 정확한 값이 아니라 대리값이다.
"""

import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path

V2_URL = "https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/data/train-00000-of-00001.parquet"
MATH_PAT = re.compile(r"\$|\\\(|\\\[|\\frac|\\sqrt|[0-9]\s*[+\-*/=]\s*[0-9]")


def official_problems():
    import pandas as pd
    cache = Path("train-main-v2.parquet")
    if not cache.is_file():
        cache.write_bytes(urllib.request.urlopen(urllib.request.Request(V2_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read())
    df = pd.read_parquet(cache)
    rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna()
        if len(lab):
            rows.append({"pidx": i, "problem": p.rstrip(), "y": int(bool(lab.astype(bool).mean() >= 0.5))})
    print(f"공식 문제 {len(rows)}개 (robust {sum(r['y'] for r in rows)} / spurious {sum(1 - r['y'] for r in rows)})", flush=True)
    return rows


def first_math_token(text, tok):
    """첫 수식 등장까지의 토큰 수. 없으면 전체 길이."""
    m = MATH_PAT.search(text)
    if not m:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return len(tok(text[:m.start()], add_special_tokens=False)["input_ids"])


def stats(out, tok):
    """한 생성물의 (평균 logprob, 평균 절단 엔트로피, 첫 수식 토큰, 토큰 수)."""
    o = out.outputs[0]
    lps, ents = [], []
    for step in (o.logprobs or []):
        if not step:
            continue
        vals = [v.logprob for v in step.values()]
        top = max(vals)
        lps.append(top)
        p = [math.exp(v) for v in vals]
        s = sum(p) or 1.0
        ents.append(-sum((q / s) * math.log(max(q / s, 1e-12)) for q in p))
    return (sum(lps) / max(1, len(lps)), sum(ents) / max(1, len(ents)),
            first_math_token(o.text, tok), len(o.token_ids), o.text[:2000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=globals().get("RUN_MODEL", "Skywork/Skywork-OR1-Math-7B"))
    ap.add_argument("--max-new", type=int, default=globals().get("RUN_MAX_NEW", 1024))
    ap.add_argument("--max-seqs", type=int, default=globals().get("RUN_MAX_SEQS", 32))
    ap.add_argument("--out", default="/kaggle/working/h14b")
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
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_new, logprobs=20, seed=0)
    print(f"[H14b] vLLM 로드 {time.time() - t0:.0f}s · 문제 {len(probs)} × (원본+rename) × 1회 · max_new {a.max_new}", flush=True)

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}],   # noqa: F821
                                       tokenize=False, add_generation_prompt=True)

    plan = []
    for p in probs:
        rn = edit_rename(p["problem"])                                                     # noqa: F821
        plan.append((p, "orig", p["problem"], False))
        plan.append((p, "rename", rn or p["problem"], rn is None))
    t1 = time.time()
    outs = llm.generate([chat(t) for _, _, t, _ in plan], sp)
    ntok = sum(len(o.token_ids) for out in outs for o in out.outputs)
    print(f"[H14b] 생성 {len(plan)}프롬프트 · {time.time() - t1:.0f}s · {ntok / max(1, time.time() - t1):.1f} tok/s", flush=True)

    rec = {p["pidx"]: {"pidx": p["pidx"], "y": p["y"], "problem": p["problem"], "by_kind": {}} for p in probs}
    for (p, kd, _, degen), out in zip(plan, outs):
        lp, ent, fm, nt, txt = stats(out, tok)
        rec[p["pidx"]]["by_kind"][kd] = {"mean_logprob": lp, "mean_entropy_top20": ent,
                                         "first_math_tok": fm, "tokens": nt, "degenerate": bool(degen), "text_head": txt}
    for r in rec.values():
        b, v = r["by_kind"].get("orig"), r["by_kind"].get("rename")
        if not b or not v or v["degenerate"]:
            r["d_logprob"] = r["d_entropy"] = r["d_first_math"] = None
            continue
        r["d_logprob"] = v["mean_logprob"] - b["mean_logprob"]
        r["d_entropy"] = v["mean_entropy_top20"] - b["mean_entropy_top20"]
        r["d_first_math"] = v["first_math_tok"] - b["first_math_tok"]
    ok = [r for r in rec.values() if r["d_logprob"] is not None]
    print(f"[H14b] 지표 산출 {len(ok)}/{len(rec)} (rename 이 원문과 같은 문제 {len(rec) - len(ok)})", flush=True)
    for key in ("d_logprob", "d_entropy", "d_first_math"):
        rb = [r[key] for r in ok if r["y"]]
        sp_ = [r[key] for r in ok if not r["y"]]
        print(f"  {key}: robust 평균 {sum(rb) / max(1, len(rb)):+.4f} · spurious 평균 {sum(sp_) / max(1, len(sp_)):+.4f}", flush=True)
    Path(a.out, f"{alias}_h14b.json").write_text(json.dumps(list(rec.values())), encoding="utf-8")
    print("[H14b] 저장 완료", flush=True)


if __name__ == "__main__":
    main()
