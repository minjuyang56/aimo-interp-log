"""판별 실험 — 미완주로 떨어진 spurious 가 진짜인지 예산 절단 아티팩트인지 가른다. (지휘 #40-1)

배경: v3 Skywork 라벨 38건(robust 24 / spurious 14) 중 **spurious 13건이 "변형이 토큰 상한을 넘겼다"** 때문이었다.
      그러면 라벨은 "추론이 깨졌나"가 아니라 "생성이 길어졌나"를 재는 것이다.
절차: 그 13문제에 대해 **원본과 최악 변형을 같은 조건으로** max_new 를 크게 올려(기본 16384) k=4 재샘플.
      원본도 같이 올리는 이유: 상한이 base_acc 자체를 바꿀 수 있어 비교가 공정해야 한다.
분류: 새 max_drop = new_base_acc − new_var_acc
      >= 0.25  -> **진짜 spurious** (상한을 늘려도 무너진다)
      <  0.25  -> **절단 아티팩트** (라벨에서 빼야 한다)
출력: /kaggle/working/genlab/<별칭>_probe13.json — 문제별 원본·변형의 완주 수·정답 수·**생성 토큰 수**(길이비 계산용)

러너: kaggle_h10b.py --header genlab_probe13_kaggle.py --body variants_official.py,variants_nonofficial.py,genlab_probe13.py
      --resume-from <v3 커널 slug>  (라벨 JSON 을 /kaggle/input 에서 읽는다)
"""

import argparse
import glob
import json
import time
from pathlib import Path


def build(labels, k):
    """미완주 우세로 spurious 가 된 문제만 골라 (원본, 최악 변형) 프롬프트 계획을 만든다."""
    out = []
    for s in labels:
        if s.get("label") is not False:
            continue
        wc = s.get("worst_cause") or {}
        if wc.get("n_unfin", 0) <= wc.get("n_wrong", 0):
            continue                                        # 오답 우세는 이미 진짜 spurious
        worst = s["worst_kind"]
        vt = dict(official_variants(s["problem"], seed=s["pidx"])).get(worst)     # noqa: F821
        if vt is None:
            continue
        out.append({"pidx": s["pidx"], "gold": s["gold"], "problem": s["problem"], "worst_kind": worst,
                    "variant": vt, "old_base_acc": s["base_acc"], "old_var_acc": s["var_acc"][worst],
                    "old_max_drop": s["max_drop"], "old_cause": wc})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=globals().get("RUN_MODEL", "Skywork/Skywork-OR1-Math-7B"))
    ap.add_argument("--labels", default=globals().get("RUN_LABELS", "/kaggle/input/**/*_v3_labels.json"))
    ap.add_argument("--k", type=int, default=globals().get("RUN_K", 4))
    ap.add_argument("--max-new", type=int, default=globals().get("RUN_MAX_NEW", 16384))
    ap.add_argument("--max-seqs", type=int, default=globals().get("RUN_MAX_SEQS", 16))
    ap.add_argument("--out", default="/kaggle/working/genlab")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    alias = a.model.split("/")[-1].replace(".", "_")

    hits = sorted(glob.glob(a.labels, recursive=True))
    if not hits:
        print(f"라벨 파일 없음: {a.labels}", flush=True); return
    labels = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    plan = build(labels, a.k)
    print(f"[판별] 라벨 {len(labels)}건 · 미완주 우세 spurious **{len(plan)}건** · max_new {a.max_new} · k={a.k}", flush=True)
    if not plan:
        return

    import torch
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, dtype="half", tensor_parallel_size=max(1, torch.cuda.device_count()),
              max_num_seqs=a.max_seqs, max_model_len=a.max_new + 2048, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=a.k, temperature=0.6, top_p=0.95, max_tokens=a.max_new, seed=0)
    print(f"[판별] vLLM 로드 {time.time() - t0:.0f}s", flush=True)

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}],    # noqa: F821
                                       tokenize=False, add_generation_prompt=True)

    prompts = [chat(p["problem"]) for p in plan] + [chat(p["variant"]) for p in plan]
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    ntok = sum(len(o.token_ids) for out in outs for o in out.outputs)
    print(f"[판별] 생성 {len(prompts)}프롬프트 × k={a.k} · {time.time() - t1:.0f}s · {ntok / max(1, time.time() - t1):.1f} tok/s", flush=True)

    n = len(plan)
    for i, p in enumerate(plan):
        for tag, out in (("base", outs[i]), ("var", outs[n + i])):
            fin = [o for o in out.outputs if o.finish_reason == "stop"]
            hit = sum(1 for o in fin if extract_answer(o.text) == p["gold"])                          # noqa: F821
            p[f"new_{tag}_acc"] = hit / a.k
            p[f"new_{tag}_nfin"] = len(fin)
            p[f"new_{tag}_tokens"] = [len(o.token_ids) for o in out.outputs]
        p["new_max_drop"] = p["new_base_acc"] - p["new_var_acc"]
        p["verdict"] = "real_spurious" if p["new_max_drop"] >= 0.25 else "truncation_artifact"
        p["length_ratio"] = (sum(p["new_var_tokens"]) / max(1, len(p["new_var_tokens"]))) / \
                            max(1.0, sum(p["new_base_tokens"]) / max(1, len(p["new_base_tokens"])))

    real = [p for p in plan if p["verdict"] == "real_spurious"]
    art = [p for p in plan if p["verdict"] == "truncation_artifact"]
    print(f"[판별 결과] 진짜 spurious **{len(real)}** / 절단 아티팩트 **{len(art)}** (총 {len(plan)})", flush=True)
    print(f"  원본 정확도 변화: 옛 평균 {sum(p['old_base_acc'] for p in plan) / len(plan):.2f} -> "
          f"새 평균 {sum(p['new_base_acc'] for p in plan) / len(plan):.2f} (상한 {a.max_new})", flush=True)
    print(f"  변형 정확도 변화: 옛 {sum(p['old_var_acc'] for p in plan) / len(plan):.2f} -> "
          f"새 {sum(p['new_var_acc'] for p in plan) / len(plan):.2f}", flush=True)
    print(f"  길이비(변형/원본) 평균 {sum(p['length_ratio'] for p in plan) / len(plan):.2f} · "
          f"진짜 {sum(p['length_ratio'] for p in real) / max(1, len(real)):.2f} · 아티팩트 {sum(p['length_ratio'] for p in art) / max(1, len(art)):.2f}", flush=True)
    for p in plan:
        print(f"   pidx {p['pidx']} {p['worst_kind']:9s} base {p['old_base_acc']:.2f}->{p['new_base_acc']:.2f} "
              f"var {p['old_var_acc']:.2f}->{p['new_var_acc']:.2f} drop {p['new_max_drop']:+.2f} len× {p['length_ratio']:.2f} {p['verdict']}", flush=True)
    Path(a.out, f"{alias}_probe13.json").write_text(json.dumps(plan), encoding="utf-8")
    print("[판별] 저장 완료", flush=True)


if __name__ == "__main__":
    main()
