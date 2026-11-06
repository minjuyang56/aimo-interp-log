"""H21cK 측정 — 상수 변형 K개의 drop. 로컬 4bit, 모델 하나씩. (지휘 #69 사전등록)

문제당: 즉답 8토큰 강제 → 그 답 문자열을 **원본 + 변형 24개**(개선규칙 v2 16개 + 옛규칙 v1 8개)에서 채점.
K=4·8 은 저장된 앞쪽 부분집합으로 쓴다 — 같은 즉답·같은 생성에서 나오므로 K 간 비교가 공정하다.

주지표(사전등록): drop 평균. 보조: 중앙값. 부호: **클수록 robust**(h21c 와 같음, 뒤집지 않는다).

사용:  .venv/Scripts/python.exe experiments/h21ck.py --model Qwen/Qwen3.5-4B
결과:  experiments/results/h21ck/<별칭>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from h21_uncertainty import chat, load, problems, score_continuation      # noqa: E402
from variants_constants import make_variants                             # noqa: E402

OUT = ROOT / "experiments" / "results" / "h21ck"
MERGED = ROOT / "experiments" / "results" / "genlab" / "skywork_labels_merged.json"
K_V2 = 16                      # 개선 규칙으로 만들 변형 수 (K=4·8·16 을 모두 덮는다)
K_V1 = 8                       # 옛 규칙 비교용 (K=8 전후 비교)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--ans-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default="official", choices=["official", "own"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.source == "own":
        # 자체 생성 라벨 88문제(Skywork run0+run50 병합). 라벨 두 정의를 그대로 들고 간다.
        d = json.loads(MERGED.read_text(encoding="utf-8"))
        rows = [{"pidx": r["pidx"], "problem": r["problem"], "y": int(bool(r["label"])),
                 "y_fin": None if r.get("label_fin") is None else int(bool(r["label_fin"])),
                 "per_model": {}} for r in d if isinstance(r.get("label"), bool)]
        print(f"[h21cK] 자체 라벨 {len(rows)}문제 "
              f"(label robust {sum(r['y'] for r in rows)} / spurious {sum(1 - r['y'] for r in rows)})", flush=True)
    else:
        rows = problems()
    if a.limit:
        rows = rows[:a.limit]
    print(f"[h21cK] {a.model} · 문제 {len(rows)} · 변형 v2 {K_V2} + v1 {K_V1}", flush=True)
    tok, model = load(a.model, a.quant)
    import torch
    out, t0 = [], time.time()
    for n, r in enumerate(rows, 1):
        text = r["problem"]
        p_orig = chat(tok, text, force=True)
        enc = tok(p_orig, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=a.ans_tokens,
                               pad_token_id=tok.pad_token_id)
        raw = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ans = raw.splitlines()[0].strip() if raw.strip() else ""     # 첫 줄만 (수다 제외)
        rec = {"pidx": r["pidx"], "y": r["y"], "y_fin": r.get("y_fin"),
               "per_model": r.get("per_model", {}), "problem": text, "answer": ans, "drops": {}}
        if ans:
            lp0 = score_continuation(tok, model, p_orig, ans)
            rec["ans_logprob"] = lp0
            if lp0 is not None:
                for rule, k in (("v2", K_V2), ("v1", K_V1)):
                    for lab, var in make_variants(text, k, rule):
                        lp = score_continuation(tok, model, chat(tok, var, force=True), ans)
                        if lp is not None:
                            rec["drops"][lab] = lp0 - lp
        out.append(rec)
        d2 = [v for kk, v in rec["drops"].items() if kk.startswith("v2")]
        print(f"  [{n}/{len(rows)}] y={r['y']} 답 {ans[:10]!r} 변형 {len(rec['drops'])}개 · "
              f"v2 평균 {('%.3f' % (sum(d2) / len(d2))) if d2 else '-'} · {time.time() - t0:.0f}s", flush=True)
    alias = a.model.split("/")[-1].replace(".", "_")
    f = OUT / (f"{alias}.json" if a.source == "official" else f"{alias}_own.json")
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[h21cK] 저장 {f} · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
