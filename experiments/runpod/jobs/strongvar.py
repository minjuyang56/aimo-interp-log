"""센 섭동(Domain ×2 · Rephrase ×1) 생성 + 정답 보존 2단계 검증. RunPod H100. (지휘 #STRONG)

**한 번만 돌리고 세 모델이 같은 섭동을 쓴다.** 모델별로 다시 만들면 비용이 3배이고, 더 중요하게는
모델 간 비교가 깨진다(같은 문제에 서로 다른 섭동이 들어가면 '모델 차이' 와 '섭동 차이' 가 섞인다).

재작성기 = `Qwen/Qwen2.5-32B-Instruct` (공식 모델 카드에서 `license: apache-2.0` 확인, 2026-10-10).
피험 모델(Skywork·Olmo·DeepSeek)이 아닌 더 큰 모델을 쓰는 이유: 피험 모델이 자기 섭동을 만들면
그 모델에 유리하게 편향될 수 있다. temperature 0 — 재현돼야 한다.

출력 `strong_variants.json` = {pidx: [{"kind": ..., "text": ...}, …]} + `strongvar_report.txt`.
라벨 생성 쪽은 `labelgen32k.py --strong strong_variants.json` 으로 받아 기존 3유형에 **덧붙인다**
→ 합계 6유형(논문과 같은 수).

사용(파드 안):
  python3 strongvar.py --problems problems.json --out /workspace/out [--rewriter Qwen/Qwen2.5-32B-Instruct]
  `problems.json` 은 [{"pidx": 40, "problem": "..."}, …] 꼴 (라벨/선별 파일에서 뽑아 올린다).
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from variants_strong import (JUDGE_PROMPT, STRONG_PROMPTS, acceptable,  # noqa: E402
                             clean_rewrite, judge_ok)


def log(m):
    print(f"[단계] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--rewriter", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="앞 N 문제만 (파일럿용)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # **파일명 방어 (2026-10-10 에 당했다).** 잡 정의가 기대하는 이름과 --upload 로 올린 이름이
    # 어긋나면 모델을 올리기 전에 FileNotFoundError 로 죽는다($0.20 로 드러났다).
    # 준 경로가 없으면 작업 디렉터리에서 problems*.json 을 찾는다.
    import glob as _g
    pf = a.problems
    if not Path(pf).is_file():
        alt = sorted(_g.glob("problems*.json")) or sorted(_g.glob(Path(pf).name))
        if not alt:
            raise SystemExit(f"문제지를 못 찾았다: {pf} · 작업 디렉터리 목록 {sorted(_g.glob('*.json'))}")
        log(f"문제지 경로가 안 맞아 대체: {pf} -> {alt[0]}")
        pf = alt[0]
    rows = json.loads(Path(pf).read_text(encoding="utf-8"))
    if a.limit:
        rows = rows[:a.limit]
    log(f"재작성기 {a.rewriter} · 문제 {len(rows)} · 유형 {[k for k, _ in STRONG_PROMPTS]}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.rewriter)
    t0 = time.time()
    llm = LLM(model=a.rewriter, dtype="bfloat16", gpu_memory_utilization=0.92,
              max_model_len=4096, enforce_eager=True, seed=0)
    log(f"vLLM 로드 {time.time() - t0:.0f}s")

    def chat(sys_user):
        return tok.apply_chat_template([{"role": "user", "content": sys_user}],
                                       tokenize=False, add_generation_prompt=True)

    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_new)
    judge_sp = SamplingParams(n=1, temperature=0.0, max_tokens=8)

    # ---- 1) 재작성
    prompts, owner = [], []
    for r in rows:
        for kind, head in STRONG_PROMPTS:
            prompts.append(chat(head + r["problem"]))
            owner.append((r["pidx"], kind, r["problem"]))
    log(f"재작성 {len(prompts)}건 생성 시작")
    t0 = time.time()
    outs = llm.generate(prompts, greedy)
    log(f"재작성 완료 {time.time() - t0:.0f}s")

    cand = []
    for (pidx, kind, orig), o in zip(owner, outs):
        rw = clean_rewrite(o.outputs[0].text, orig)
        cand.append({"pidx": pidx, "kind": kind, "original": orig, "text": rw})

    # ---- 2) 1단계(숫자) 통과한 것만 자기 판정으로 보낸다 (판정 호출을 아끼려고)
    from variants_strong import numbers_preserved
    stage1 = [c for c in cand if c["text"] and len(c["text"]) >= 25
              and c["text"].strip() != c["original"].strip() and numbers_preserved(c["original"], c["text"])]
    log(f"1단계(숫자 다중집합) 통과 {len(stage1)}/{len(cand)}")

    # ---- 3) 자기 판정 — **양방향**으로 두 번 묻는다(한 방향만 물으면 YES 로 쏠린다)
    jp, jown = [], []
    for c in stage1:
        jp.append(chat(JUDGE_PROMPT.format(a=c["original"], b=c["text"])))
        jown.append((id(c), "ab"))
        jp.append(chat(JUDGE_PROMPT.format(a=c["text"], b=c["original"])))
        jown.append((id(c), "ba"))
    judg = {}
    if jp:
        t0 = time.time()
        jouts = llm.generate(jp, judge_sp)
        log(f"자기 판정 {len(jp)}건 {time.time() - t0:.0f}s")
        for (cid, which), o in zip(jown, jouts):
            judg.setdefault(cid, {})[which] = o.outputs[0].text

    # ---- 4) 최종 채택
    accepted, rejected = {}, []
    for c in cand:
        js = judg.get(id(c), {})
        ok, why = acceptable(c["original"], c["text"], [js.get("ab", ""), js.get("ba", "")]) \
            if id(c) in judg else (False, "1단계 탈락")
        if ok:
            accepted.setdefault(str(c["pidx"]), []).append({"kind": c["kind"], "text": c["text"]})
        else:
            rejected.append({"pidx": c["pidx"], "kind": c["kind"], "why": why,
                             "judge": [js.get("ab", "")[:20], js.get("ba", "")[:20]],
                             "text": c["text"][:200]})

    (out / "strong_variants.json").write_text(json.dumps(accepted, ensure_ascii=False), encoding="utf-8")

    # ---- 5) 보고 — **탈락률이 0 에 가까우면 검증이 무력하다는 뜻**이므로 그대로 적는다
    lines = [f"센 섭동 생성 — 재작성기 {a.rewriter} · 문제 {len(rows)}", ""]
    n_tot = len(cand)
    n_acc = sum(len(v) for v in accepted.values())
    lines += [f"재작성 시도 {n_tot} · 1단계 통과 {len(stage1)} · **최종 채택 {n_acc} "
              f"({n_acc / max(1, n_tot):.0%})** · 탈락 {n_tot - n_acc}", ""]
    lines.append("유형별 채택:")
    for kind, _ in STRONG_PROMPTS:
        tried = sum(1 for c in cand if c["kind"] == kind)
        got = sum(1 for v in accepted.values() for x in v if x["kind"] == kind)
        lines.append(f"  {kind}: {got}/{tried}")
    lines += ["", f"탈락 사유 분포: {dict(Counter(r['why'] for r in rejected))}", ""]
    lines.append("문제당 확보된 센 섭동 수 분포: "
                 f"{dict(sorted(Counter(len(v) for v in accepted.values()).items()))}")
    lines += ["", "탈락 예시 (최대 8건):"]
    for r in rejected[:8]:
        lines.append(f"  pidx {r['pidx']} {r['kind']} · {r['why']} · 판정 {r['judge']}")
        lines.append(f"    {r['text'][:140]}")
    lines += ["", "**읽는 법**: 탈락률이 0 에 가까우면 자기 판정이 무력하다는 뜻이다(모델이 기계적으로 YES).",
              "그 경우 센 섭동의 '정답 보존' 을 신뢰할 수 없으므로 라벨에 쓰기 전에 사람이 표본을 봐야 한다."]
    (out / "strongvar_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    # 품질 확인용 샘플 — 원문 그대로 (지휘 #STRONG: 재작성 품질 샘플 3개를 원문 그대로 보여줄 것)
    sample = []
    for pidx, vs in list(accepted.items())[:3]:
        orig = next(c["original"] for c in cand if str(c["pidx"]) == pidx)
        sample.append({"pidx": pidx, "original": orig, "variants": vs})
    (out / "strongvar_samples.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    log(f"저장: strong_variants.json · strongvar_report.txt · strongvar_samples.json")


if __name__ == "__main__":
    main()
