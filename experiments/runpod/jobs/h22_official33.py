"""H22 를 공식 train-main-v2 문제단위 라벨(33문제)에 실제로 돌린다. RunPod 용. (지휘 #RP ①)

질문 하나에 답하는 실험이다: **라벨 정의를 그대로 재현하면 리더보드 상위권의 86% 근처가 나오는가.**

**측정 로직은 제출물 코드를 그대로 쓴다** — 섭동 생성(`rule_perturbations`·`numbers_preserved`),
답 추출(`answer_of`), 무한반복 감지(`is_degenerate`), 점수(`score_one`)는 전부 `h22_lib` 의 함수다.
여기서 따로 구현하면 측정과 제출물이 갈라져 숫자가 제출물의 숫자가 아니게 된다.

**다른 점은 생성 엔진 하나다.** 제출물은 HF transformers 로 생성하지만(워커가 그렇게 돌린다) 여기서는
**vLLM** 을 쓴다. 이유는 비용이다: HF 패딩 배치는 cap 스텝을 전부 돌지만 vLLM 연속 배칭은 평균 토큰에서
끝나므로 같은 결과를 몇 배 싸게 얻는다. 프롬프트·채팅템플릿·temperature 0.6·top_p 0.95·cap 은 같게 맞췄고,
완주 판정은 vLLM 의 `finish_reason == "stop"`(= EOS)으로 HF 쪽 EOS 검사와 같은 뜻을 쓴다.
**점수에 영향을 주는 축은 바뀌지 않는다** — 바뀌는 것은 벽시계뿐이다.

단계:
  1. HF 에서 train-main-v2 를 받아 **문제 단위 라벨**을 만든다(모델 간 다수결 — H2 에서 11/11 일치 확인).
  2. 섭동을 만든다. rephrase 는 모델에게 바꿔쓰게 하고 **숫자 보존 검사**를 통과한 것만 쓰며,
     모자라면 규칙 섭동으로 K 를 메운다(제출물과 같은 규칙).
  3. 원본 N 샘플 + 섭동 K개 × S 샘플을 **한 번에** vLLM 에 넣고 `score_one` 으로 점수를 낸다.
  4. AUC·셔플 null·임계값 0.5 정확도를 내고 out/ 에 저장한다.

사용(파드 안):  python h22_official33.py --model Skywork/... --n-problems 5 --cap 24576
"""

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

OUT = Path("/workspace/out")
V2_URL = ("https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/"
          "data/train-00000-of-00001.parquet")

os.environ.setdefault("H22_SOFT", "100000000")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h22_lib as L                                  # noqa: E402


def log(m):
    print(f"[단계] {m}", flush=True)


def problem_labels():
    """문제 단위 라벨 — 같은 문제의 여러 모델 행을 다수결로 묶는다."""
    import pandas as pd
    raw = urllib.request.urlopen(urllib.request.Request(
        V2_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read()
    df = pd.read_parquet(io.BytesIO(raw))
    out = []
    for prob, g in df.groupby("problem", sort=True):
        vals = [bool(v) for v in g["is_robust"].tolist() if v is not None and v == v]
        if not vals:
            continue
        c = Counter(vals)
        if c[True] == c[False]:
            continue
        out.append({"problem": str(prob).strip(), "y": int(c[True] > c[False])})
    log(f"문제 단위 라벨 {len(out)}건 (robust {sum(r['y'] for r in out)})")
    return out


class VLLM:
    """vLLM 래퍼. 반환 형식을 h22_lib.generate 와 같게 맞춘다: [(텍스트, 완주여부), …]"""

    def __init__(self, model_id, cap):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tok = AutoTokenizer.from_pretrained(model_id)
        t0 = time.time()
        # **enforce_eager 기본 True.** Olmo3 는 vllm 0.31 + torch 2.13 에서 torch.compile 경로가
        # AssertionError: Source mismatch for collections.abc 로 죽는다(2026-10-10 실측).
        # Skywork(Qwen2)는 컴파일해도 통과하지만, 모델마다 다른 걸 기억하느니 끄고 쓴다 —
        # 디코딩이 조금 느려지는 대가로 아키텍처에 상관없이 돈다. H22_EAGER=0 으로 켤 수 있다.
        eager = os.environ.get("H22_EAGER", "1") == "1"
        self.llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.92,
                       max_model_len=cap + 2048, enforce_eager=eager, seed=0)
        log(f"vLLM 로드 {time.time() - t0:.0f}s")

    def _prompt(self, text, sysp, suffix=""):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": f"{sysp}\n\n{text}"}],
            tokenize=False, add_generation_prompt=True) + suffix

    def run(self, texts, cap, temp=0.6, sysp=None, suffix=""):
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=temp, top_p=0.95 if temp > 0 else 1.0,
                            max_tokens=cap, seed=None if temp > 0 else 0)
        prompts = [self._prompt(t, sysp or L.SOLVE_PROMPT, suffix) for t in texts]
        outs = self.llm.generate(prompts, sp)
        rows = []
        for o in outs:
            c = o.outputs[0]
            fin = (c.finish_reason == "stop") and not L.is_degenerate(c.text)
            rows.append((c.text, bool(fin)))
        return rows


def build_perturbations(eng, texts, K):
    """rephrase(모델) + 규칙 섭동. 제출물 h22_lib.rephrase 와 **같은 규칙**으로 거른다."""
    perts = [L.rule_perturbations(t)[:max(0, K - 2)] for t in texts]
    for p_i, sysp in enumerate(L.REPHRASE_PROMPTS):
        rows = eng.run(texts, L.REPHRASE_CAP, temp=0.0, sysp=sysp, suffix=L.REPHRASE_FORCE)
        kept = 0
        for i, (txt, fin) in enumerate(rows):
            cand = (txt or "").strip().split("\n\n")[0].strip()
            if not fin or len(cand) < 20 or cand == texts[i] or not L.numbers_preserved(texts[i], cand):
                continue
            perts[i].append((f"rephrase{p_i}", cand))
            kept += 1
        log(f"rephrase {p_i}: 숫자보존 통과 {kept}/{len(texts)}")
    for i, t in enumerate(texts):                  # 모자라면 규칙 섭동으로 메운다
        if len(perts[i]) < K:
            have = {v for _, v in perts[i]}
            perts[i] += [(k, v) for k, v in L.rule_perturbations(t) if v not in have]
        perts[i] = perts[i][:K]
    log(f"섭동 수 분포 {Counter(len(p) for p in perts)} (목표 K={K})")
    return perts


def evaluate(results, out_path):
    """AUC·셔플 null·정확도. 판정된 문제만 쓴다(중립은 따로 센다)."""
    import numpy as np
    dec = [r for r in results if r["score"] is not None and not (r["diag"] or {}).get("neutral")]
    lines = [f"H22 공식 33문제 — 판정 {len(dec)}/{len(results)} · 중립 {len(results) - len(dec)}"]
    if len(dec) < 8 or len({r["y"] for r in dec}) < 2:
        lines.append("판정된 문제가 너무 적어 AUC 를 내지 않는다.")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        return
    y = np.array([r["y"] for r in dec])
    s = np.array([float(r["score"]) for r in dec])
    rob = -s                                        # 점수는 '클수록 spurious' 이므로 뒤집어 robust 쪽 AUC
    o = rob.argsort(kind="mergesort")
    rk = np.empty(len(rob), float)
    rk[o] = np.arange(1, len(rob) + 1, dtype=float)
    for v in np.unique(rob):
        m = rob == v
        rk[m] = rk[m].mean()
    n1, n0 = float(y.sum()), float((1 - y).sum())
    auc = (float(rk[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
    rng = np.random.default_rng(0)
    nl = np.array([(rng.permutation(y) @ rk - n1 * (n1 + 1) / 2) / (n1 * n0) for _ in range(1000)])
    lo, hi = np.percentile(nl, [2.5, 97.5])
    acc = float(((s < 0.5) == (y == 1)).mean())
    base = max(float((y == 1).mean()), float((y == 0).mean()))
    lines += [
        f"AUC(robust) {auc:.3f} {'▲' if auc > hi else ('▼' if auc < lo else ' ')} · 셔플 null 95% [{lo:.2f}, {hi:.2f}]",
        f"임계값 0.5 정확도 {acc:.3f} · 상수 예측 기준선 {base:.3f} · n={len(dec)}",
        f"robust 평균 score {s[y == 1].mean():.3f} vs spurious {s[y == 0].mean():.3f}",
        f"score 분포 {dict(sorted(Counter(np.round(s, 3).tolist()).items()))}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-problems", type=int, default=5)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--cap", type=int, default=24576)
    ap.add_argument("--chunk", type=int, default=8,
                    help="이 문제 수마다 생성·채점·저장을 끊는다 (시간 초과로 전부 잃지 않게)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", a.model.split("/")[-1])

    rows = problem_labels()
    if a.n_problems > 0:
        rows = rows[:a.n_problems]
    log(f"모델 {a.model} · 문제 {len(rows)} · N={a.N} K={a.K} S={a.S} cap={a.cap}")

    t_all = time.time()
    eng = VLLM(a.model, a.cap)
    texts = [r["problem"] for r in rows]
    perts = build_perturbations(eng, texts, a.K)

    # **문제를 조각으로 나눠 조각마다 저장한다.** vLLM 의 generate() 는 전부 끝나야 돌아오므로
    # 한 번에 다 넣으면 시간 초과·파드 종료에 **전부 날아간다**(첫 실행에서 526시퀀스를 한 번에
    # 넣었다가 겪었다). 조각을 쪼개면 배칭 효율이 조금 떨어지지만 잃는 것이 한 조각으로 줄어든다.
    results, gen_s = [], 0.0
    out_path = OUT / f"{tag}_h22.json"
    for c0 in range(0, len(texts), a.chunk):
        idxs = list(range(c0, min(c0 + a.chunk, len(texts))))
        seqs, owner = [], []
        for i in idxs:
            for _ in range(a.N):
                seqs.append(texts[i])
                owner.append((i, "orig"))
            for q, (kind, v) in enumerate(perts[i]):
                for _ in range(a.S):
                    seqs.append(v)
                    owner.append((i, f"p{q}"))
        log(f"조각 {c0 // a.chunk + 1}/{-(-len(texts) // a.chunk)} — 문제 {idxs[0] + 1}~{idxs[-1] + 1} "
            f"· 시퀀스 {len(seqs)}개 · cap {a.cap}")
        t0 = time.time()
        rowsg = eng.run(seqs, a.cap)
        el = time.time() - t0
        gen_s += el
        log(f"  조각 생성 {el:.0f}s · 완주 {sum(1 for _, f in rowsg if f)}/{len(rowsg)}")

        bucket = {}
        for (i, slot), (txt, fin) in zip(owner, rowsg):
            bucket.setdefault((i, slot), []).append((L.answer_of(txt), fin))
        for i in idxs:
            oa = bucket.get((i, "orig"), [])
            pa = [bucket[(i, f"p{q}")] for q in range(len(perts[i])) if (i, f"p{q}") in bucket]
            sc, diag = L.score_one(oa, pa)
            note = (diag.get("why")
                    or f"원본완주 {diag.get('n_fin_orig')} · 무너진섭동 {diag.get('broken')}/{diag.get('n_pert')}")
            log(f"문제 {i + 1}/{len(rows)} y={rows[i]['y']} score={sc} ({note})")
            results.append({"problem": rows[i]["problem"][:200], "y": rows[i]["y"], "score": sc,
                            "diag": diag, "kinds": [k for k, _ in perts[i]], "cap": a.cap,
                            "N": a.N, "K": a.K, "S": a.S})
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"  저장 {out_path.name} ({len(results)}문제) · 누적 생성 {gen_s:.0f}s")
        evaluate(results, OUT / f"{tag}_report.txt")      # 중간 평가도 남긴다

    per = gen_s / max(1, len(rows))
    log(f"문제당 생성 {per:.0f}s · 전체 33문제 추정 {per * 33 / 3600:.2f}h "
        f"(+ 로드) · 4모델이면 {per * 33 * 4 / 3600:.2f}h")
    evaluate(results, OUT / f"{tag}_report.txt")
    n_seq_total = sum(a.N + a.S * len(perts[i]) for i in range(len(texts)))
    (OUT / f"{tag}_summary.json").write_text(json.dumps(
        {"model": a.model, "n": len(rows), "gen_seconds": round(gen_s, 1),
         "per_problem_s": round(per, 1), "total_seconds": round(time.time() - t_all, 1),
         "cap": a.cap, "N": a.N, "K": a.K, "S": a.S, "chunk": a.chunk,
         "n_seq": n_seq_total, "n_problems_done": len(results)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    log("완료")


if __name__ == "__main__":
    main()
