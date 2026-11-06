"""#GEN-PROBE 단계 2 — 원문제 greedy 풀이 **재생성**. RunPod H100. (지휘 #GEN-PROBE)

## 왜 재생성하나
k=8 라벨 생성은 전부 `--no-traj` 로 돌았고 `genlab_v2.py` 는 `greedy_text` 를 그 분기에서만
저장한다 → **풀이 텍스트가 어느 산출물에도 없다**(단계 0 전수 확인: 라벨 파일 4종 모두 0건).
#GEN-PROBE 의 특징은 '풀이 도중 벡터' 이므로 풀이 텍스트가 없으면 아무것도 못 한다.

## 라벨 생성과 **같은 코드·같은 프롬프트**를 쓴다
`genlab_v2.chat()`(→ `solve_common.SOLVE_PROMPT`)을 그대로 불러 프롬프트를 만든다. 여기서
프롬프트를 다시 쓰면 특징이 라벨과 다른 입력에서 나온 것이 되어 비교가 깨진다. 네임스페이스
주입도 `labelgen32k.py` 와 같은 네 파일·같은 순서다.

## 단계 0 의 결함을 되풀이하지 않는다
행마다 **텍스트·토큰 수·완주여부·추출답·정답여부**를 저장한다. 토큰 수는 vLLM 이 준
`len(o.token_ids)` 를 쓴다(사후에 토크나이저로 다시 세면 특수토큰 처리가 달라질 수 있다).
모델 하나가 끝날 때마다 **즉시 저장**한다 — 수명 상한에 걸려도 앞 모델은 남고, 폴링이 5분마다
회수하기 때문이다(#RP 운영 구멍 3: 끝에만 회수해서 $6.04 를 잃은 사고).

사용(파드 안):  python3 genprobe.py --problems genprobe_problems.json --max-new 32768
"""

import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/workspace/out")
SF_N, SF_TEMP, SF_TOPK = 1, 0.0, 0


def log(m):
    print(f"[단계] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="genprobe_problems.json")
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--max-seqs", type=int, default=48)
    ap.add_argument("--order", default=None, help="모델 처리 순서(쉼표). 비우면 파일 순서")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # 샘플 모드 설정 (없으면 기존 greedy 경로와 **글자 그대로 같다**)
    global SF_N, SF_TEMP, SF_TOPK
    SF_N = int(os.environ.get("SF_N", 1) or 1)
    SF_TEMP = float(os.environ.get("SF_TEMP", 0.0) or 0.0)
    SF_TOPK = int(os.environ.get("SF_TOPK", 0) or 0)
    if SF_N > 1:
        log(f"[#SAMP-FEAT] 샘플 모드 n={SF_N} · T={SF_TEMP} · top_k={SF_TOPK} · cap {a.max_new}")

    # vLLM 컴파일 경로 우회 (2026-10-10 Olmo3 에서 AssertionError: Source mismatch for collections.abc)
    if os.environ.get("LG_EAGER", "1") == "1":
        import vllm
        _LLM = vllm.LLM

        def _patched(*args, **kw):
            kw.setdefault("enforce_eager", True)
            log("vLLM enforce_eager=True 로 생성 (컴파일 경로 우회)")
            return _LLM(*args, **kw)

        vllm.LLM = _patched

    ns = {"__name__": "genlab_v2_runpod"}
    for f in ("variants_official.py", "variants_nonofficial.py", "solve_common.py", "genlab_v2.py"):
        src = (HERE / f) if (HERE / f).is_file() else Path(f)
        log(f"로드 {src.name}")
        exec(compile(src.read_text(encoding="utf-8"), str(src), "exec"), ns)
    Gen, chat = ns["Gen"], ns["chat"]
    same_answer, extract_answer = ns["same_answer"], ns["extract_answer"]
    # **디토큰화 가드 (지휘 #DS-RELABEL, 2026-10-12).** 아래 vLLM 분기는 원래 `o.outputs[0].text`
    # 를 그대로 썼고, 그게 #TOK-AUDIT 가 찾아낸 손상 경로다(DeepSeek 41/41 행에 Ġ/Ċ).
    # 첫 실행은 사후 왕복 재디코드로 살렸지만, 보충 생성이 같은 경로로 또 손상 텍스트를 만들면
    # 같은 묶음 안에서 **일부 행만 다른 경로로 디코드된 특징**이 섞인다 → 비교가 깨진다.
    # 그래서 라벨 생성과 같은 `decode_out(o, dtok)` 로 통일한다.
    safe_detok, decode_out, marks_in = ns["safe_detok"], ns["decode_out"], ns["marks_in"]

    pf = Path(a.problems) if Path(a.problems).is_file() else Path(Path(a.problems).name)
    jobs = json.loads(pf.read_text(encoding="utf-8"))
    order = a.order.split(",") if a.order else list(jobs)
    log(f"문제 파일 {pf} · 모델 {len(order)}종 · 시퀀스 합 {sum(len(jobs[m]) for m in order)}")

    t_all = time.time()
    for mid in order:
        rows = jobs[mid]
        alias = mid.split("/")[-1].replace(".", "_")
        log(f"== {mid} · {len(rows)}문제 · greedy(T=0) · cap {a.max_new}")
        t0 = time.time()
        g = Gen(mid, a.max_new, a.max_seqs)
        t_load = time.time() - t0
        prompts = [chat(g.tok, r["problem"]) for r in rows]
        t1 = time.time()
        if g.kind == "vllm":
            from vllm import SamplingParams
            # **샘플 모드 (지휘 #SAMP-FEAT, 2026-10-12).** `SF_N` 이 없으면 아래는 정확히
            # 기존 greedy 호출(n=1 · T=0 · top_p 1.0)이고 기존 산출물이 그대로 재현된다.
            # `SF_N>1` 이면 라벨 생성과 같은 디코딩으로 n 샘플을 받아 **행마다 전부 저장**한다.
            if SF_N > 1:
                kw = dict(n=SF_N, temperature=SF_TEMP, top_p=0.95, max_tokens=a.max_new, seed=0)
                if SF_TOPK > 0:
                    kw["top_k"] = SF_TOPK
                sp = SamplingParams(**kw)
                log(f"샘플 모드: n={SF_N} · temperature={SF_TEMP} · top_k={SF_TOPK or '없음'} · "
                    f"top_p=0.95 · max_tokens={a.max_new}")
            else:
                sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=a.max_new, seed=0)
            outs = g.llm.generate(prompts, sp)
            dtok = safe_detok(mid)
            got, m_raw, m_safe, n_diff = [], 0, 0, 0
            for o in outs:
                grp = []
                for oo in o.outputs:
                    txt, mr, ms, diff = decode_out(oo, dtok)
                    m_raw += mr
                    m_safe += ms
                    n_diff += int(diff)
                    grp.append((txt, oo.finish_reason == "stop", len(oo.token_ids)))
                got.append(grp if SF_N > 1 else grp[0])
            log(f"[디토큰화 계측] 행 {len(got)} · vLLM o.text 마커 {m_raw} · "
                f"쓰는 텍스트 마커 {m_safe} · 두 텍스트가 다른 행 {n_diff}/{len(got)}")
            if m_safe:
                raise SystemExit(f"!! 쓰는 텍스트에 마커 {m_safe}건 — 저장하지 않는다")
            first = got[0][0][0] if SF_N > 1 else got[0][0]
            log(f"샘플 앞 160자: {first[:160]!r}")
        else:                                   # HF 폴백: 토큰 수는 토크나이저로 센다
            grp, _ = g.samples(prompts, 1, 0.0)
            got = [(gl[0][0], gl[0][1], len(g.tok(gl[0][0], add_special_tokens=False)["input_ids"]))
                   for gl in grp]
        dt = time.time() - t1
        # 샘플 모드면 got 의 각 원소가 **리스트**다 → 평평하게 센다.
        # (2026-10-12: 이 줄을 안 고쳐 ValueError 로 파드가 죽었다 — $0.43. 로컬 스모크 없이 띄운 대가다.)
        ntok = sum(n for grp in got for _, _, n in (grp if SF_N > 1 else [grp]))
        res = []
        if SF_N > 1:
            # **규칙 ① 대로 샘플 텍스트를 전량 저장한다.** 집계만 남기면 집계의 버그가 보이지 않는다.
            for r, grp in zip(rows, got):
                sams = []
                for txt, fin, n in grp:
                    ans = extract_answer(txt)
                    sams.append({"text": txt, "n_tok": int(n), "finished": bool(fin),
                                 "answer": None if ans is None else str(ans),
                                 "correct": bool(fin and same_answer(ans, r["gold"]))})
                res.append({**r, "samples": sams, "gen_cap": int(a.max_new), "greedy": False,
                            "n_samples": SF_N, "temperature": SF_TEMP, "top_k": SF_TOPK,
                            "n_correct": sum(1 for x in sams if x["correct"]),
                            "mean_tok": float(sum(x["n_tok"] for x in sams) / max(1, len(sams)))})
        else:
            for r, (txt, fin, n) in zip(rows, got):
                ans = extract_answer(txt)
                res.append({**r, "greedy_text": txt, "n_tok": int(n), "finished": bool(fin),
                            "answer": None if ans is None else str(ans),
                            "correct": bool(fin and same_answer(ans, r["gold"])),
                            "gen_cap": int(a.max_new), "greedy": True})
        p = OUT / (f"{alias}_samp.json" if SF_N > 1 else f"{alias}_greedy.json")
        p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        if SF_N > 1:
            nf = sum(1 for x in res for s_ in x["samples"] if s_["finished"])
            nc = sum(1 for x in res for s_ in x["samples"] if s_["correct"])
        else:
            nf = sum(1 for x in res if x["finished"])
            nc = sum(1 for x in res if x["correct"])
        # 샘플 모드면 n_tok 이 행이 아니라 samples 안에 있다 (같은 함정에 두 번 걸렸다).
        toks = (sorted(s_["n_tok"] for x in res for s_ in x["samples"]) if SF_N > 1
                else sorted(x["n_tok"] for x in res))
        log(f"[{alias}] 로드 {t_load:.0f}s · 생성 {dt:.0f}s · {ntok / max(1, dt):.1f} tok/s · "
            f"완주 {nf}/{len(toks)} · 정답 {nc}/{len(toks)} · "
            f"토큰 최소 {toks[0]} 중앙 {toks[len(toks) // 2]} 최대 {toks[-1]} · 저장 {p.name}")
        g.close()
    log(f"완료 {time.time() - t_all:.0f}s · 산출 {sorted(p.name for p in OUT.glob('*'))}")


if __name__ == "__main__":
    main()
