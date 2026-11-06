"""#TT-PERTURB (루프 R1) 검증 생성 — RunPod H100. **제출 라이브러리를 그대로 import 한다.**

왜 그렇게 하나: 검증과 제출물이 다른 코드면 '방법이 나쁘다' 와 '코드가 달라서 나쁘다' 가 섞인다.
h22 가 같은 방식을 썼고(#RP ①: "제출물 h22_lib 의 함수를 그대로 불렀다") 이번엔 생성 엔진까지
**HF generate 로 같게** 맞췄다. vLLM 이 더 싸지만 엔진 차이를 사지 않는다.

한 번 생성해서 라운드 2~8 을 **$0 오프라인 격자**로 돌릴 재료를 전부 저장한다:
  * 모델 4종 × 공식 33문제 × (원본 greedy 1 + 섭동 K개) 를 cap 1024 로 생성
  * 체크포인트 **0 · 256 · 512 · 1024** 에서 각각 강제 즉답 → CoT 예산 4칸이 공짜로 생긴다
    (ckpt 0 = CoT 없이 즉답 = h21c 와 같은 양 → '확신도 신호 결합' 라운드 재료)
  * 원본 **T=1.0 · top_k 40 샘플 3개**(라벨 생성과 같은 디코딩) → 자기일관성 항 재료
  * 상설 규칙 ①: CoT 원문·즉답 문자열·추출답·섭동 kind·토큰 수를 **전량 저장**
  * 상설 규칙 ②: 모델마다 **첫 샘플 원문을 로그에 찍는다**(마커·추출 결함 육안 확인용)

모델 하나가 끝날 때마다 **즉시 저장**한다 — 수명 상한에 걸려도 앞 모델은 남고 폴링이 5분마다
회수한다(#RP 운영 구멍 3: 끝에만 회수해서 $6.04 를 잃은 사고).

사용(파드 안):  python3 ttperturb.py --problems org33_problems.json --cap 1024
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/workspace/out")
sys.path.insert(0, str(HERE))          # flatten 되어 ttp_lib.py 가 같은 폴더에 온다

MODELS = ["Skywork/Skywork-OR1-Math-7B", "Qwen/Qwen3.5-4B",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "allenai/Olmo-3-7B-Think"]
# Pass A (R1 주지표) — greedy · 섭동 8종 · 주 체크포인트 512 (1024 칸은 R2 예산으로 돌렸다)
CKPTS_A = [int(x) for x in (os.environ.get("TTP_CKPTS_A") or "0,256,512").split(",")]
# Pass B (R2 재료) — 같은 섭동을 N_SAMP 번 · 주최측에 가까운 디코딩 · 섭동 부분집합 4종
CKPTS_B = [int(x) for x in (os.environ.get("TTP_CKPTS_B") or "0,512").split(",")]
SUBSET_B = tuple((os.environ.get("TTP_SUBSET_B") or
                  "rename,rename2,typos0,distract0").split(","))
# Pass A 도 같은 부분집합으로 줄일 수 있다(R2 는 섭동 2종만 쓰므로 8종을 greedy 로 살 이유가 없다).
SUBSET_A = tuple(x for x in (os.environ.get("TTP_SUBSET_A") or "").split(",") if x) or None
N_SAMP = 3
SAMP_TEMP, SAMP_TOPK = 1.0, 40


def log(m):
    print(f"[단계 {time.time() - T0:7.1f}s] {m}", flush=True)


T0 = time.time()


def alias(m):
    return m.split("/")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="org33_problems.json")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--slots", type=int, default=0, help="0 이면 VRAM 으로 자동 계산")
    ap.add_argument("--order", default=None)
    ap.add_argument("--samples", type=int, default=N_SAMP)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import ttp_lib as T
    T.SOFT = 1e9                        # 검증에서는 제출 예산 시계를 쓰지 않는다(파드 수명이 상한이다)

    probs = json.loads(Path(a.problems).read_text(encoding="utf-8"))
    if isinstance(probs, dict):
        # org33_problems.json 은 {model_id: [{pidx, problem}, …]} 이고 모델 간 리스트가 동일하다(확인함)
        probs = probs.get("problems", list(probs.values())[0])
    texts = [p["problem"] if isinstance(p, dict) else p for p in probs]
    pidxs = [p.get("pidx", i) if isinstance(p, dict) else i for i, p in enumerate(probs)]
    log(f"문제 {len(texts)}개 (pidx {pidxs[0]}..{pidxs[-1]}) · cap {a.cap}")
    log(f"  Pass A (R1): greedy · 섭동 8종 · 체크포인트 {CKPTS_A}")
    log(f"  Pass B (R2): 섭동 {SUBSET_B} · {a.samples}샘플 · T={SAMP_TEMP} top_k={SAMP_TOPK} · "
        f"체크포인트 {CKPTS_B}")

    perts = [T.perturbations(t) for t in texts]
    npert = sum(len(p) for p in perts)
    log(f"섭동 총 {npert}개 (문제당 평균 {npert / len(texts):.2f}) · 유형 분포 "
        + str({k: sum(1 for ps in perts for kk, _ in ps if kk == k)
               for k in ("rename", "rename2", "typos0", "typos1", "distract0", "distract1",
                         "reorder", "notation")}))

    order = [m for m in (a.order.split(",") if a.order else MODELS)]
    for mid in order:
        t0 = time.time()
        tok, model = T.get_model(mid)
        if model is None:
            log(f"!! {mid} 로드 실패 — 건너뛴다")
            continue
        slots = a.slots or T.token_slots(model, a.cap)
        log(f"{alias(mid)} 로드 {time.time() - t0:.0f}s · 배치칸 {slots} "
            f"(토큰당 KV {T.kv_kb_per_token(model):.0f} KB)")

        # ---- Pass A (R1 주지표): 문제마다 원본 1 + 섭동 8종, greedy
        flat, owner = [], []
        for i, (t, ps) in enumerate(zip(texts, perts)):
            flat.append(t)
            owner.append((i, "orig", -1))
            for j, (k, v) in enumerate(ps):
                if SUBSET_A is not None and k not in SUBSET_A:
                    continue
                flat.append(v)
                owner.append((i, k, j))
        tg = time.time()
        cots, res = T.cot_then_answer_multi(tok, model, flat, a.cap, slots,
                                            deadline=time.time() + 36000, ckpts=CKPTS_A)
        log(f"  Pass A greedy {len(flat)} 시퀀스 생성+즉답 {time.time() - tg:.0f}s")

        rows = []
        for n, ((i, kind, j), cot) in enumerate(zip(owner, cots)):
            rows.append(dict(pidx=pidxs[i], kind=kind, jdx=j, cot=cot, n_char=len(cot),
                             ans={str(c): res[c][n][0] for c in CKPTS_A},
                             val={str(c): res[c][n][1] for c in CKPTS_A}))

        # ---- Pass B (R2 재료): 원본 + 섭동 부분집합을 **같은 입력으로 N 번** 샘플링
        flat_b, owner_b = [], []
        for i, (t, ps) in enumerate(zip(texts, perts)):
            flat_b.append(t)
            owner_b.append((i, "orig"))
            for k, v in ps:
                if k in SUBSET_B:
                    flat_b.append(v)
                    owner_b.append((i, k))
        log(f"  Pass B 입력 {len(flat_b)} (문제당 평균 {len(flat_b) / len(texts):.2f}) × {a.samples}샘플")
        srows = []
        for s in range(a.samples):
            ts = time.time()
            cs, rs = T.cot_then_answer_multi(tok, model, flat_b, a.cap, slots,
                                             deadline=time.time() + 36000, ckpts=CKPTS_B,
                                             sample=True, temperature=SAMP_TEMP, top_k=SAMP_TOPK, seed=s)
            log(f"  Pass B 샘플 {s} ({len(flat_b)} 시퀀스) {time.time() - ts:.0f}s")
            for n, ((i, kind), cot) in enumerate(zip(owner_b, cs)):
                srows.append(dict(pidx=pidxs[i], kind=kind, sdx=s, cot=cot, n_char=len(cot),
                                  ans={str(c): rs[c][n][0] for c in CKPTS_B},
                                  val={str(c): rs[c][n][1] for c in CKPTS_B}))
            # 중간 저장 — 수명 상한에 걸려도 앞 샘플은 남는다 (#RP 운영 구멍 3)
            (OUT / f"{alias(mid)}_ttp_samp.json").write_text(
                json.dumps(srows, ensure_ascii=False), encoding="utf-8")

        # ---- 상설 규칙 ②: 원문 육안 확인용 샘플을 로그에 찍는다
        marks = sum(r["cot"].count("Ġ") + r["cot"].count("Ċ") for r in rows + srows)
        log(f"  BPE 마커(Ġ/Ċ) 총 {marks}건  (0 이 아니면 디토큰화 손상 — 해석 전에 감사한다)")
        for r in rows[:2]:
            log(f"  --- 원문 샘플 pidx {r['pidx']} {r['kind']} · CoT {r['n_char']}자 ---")
            log("  " + r["cot"][:400].replace("\n", " ") + " …")
            log("  즉답: " + " | ".join(f"ck{c}={r['ans'][str(c)][:24]!r}->{r['val'][str(c)]}"
                                        for c in CKPTS_A))
        # 추출 성공률 — 이 설계의 핵심 주장(강제 즉답이 절단을 무력화한다)의 직접 계측
        for nm, rs_, cks in (("A", rows, CKPTS_A), ("B", srows, CKPTS_B)):
            for c in cks:
                got = sum(1 for r in rs_ if r["val"][str(c)] is not None)
                log(f"  Pass {nm} ckpt {c:>4}: 정수 답 추출 {got}/{len(rs_)} "
                    f"({got / max(1, len(rs_)):.1%})")

        # ---- 즉시 저장
        for name, obj in (("rows", rows), ("samp", srows)):
            p = OUT / f"{alias(mid)}_ttp_{name}.json"
            p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            log(f"  저장 {p.name} {p.stat().st_size / 1e6:.2f} MB")
        meta = dict(model=mid, cap=a.cap, ckpts_a=CKPTS_A, ckpts_b=CKPTS_B,
                    subset_b=list(SUBSET_B), samp_temp=SAMP_TEMP, samp_topk=SAMP_TOPK,
                    slots=slots, samples=a.samples,
                    n_problems=len(texts), n_seq_greedy=len(flat), marks=marks,
                    seconds=round(time.time() - t0, 1))
        (OUT / f"{alias(mid)}_ttp_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
        log(f"{alias(mid)} 완료 {time.time() - t0:.0f}s")

    log("전체 완료")


if __name__ == "__main__":
    main()
