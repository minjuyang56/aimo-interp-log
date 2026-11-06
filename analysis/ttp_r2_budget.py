"""#LOOP R2 예산 표 — cap × 섭동 수 × 반복 수 조합의 **제출 시간**과 **검증 비용**. 비용 $0.

지휘 #LOOP R2 보완: "R2 사전등록 전에 cap 1024/2048 x 변형 수 x 반복 수 조합의 제출 시간·검증
비용 표를 먼저 계산하고, 3600초 안 + 라운드 $3 안에서 **가장 긴 예산**을 고른다."

제출 시간: 워커 RTX PRO 6000 96 GB · 합산 3600s(안전 3300s) · ms(B,L) = 1.15 x (15 + 0.375B +
7.21e-4 B L) · 프리필 토큰 상한 `MAX_PREFILL_TOK` 로 배치 칸이 묶인다(R1 OOM 에서 배운 것).
문제당 시퀀스 = (원본 1 + 섭동 K) x 반복 R  — R2 는 원본도 여러 번 풀려 `base_acc` 를 만든다.

검증 비용: H100 80 GB · 공식 33문제 x 4모델. 라운드 상한 $3 에서 **이미 쓴 $0** 을 빼고 남는 돈.
수명 상한은 **가장 비싼 GPU 후보 $3.99/h** 기준으로 잡는다(R1 에서 $0.85 를 그렇게 버렸다).

사용: .venv/Scripts/python.exe analysis/ttp_r2_budget.py
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "experiments/results/ttp_r2_budget.txt"

KV_KB = {"Skywork": 56, "Qwen3.5": 128, "DeepSeek": 144, "Olmo": 512}
WEIGHTS_GB = {"Skywork": 15.2, "Qwen3.5": 9.3, "DeepSeek": 144 and 16.4, "Olmo": 14.6}
PROMPT = 600
MAX_PREFILL = 40000          # 워커는 96 GB 라 더 줄 수도 있지만 보수적으로 R1 검증값을 쓴다
ANS_STEPS = 8
PRICE_WORST = 3.99           # 가장 비싼 GPU 후보 ($/h)
ROUND_CAP = 3.00
LOOP_LEFT = 8.00 - 2.55      # 루프 $8 중 R1 이 $2.55 를 썼다
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def ms(B, Lctx):
    return 1.15 * (15.0 + 0.375 * B + 7.21e-4 * B * Lctx)


def slots(model, vram, cap, prompt=PROMPT):
    free = (vram - WEIGHTS_GB[model]) * 0.85
    kv = max(1, int(free * 1024 * 1024 / ((prompt + cap) * KV_KB[model])))
    return max(1, min(kv, int(MAX_PREFILL / prompt)))


def seconds(model, n_seq, cap, vram, ckpts=(None,)):
    s = slots(model, vram, cap)
    tot = 0.0
    for g in range(math.ceil(n_seq / s)):
        B = min(s, n_seq - g * s)
        tot += ms(B, PROMPT + cap / 2) * cap / 1000.0
    for ck in ckpts:
        c = cap if ck is None else ck
        sc = max(1, min(s, int(MAX_PREFILL / (PROMPT + c + 8))))
        for g in range(math.ceil(n_seq / sc)):
            B = min(sc, n_seq - g * sc)
            tot += ms(B, PROMPT + c) * ANS_STEPS / 1000.0
    return tot


def main():
    out("#LOOP R2 예산 표 — 3600s 안 + 라운드 $3 안에서 가장 긴 예산 고르기")
    out(f"   문제당 시퀀스 = (원본 1 + 섭동 K) x 반복 R · 프리필 상한 {MAX_PREFILL} 토큰")
    out(f"   라운드 상한 ${ROUND_CAP:.2f} · 루프 남은 돈 ${LOOP_LEFT:.2f} · 최악 가격 ${PRICE_WORST}/h 기준")
    out()

    grid = [(K, R, cap) for cap in (1024, 2048) for K in (2, 3) for R in (3, 4)]
    out("## 제출 시간 (워커 96 GB · 안전 예산 3300s)")
    out(f"   {'설정':<22} {'시퀀스/모델':>11} " +
        " ".join(f"{m:>8}" for m in KV_KB) + f"{'합계':>9} {'3300s%':>8} {'판정':>6}")
    rows = []
    for K, R, cap in grid:
        for nprob, nmodel, tag in ((40, 4, "Small 4x40"),):
            nseq = nprob * (1 + K) * R
            per, tot = [], 0.0
            for m in KV_KB:
                v = seconds(m, nseq, cap, 96, ckpts=(None,))
                per.append(v)
                tot += v
            ok = tot <= 3300
            rows.append((K, R, cap, nseq, tot, ok))
            out(f"   K={K} R={R} cap={cap:<5} ({tag})" + f"{nseq:>5}      " +
                " ".join(f"{v:8.0f}" for v in per) +
                f"{tot:9.0f} {tot / 3300 * 100:7.1f}% {'OK' if ok else '초과':>6}")
    out()
    out("   (Main 5모델은 gpt-oss 가 길이 폴백이라 생성 모델 수가 같다 → 위 합계와 같다)")
    out()

    out("## 검증 비용 (H100 80 GB · 공식 33문제 x 4모델 · 체크포인트 2개)")
    out(f"   {'설정':<22} {'시퀀스/모델':>11} {'초/모델':>9} {'4모델':>8} {'시간':>7} "
        f"{'예상$':>7} {'수명':>6} {'최악$':>7} {'판정':>14}")
    best = None
    for K, R, cap in grid:
        nseq = 33 * (1 + K) * R
        per = max(seconds(m, nseq, cap, 80, ckpts=(cap // 2, None)) for m in KV_KB)
        tot = sum(seconds(m, nseq, cap, 80, ckpts=(cap // 2, None)) for m in KV_KB) + 500
        hrs = tot / 3600
        # **수명 상한은 돈이 정한다**: 라운드 상한 / 가장 비싼 후보 가격. 그 안에 예상 시간이
        # 들어가야 채택 가능이다(R1 에서 가격을 생성 후에 알아 $0.85 를 버린 교훈).
        life = min(ROUND_CAP, LOOP_LEFT) / PRICE_WORST
        worst = life * PRICE_WORST
        fits_sub = next(t for kk, rr, cc, _, t, _ in rows if (kk, rr, cc) == (K, R, cap)) <= 3300
        okc = hrs + 0.05 <= life          # +0.05h = 부팅·pip 여유
        verd = "채택 가능" if (fits_sub and okc) else ("제출시간 초과" if not fits_sub else "수명 부족")
        out(f"   K={K} R={R} cap={cap:<5}      {nseq:>5} {per:>9.0f} {tot:>8.0f} "
            f"{hrs:>6.2f}h {hrs * 2.89:>7.2f} {life:>5.2f}h {worst:>7.2f} {verd:>14}")
        if fits_sub and okc:
            score = (cap, K, R)                     # 가장 긴 예산 = cap 우선, 그다음 K, 그다음 R
            if best is None or score > best[0]:
                best = (score, K, R, cap, hrs, life, worst)
    out()
    if best:
        _, K, R, cap, hrs, life, worst = best
        out(f"## 선택 — **cap {cap} · 섭동 K={K} · 반복 R={R}**")
        out(f"   근거는 지휘가 정한 규칙 하나다: **3600s 안 + 라운드 $3 안에서 가장 긴 예산**.")
        out(f"   제출 시간 {next(t for kk,rr,cc,_,t,_ in rows if (kk,rr,cc)==(K,R,cap)):.0f}s "
            f"= 3300s 의 {next(t for kk,rr,cc,_,t,_ in rows if (kk,rr,cc)==(K,R,cap))/3300*100:.0f}%")
        out(f"   검증 {hrs:.2f}h ≈ ${hrs * 2.89:.2f} · 수명 상한 {life:.2f}h → 최악 ${worst:.2f}")
    else:
        out("## 선택 — **없다.** 어떤 조합도 제출 시간 또는 비용을 못 산다")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
