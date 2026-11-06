"""#TT-PERTURB 시간·VRAM 예산 — 제출 환경(RTX PRO 6000 96 GB · 3600s 합산)과 검증 파드(H100 80 GB).

왜 이 계산이 설계를 정하는가: 테스트타임 섭동은 (1+K) 배의 생성을 요구한다. 3600초는 **배치 전체
합산**(공식 FAQ 8)이므로 'CoT 몇 토큰까지 살 수 있나' 가 곧 방법의 생사다. h20 계열이 cap 8192 를
못 사서 죽었고(CLAUDE.md), H22 는 cap 24576 이 필요해 리더보드에서는 텍스트 폴백만 돌았다.

시간 모델: ms(B, L) = 1.15 x (15 + 0.375 B + 7.21e-4 B L)  — 워커 4점 실측 적합, 오차 +-0.6 ms
           (`analysis/ms_per_step_model.py`). L 은 문맥 길이(프롬프트+생성 중), B 는 동시 시퀀스.
VRAM: 토큰당 KV = Skywork 56 KB · Qwen3.5 128 · DeepSeek 144 · **Olmo 512**(MHA) → Olmo 가 병목.

사용: .venv/Scripts/python.exe analysis/ttp_budget.py
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "experiments/results/ttp_budget.txt"

KV_KB = {"Skywork": 56, "Qwen3.5": 128, "DeepSeek": 144, "Olmo": 512}
WEIGHTS_GB = {"Skywork": 15.2, "Qwen3.5": 9.3, "DeepSeek": 16.4, "Olmo": 14.6}
PROMPT = 600          # 채팅 템플릿 + 지시문 + 올림피아드 문제. train-main-v2 단어 중앙 59 기준 보수적
ANS_STEPS = 8         # `</think>\nANSWER:` 강제 즉답
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def ms_per_step(B, Lctx):
    return 1.15 * (15.0 + 0.375 * B + 7.21e-4 * B * Lctx)


def slots(model, vram_gb, cap):
    """그 VRAM 에서 동시에 들 수 있는 시퀀스 수 (KV 만 고려, 활성값 여유 85%)."""
    free = (vram_gb - WEIGHTS_GB[model]) * 0.85
    per_seq_gb = (PROMPT + cap) * KV_KB[model] / 1024 / 1024
    return max(1, int(free / per_seq_gb))


def gen_seconds(model, n_seq, cap, vram_gb):
    """패딩 배치 HF generate: 그룹마다 cap 스텝을 전부 돈다(제일 긴 시퀀스를 기다린다)."""
    s = slots(model, vram_gb, cap)
    groups = math.ceil(n_seq / s)
    tot = 0.0
    for g in range(groups):
        B = min(s, n_seq - g * s)
        tot += ms_per_step(B, PROMPT + cap / 2) * cap / 1000.0
    return tot, groups, s


def ans_seconds(model, n_seq, prefix_len, vram_gb):
    """강제 즉답 8스텝. 문맥이 프롬프트+prefix 라 KV 는 그만큼."""
    s = slots(model, vram_gb, prefix_len)
    groups = math.ceil(n_seq / s)
    tot = 0.0
    for g in range(groups):
        B = min(s, n_seq - g * s)
        tot += ms_per_step(B, PROMPT + prefix_len) * ANS_STEPS / 1000.0
    return tot


def main():
    out("#TT-PERTURB 예산 — 테스트타임 섭동이 3600초 안에 들어가나")
    out(f"   시간모델 ms(B,L) = 1.15 x (15 + 0.375B + 7.21e-4 B L) · 프롬프트 {PROMPT} 토큰 가정")
    out(f"   토큰당 KV: " + " · ".join(f"{k} {v} KB" for k, v in KV_KB.items()))
    out()

    # ---------------------------------------------------------------- (1) 제출 환경
    out("## (1) 제출 환경 — 워커 RTX PRO 6000 96 GB · 합산 3600s (안전 3300s)")
    out("   K = 섭동 수 · 문제당 시퀀스 = 1(원본 greedy) + K.  모델 수로 예산을 나눈다.")
    out()
    out("   호출당 예산 = 3300 / 모델수.  '전부' = 전 모델 합계 초과 여부")
    hdr = f"   {'설정':<28} {'Skywork':>9} {'Qwen3.5':>9} {'DeepSeek':>9} {'Olmo':>9} {'합계':>9} {'예산':>8}"
    for nmodel, nprob, label in [(4, 40, "Small 4모델x40문제"), (4, 35, "Small 4모델x35문제"),
                                 (5, 40, "Main 5모델x40문제(gpt-oss 폴백)")]:
        budget = 3300.0
        out(f"   [{label}]  (gpt-oss 는 길이 폴백 → 생성 0)")
        out(hdr)
        for K, cap in [(8, 128), (8, 256), (8, 512), (8, 1024), (6, 512), (6, 1024), (4, 2048), (2, 4096)]:
            nseq = nprob * (1 + K)
            row, tot = [], 0.0
            for m in ("Skywork", "Qwen3.5", "DeepSeek", "Olmo"):
                g, _, _ = gen_seconds(m, nseq, cap, 96)
                a = ans_seconds(m, nseq, cap, 96)
                row.append(g + a)
                tot += g + a
            ok = "OK" if tot <= budget else "초과"
            out(f"   K={K} cap={cap:<5} {'':<14}" + "".join(f"{v:9.0f}" for v in row) +
                f"{tot:9.0f} {ok:>8}")
        out()

    # ---------------------------------------------------------------- (2) VRAM 칸
    out("## (2) VRAM 칸 — Olmo(토큰당 512 KB) 가 병목이라 그룹이 쪼개진다")
    out(f"   {'cap':>6} " + " ".join(f"{m:>10}" for m in KV_KB))
    for cap in (128, 256, 512, 1024, 2048):
        out(f"   {cap:>6} " + " ".join(f"{slots(m, 96, cap):>10}" for m in KV_KB))
    out("   (숫자 = 96 GB 워커에서 한 배치에 드는 시퀀스 수. 40문제 x K=8 = 360 시퀀스면 그룹 수 = 360/칸)")
    out()

    # ---------------------------------------------------------------- (3) 검증 파드
    out("## (3) 검증 파드 — H100 80 GB · 공식 33문제 · 라운드 2~8 을 $0 로 재활용할 저장 설계")
    out("   문제당 시퀀스 = 3(원본: greedy 1 + T=1.0 샘플 2) + K.  CoT 는 한 번만 생성하고")
    out("   체크포인트 여러 개에서 강제 즉답을 받아 **CoT 예산 격자를 오프라인에 만든다**.")
    out()
    PRICE = 2.89 / 3600  # H100 PCIe $/s
    for K, cap, ckpts in [(8, 512, (128, 256, 512)), (8, 1024, (256, 512, 1024)),
                          (6, 512, (128, 256, 512)), (8, 256, (64, 128, 256))]:
        nseq = 33 * (3 + K)
        tot = 0.0
        parts = []
        for m in ("Skywork", "Qwen3.5", "DeepSeek", "Olmo"):
            g, grp, sl = gen_seconds(m, nseq, cap, 80)
            a = sum(ans_seconds(m, nseq, c, 80) for c in ckpts)
            parts.append((m, g, a, grp, sl))
            tot += g + a
        load = 4 * 60  # 모델 로드 4회
        tot += load
        out(f"   K={K} cap={cap} 체크포인트{ckpts} · 시퀀스/모델 {nseq}")
        for m, g, a, grp, sl in parts:
            out(f"      {m:<9} 생성 {g:7.0f}s (그룹 {grp}, 칸 {sl}) + 즉답 {a:6.0f}s")
        out(f"      합계 {tot:7.0f}s = {tot / 3600:.2f}h  ≈ ${tot * PRICE:.2f}   "
            f"(수명상한 {tot / 3600 * 1.5:.1f}h → 최악 ${tot / 3600 * 1.5 * 2.89:.2f})")
        out()

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"저장 {REPORT}")


if __name__ == "__main__":
    main()
