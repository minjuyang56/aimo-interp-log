"""H22 라벨 정의 재현 — 시간 예산표. GPU 0. (지휘 #H22-C)

왜 이것부터 쓰는가
  H22 는 (model, problem) 마다 원본 N 샘플 + 섭동 K개 × S 샘플을 **끝까지 생성**해야 한다.
  h20 계열을 죽인 것은 신호가 아니라 **예산**이었다(cap 8192 이상에서만 신호가 살지만 예산은 cap 6144 까지만 샀다).
  그래서 설계를 확정하기 전에 어떤 (N, K, S, cap) 이 들어가는지를 먼저 센다.

예산 해석이 두 개다 — 양쪽 다 낸다
  (i) **공식 FAQ 8**: "3600 seconds for the entire prediction run, covering all models and all problems
      together" → 배치 전체 합산 3600초. CLAUDE.md 가 지휘 #44 에서 이 문장으로 확정해 기록해 둔 값이다.
  (ii) **논문(arXiv 2607.13899) 기술로 전달된 값**: 문제당 1시간 · 8-GPU 노드.
      이 값은 주최측이 **라벨을 만들 때** 쓴 예산일 수도 있다(라벨 생성은 문제당 수백 샘플이 필요하다).
      판단을 유보하고 (ii) 도 같이 계산한다: 문제당 3600s × 8 GPU = **28,800 GPU-초**.
  우리 워커 실측은 **1 GPU**(RTX PRO 6000 96 GB)이므로 (ii) 는 '1-GPU 환산 28,800초' 로 읽는다.

시간 모델 (전부 실측 적합, 새로 추정한 값 없음)
  `analysis/ms_per_step_model.py` 의 ms(B, L) = 1.15 × (15 + 0.375·B + 7.21e-4·B·L) — 워커 4측정점 오차 ±0.6 ms.
  HF 패딩 배치는 **한 배치가 cap 스텝을 전부 돈다**(제일 긴 시퀀스가 끝날 때까지) → 벽시계는 평균 토큰이 아니라
  **cap 이 정한다**. 이게 vLLM(연속 배칭, 평균 토큰으로 끝남)과 갈리는 지점이라 둘을 따로 낸다.

VRAM 제약 (h20e 에서 Olmo 를 터뜨린 그 제약)
  토큰당 KV: Skywork 56 KB · Qwen3.5 128 · DeepSeek 144 · **Olmo-3-7B-Think 512**(MHA).
  96 GB 에서 가중치를 빼고 남는 만큼만 '배치 × cap' 칸을 살 수 있다. 모자라면 그룹을 쪼개고,
  쪼개면 고정비가 그룹 수만큼 더 붙는다.

사용:  .venv/Scripts/python.exe analysis/h22_budget.py
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "h22_budget.txt"

# --- 워커 실측 적합 (analysis/ms_per_step_model.py 와 같은 상수) ---
A, B_COEF, C_COEF, MARGIN, PROMPT = 15.0, 0.375, 7.21e-4, 1.15, 400.0

# --- 모델별 KV·가중치 (CLAUDE.md 환경 제약 절, 실측) ---
MODELS = {
    "Skywork/Skywork-OR1-Math-7B":          dict(kv=56,  w=15.2),
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(kv=144, w=16.4),
    "allenai/Olmo-3-7B-Think":              dict(kv=512, w=14.6),
    "Qwen/Qwen3.5-4B":                      dict(kv=128, w=9.3),
}
VRAM_GB = 96.0
VRAM_HEADROOM_GB = 8.0          # 활성값·단편화 여유 (h20e 에서 쓴 값)

BUDGET_FAQ8 = 3600.0            # (i) 배치 전체 합산
BUDGET_PAPER_GPUS = 8           # (ii) 문제당 1시간 × 8 GPU
BUDGET_PAPER_PER_PROBLEM = 3600.0 * BUDGET_PAPER_GPUS

N_MODELS, N_PROBLEMS = 4, 35    # 최종셋형 Small (문제 30~40, 모델 4종)


def seconds_for(batch, cap, prompt=PROMPT):
    """cap 스텝을 한 배치로 생성할 때의 예상 초. 문맥이 자라는 것을 적분한다."""
    lin = (A + B_COEF * batch + C_COEF * batch * prompt) * cap / 1000.0
    quad = C_COEF * batch * cap * cap / 2000.0
    return (lin + quad) * MARGIN


def max_batch(model, cap):
    """VRAM 이 허용하는 최대 동시 시퀀스 수."""
    free_kb = (VRAM_GB - MODELS[model]["w"] - VRAM_HEADROOM_GB) * 1024 * 1024
    return max(1, int(free_kb / (MODELS[model]["kv"] * cap)))


def instance_seconds(model, cap, seqs):
    """한 (model, problem) 의 seqs 개 시퀀스를 생성하는 데 걸리는 초.
    VRAM 이 모자라면 그룹으로 쪼갠다 — 쪼개면 고정비가 그룹 수만큼 더 붙는다."""
    bmax = max_batch(model, cap)
    g = math.ceil(seqs / bmax)
    return g * seconds_for(math.ceil(seqs / g), cap), g, bmax


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("H22 시간 예산표 — 라벨 정의 충실 재현이 제출 예산 안에 드는가")
    out()
    out(f"전제: 최종셋형 {N_MODELS}모델 × {N_PROBLEMS}문제 = {N_MODELS * N_PROBLEMS} 인스턴스 · 프롬프트 {PROMPT:.0f}토큰")
    out(f"예산 (i) 공식 FAQ 8 = 배치 전체 합산 **{BUDGET_FAQ8:.0f}s**  →  인스턴스당 "
        f"{BUDGET_FAQ8 / (N_MODELS * N_PROBLEMS):.1f}s")
    out(f"예산 (ii) 논문 기술 = 문제당 3600s × {BUDGET_PAPER_GPUS} GPU = **{BUDGET_PAPER_PER_PROBLEM:.0f} GPU-초**/문제 "
        f"→ 1-GPU 환산 인스턴스당 {BUDGET_PAPER_PER_PROBLEM / N_MODELS:.0f}s (4모델이 그 문제를 나눠 쓴다고 볼 때)")
    out()

    out("## VRAM 이 허용하는 동시 시퀀스 수 (96 GB · 가중치·여유 차감)")
    out()
    caps = (8192, 16384, 24576, 32768)
    out("| 모델 | KV KB/tok | " + " | ".join(f"cap {c}" for c in caps) + " |")
    out("|---|---:|" + "---:|" * len(caps))
    for m in MODELS:
        out(f"| {m.split('/')[-1]} | {MODELS[m]['kv']} | "
            + " | ".join(f"{max_batch(m, c)}" for c in caps) + " |")
    out()
    out("→ Olmo 는 cap 16384 에서 동시 **9 시퀀스**뿐이다. H22 의 기본 설계(원본 8 + 섭동 8×4 = 40 시퀀스)는")
    out("   한 문제조차 한 배치에 못 넣고 5그룹으로 쪼개진다 = 시간이 5배로 붙는다.")
    out()

    out("## 설계 후보별 (model, problem) 1건 소요 — HF 패딩 배치 (제출물이 실제로 쓰는 경로)")
    out()
    out("패딩 배치는 cap 스텝을 전부 돈다 → 아래 숫자는 평균 토큰이 아니라 cap 이 정한다.")
    out()
    CONFIGS = [
        ("H22 완전형  N8·K8·S4", 8, 8, 4),
        ("H22 cap우선 N6·K6·S4", 6, 6, 4),      # 6유형 각 1개 · S4 유지 · cap 을 더 산다
        ("H22 cap우선 N4·K6·S4", 4, 6, 4),
        ("H22 cap우선 N4·K6·S3", 4, 6, 3),
        ("H22 축소 1   N8·K8·S2", 8, 8, 2),
        ("H22 축소 2   N4·K6·S2", 4, 6, 2),
        ("H22 축소 3   N4·K4·S2", 4, 4, 2),
        ("H22 축소 4   N2·K4·S1", 2, 4, 1),
        ("H22 최소형   N2·K2·S1", 2, 2, 1),
    ]
    out("| 설계 | 시퀀스/문제 | " + " | ".join(f"cap {c}" for c in caps) + " |")
    out("|---|---:|" + "---:|" * len(caps))
    for tag, N, K, S in CONFIGS:
        seqs = N + K * S
        cells = []
        for c in caps:
            worst = max(instance_seconds(m, c, seqs)[0] for m in MODELS)
            cells.append(f"{worst:.0f}s")
        out(f"| {tag} | {seqs} | " + " | ".join(cells) + " |")
    out()
    out("(가장 느린 모델 = Olmo 기준. 모델 평균이 아니라 **최악**을 쓴다 — 한 모델이 늦으면 호출이 늦는다.)")
    out()

    out("## 예산 (i) 배치 합산 3600s 에서 들어가는 조합")
    out()
    out("| 설계 | cap | 전 인스턴스 합계 | 3600s 중 | 판정 |")
    out("|---|---:|---:|---:|---|")
    any_fit_i = False
    for tag, N, K, S in CONFIGS:
        seqs = N + K * S
        for c in caps:
            tot = sum(instance_seconds(m, c, seqs)[0] for m in MODELS) * N_PROBLEMS
            ok = tot <= BUDGET_FAQ8
            if ok:
                any_fit_i = True
            out(f"| {tag} | {c} | {tot:,.0f}s | {tot / BUDGET_FAQ8:.0%} | {'**들어감**' if ok else '초과'} |")
    out()
    if not any_fit_i:
        out("→ **예산 (i) 에서는 어떤 조합도 들어가지 않는다.** 최소형(N2·K2·S1 = 4시퀀스)·cap 4096 조차")
        out("   초과다. 이건 h20 계열을 죽인 벽과 같은 벽이고, 이번엔 더 두껍다(H22 는 섭동을 **끝까지** 풀어야 한다).")
    out()

    out("## 예산 (ii) 문제당 3600s × 8 GPU 에서 들어가는 조합")
    out()
    out(f"인스턴스당 허용 = {BUDGET_PAPER_PER_PROBLEM / N_MODELS:,.0f}s (1-GPU 환산)")
    out()
    out("| 설계 | cap | 최악 인스턴스 | 허용 대비 | 판정 |")
    out("|---|---:|---:|---:|---|")
    allow = BUDGET_PAPER_PER_PROBLEM / N_MODELS
    for tag, N, K, S in CONFIGS:
        seqs = N + K * S
        for c in caps:
            worst = max(instance_seconds(m, c, seqs)[0] for m in MODELS)
            ok = worst <= allow
            out(f"| {tag} | {c} | {worst:,.0f}s | {worst / allow:.0%} | {'**들어감**' if ok else '초과'} |")
    out()

    out("## 예산 (ii) 에서 **살 수 있는 최대 cap** (지휘 #H22-c)")
    out()
    out("왜 cap 을 최대한 길게 가져가야 하나: Skywork 자체 라벨에서 spurious 를 결정한 변형 실패의")
    out("**69% 가 오답이 아니라 cap 16384 절단**이었다(`unfinished_cause.txt`). 절단은 신호가 아니라 예산이다.")
    out()
    out("| 설계 | 최대 cap | 그때 걸리는 시간 | 허용 대비 |")
    out("|---|---:|---:|---:|")
    FINE = (8192, 12288, 16384, 24576, 32768, 49152, 65536)
    best_full = None
    for tag, N, K, S in CONFIGS:
        seqs = N + K * S
        fit = [c for c in FINE if max(instance_seconds(m, c, seqs)[0] for m in MODELS) <= allow]
        if not fit:
            out(f"| {tag} | — | — | 최소 cap 도 초과 |")
            continue
        c = max(fit)
        worst = max(instance_seconds(m, c, seqs)[0] for m in MODELS)
        out(f"| {tag} | **{c}** | {worst:,.0f}s | {worst / allow:.0%} |")
        if best_full is None:
            best_full = (tag, N, K, S, c, worst)
    out()
    if best_full:
        tag, N, K, S, c, worst = best_full
        out(f"→ **완전형(N{N}·K{K}·S{S})을 유지하면서 살 수 있는 최대 cap = {c}** "
            f"(최악 모델 {worst:,.0f}s = 허용의 {worst / allow:.0%}).")
        out("   요청한 32768 이 들어간다." if c >= 32768
            else f"   요청한 32768 은 안 들어간다 — 들어가는 최대값은 {c} 다.")
        out("   병목은 Olmo 다(토큰당 KV 512 KB → cap 이 길어질수록 동시 시퀀스가 줄어 그룹이 늘어난다):")
        for m in MODELS:
            sec, g, bmax = instance_seconds(m, c, N + K * S)
            out(f"     {m.split('/')[-1]:28s} 배치상한 {bmax:3d} · 그룹 {g} · {sec:,.0f}s")
    out()
    out("## 줄일 순서 (예산이 모자랄 때)")
    out()
    out("시간은 cap 에 **제곱**으로, 시퀀스 수에 거의 **선형**으로 든다(ms = a + b·B + c·B·L 에서 L 이 cap 과 함께 자란다).")
    out("그래서 같은 비율을 깎을 때 **cap 을 내리는 것이 가장 크게 준다** — 그런데 cap 은 신호가 사는 자리라")
    out("가장 늦게 건드려야 한다(격자 실측: cap 4096·6144 는 임계값이 cap 과 같아져 **무정보**, 신호는 8192 이상에서만 산다).")
    out()
    for i, s in enumerate([
        "**S (섭동당 샘플)** 4 → 2 → 1. 다수결의 분해능은 떨어지지만 섭동 **종류**는 유지된다. "
        "논문 정의의 핵은 '과반의 섭동에서 틀린다' 이므로 K 를 지키는 게 S 보다 중요하다.",
        "**N (원본 샘플)** 8 → 4 → 2. P(원본 정답) 은 선별용이고 라벨 대비를 만드는 쪽은 섭동이다.",
        "**K (섭동 수)** 8 → 6 → 4. 유형 다양성을 지키려면 유형당 1개씩 6유형(Rephrase·Rename·Domain·"
        "Distract·Typos·상수)을 남기는 것이 K 를 고르게 줄이는 것보다 낫다.",
        "**판정 문제 수** — 전수 생성을 포기하고 m 개만 생성, 나머지는 텍스트 규칙. "
        "단 h20e 실측에서 혼합은 두 군의 분할이 어긋나 텍스트 단독을 못 넘었다(m≤20 에서 0.485~0.667).",
        "**cap** 16384 → 12288 → 8192. **8192 미만은 내리지 않는다** — 그 아래는 측정이 포화해 무정보다.",
    ], 1):
        out(f"{i}. {s}")
    out()
    out("## 결론")
    out()
    out("예산 (i) 이 맞으면 **H22 는 제출물로 성립하지 않는다** — 리포트의 negative result 로 쓰고,")
    out("라벨 정의 재현은 **오프라인 라벨 생성**(지금 Kaggle 큐가 하는 일)에만 쓴다.")
    out("예산 (ii) 가 맞으면 완전형 N8·K8·S4·cap 16384 까지 들어간다. **어느 쪽인지 확인이 선행 과제다.**")
    out("그래서 제출물은 **예산 적응형**으로 만든다: 남은 시간을 재서 위 사다리를 위에서부터 내려오고,")
    out("최소형조차 못 사면 생성을 포기하고 h21cK 경로(forward 만, 67s)로 떨어진다. 그러면 (i)·(ii) 어느 쪽이든 유효하다.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
