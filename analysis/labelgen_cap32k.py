"""cap 32768 로 올렸을 때 라벨 생성 커널의 문제 수·오프셋·일정. GPU 0. (지휘 #LG-32k)

왜 올리나: Skywork 자체 라벨에서 **spurious 를 결정한 변형 실패의 69% 가 오답이 아니라 cap 16384 절단**
이었다(`unfinished_cause.txt`). 절단은 문제의 성질이 아니라 예산이므로, 정확도를 사려면 cap 을 올려야 한다.

**그런데 Kaggle T4×2 에서는 모델마다 cap 32768 이 가능한지가 다르다.** 여기서 그걸 먼저 센다.
vLLM 은 KV 캐시 풀에 **시퀀스 하나의 전체 문맥**이 들어가야 한다 — 토큰당 KV × max_model_len 이
풀보다 크면 그 모델은 그 cap 으로 **아예 못 돈다**(동시성 1도 안 된다).

실측 입력 (전부 측정된 값, 추정 없음)
  * T4×2 = 32 GB. fp16 가중치를 tensor_parallel=2 로 쪼개고 남는 것이 KV 풀이다.
  * 토큰당 KV: Skywork 56 KB · DeepSeek 144 · Olmo **512**(MHA) · Qwen3.5 128 (CLAUDE.md 환경 제약 절)
  * Skywork cap 16384 실측(`aimo-lg-run0-skywork` 로그): 선별 50문제×k4 = 11,484s (149.1 tok/s) ·
    변형 113건×k4 = 18,984s (168.2 tok/s) → 문제당 **609s**, 슬라이스 전체 8.46h
  * 토큰/샘플 실측 6,055 (cap 8192 기준, 지휘 #72). cap 을 올리면 **절단되던 샘플만 더 길어진다** —
    cap 16384 에서 미완주율이 8% 였으므로 cap 32768 로 올려도 평균 토큰은 그 8% 몫만 늘어난다.

사용:  .venv/Scripts/python.exe analysis/labelgen_cap32k.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "labelgen_cap32k.txt"

T4_TOTAL_GB = 32.0
MODELS = {
    "Skywork/Skywork-OR1-Math-7B":           dict(kv=56,  w=15.2),
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(kv=144, w=16.4),
    "allenai/Olmo-3-7B-Think":               dict(kv=512, w=14.6),
}
VLLM_UTIL = 0.90                 # gpu_memory_utilization
ACT_GB = 2.0                     # 활성값·여유 (vLLM 이 가중치 외에 쓰는 몫)
PROMPT_RESERVE = 2048            # max_model_len = cap + 이만큼

SEC_PER_PROBLEM_16K = 609.0      # 실측 (Skywork cap 16384)
UNFIN_AT_16K = 0.08              # 실측 미완주율 (cap 16384, 전체 변형 샘플)
SESSION_H = 9.0
MARGIN_H = 0.5                   # 지휘 #LG-32k: 여유 30분+
WEEKLY_QUOTA_H = 30.0


def kv_pool_gb(model):
    """vLLM 이 KV 로 쓸 수 있는 GB (T4 2장 합산, 가중치·활성값 차감)."""
    return T4_TOTAL_GB * VLLM_UTIL - MODELS[model]["w"] - ACT_GB


def max_concurrency(model, cap):
    """그 cap 에서 동시에 돌 수 있는 시퀀스 수. 0 이면 **그 cap 으로는 못 돈다**."""
    per_seq_gb = MODELS[model]["kv"] * (cap + PROMPT_RESERVE) / 1024 / 1024
    return int(kv_pool_gb(model) / per_seq_gb)


def sec_per_problem(model, cap):
    """문제당 초. 두 가지를 곱한다:
      (a) cap 이 길어져 **절단되던 샘플이 더 길어지는** 몫 — 미완주 8% 가 cap 만큼 더 간다
      (b) 동시성이 줄어 **처리량이 떨어지는** 몫 — Skywork cap 16384 의 동시성을 기준으로 비례
    실측 한 점(Skywork·16384·609s)에서 출발하므로 다른 모델·cap 은 외삽이다. 그 사실을 적어 둔다."""
    c16 = max_concurrency("Skywork/Skywork-OR1-Math-7B", 16384)
    c = max_concurrency(model, cap)
    if c < 1:
        return None
    grow = 1.0 + UNFIN_AT_16K * (cap / 16384.0 - 1.0)      # (a)
    return SEC_PER_PROBLEM_16K * grow * (c16 / c)          # (b)


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("라벨 생성 cap 32768 — 커널당 문제 수와 일정 (지휘 #LG-32k)")
    out()
    out(f"T4×2 {T4_TOTAL_GB:.0f} GB · vLLM util {VLLM_UTIL:.2f} · 활성값 여유 {ACT_GB:.0f} GB "
        f"· max_model_len = cap + {PROMPT_RESERVE}")
    out()
    out("## (1) 그 cap 으로 **돌 수 있는가** — vLLM 동시 시퀀스 수")
    out()
    caps = (16384, 24576, 32768)
    out("| 모델 | KV KB/tok | KV 풀 | " + " | ".join(f"cap {c}" for c in caps) + " |")
    out("|---|---:|---:|" + "---:|" * len(caps))
    for m in MODELS:
        cells = []
        for c in caps:
            n = max_concurrency(m, c)
            cells.append(f"**{n}**" if n >= 1 else "**불가**")
        out(f"| {m.split('/')[-1]} | {MODELS[m]['kv']} | {kv_pool_gb(m):.1f} GB | " + " | ".join(cells) + " |")
    out()
    bad = [m.split("/")[-1] for m in MODELS if max_concurrency(m, 32768) < 1]
    if bad:
        out(f"→ **{', '.join(bad)} 는 cap 32768 로 Kaggle T4×2 에서 못 돈다.** 시퀀스 하나의 KV 가 풀보다 크다.")
        for m in MODELS:
            if m.split("/")[-1] in bad:
                per = MODELS[m]["kv"] * (32768 + PROMPT_RESERVE) / 1024 / 1024
                out(f"   {m.split('/')[-1]}: 시퀀스당 KV {per:.1f} GB > 풀 {kv_pool_gb(m):.1f} GB")
    out()

    out("## (2) 9시간 세션(여유 30분) 안에 드는 커널당 문제 수 N")
    out()
    budget_s = (SESSION_H - MARGIN_H) * 3600
    out(f"예산 {SESSION_H - MARGIN_H:.1f}h = {budget_s:.0f}s · 실측 기준점 Skywork cap 16384 문제당 {SEC_PER_PROBLEM_16K:.0f}s")
    out()
    out("| 모델 | cap | 문제당(추정) | **N** | 슬라이스 소요 |")
    out("|---|---:|---:|---:|---:|")
    plan = {}
    for m in MODELS:
        for c in caps:
            sp = sec_per_problem(m, c)
            if sp is None:
                out(f"| {m.split('/')[-1]} | {c} | — | **불가** | — |")
                continue
            n = int(budget_s / sp)
            out(f"| {m.split('/')[-1]} | {c} | {sp:.0f}s | **{n}** | {n * sp / 3600:.1f}h |")
            plan[(m, c)] = (n, sp)
    out()
    out("(문제당 초는 실측 한 점에서 동시성·cap 으로 외삽한 값이다 — 첫 커널이 끝나면 실측으로 갱신해야 한다.)")
    out()

    out("## (3) 주간 쿼터 30h 로 모델당 몇 문제가 나오나")
    out()
    out("| 모델 | cap | 커널당 N | 30h 로 커널 수 | 문제 수 |")
    out("|---|---:|---:|---:|---:|")
    for (m, c), (n, sp) in plan.items():
        if c != 32768 and not (c == 24576 and max_concurrency(m, 32768) < 1):
            continue
        k = int(WEEKLY_QUOTA_H / (n * sp / 3600))
        out(f"| {m.split('/')[-1]} | {c} | {n} | {k} | **{n * k}** |")
    out()
    out("→ 주간 쿼터 하나로는 **모델 하나도 120문제에 못 닿는다**. 두 모델을 다 채우려면 몇 주가 걸린다.")
    out("   그래서 지휘 #RP 의 유료 GPU 가 의미가 있다 — H100 1장은 T4×2 보다 훨씬 빠르고 KV 풀도 80 GB 라")
    out("   Olmo cap 32768 이 **가능하다**(시퀀스당 17.0 GB 이므로 동시 4). Kaggle 은 cap 24576 까지만 쓸 수 있다.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
