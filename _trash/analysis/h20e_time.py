"""h20e 예상 소요 시간 — 제출물이 실제로 쓰는 선택 로직(plan)을 그대로 돌려 본다. GPU 0. (지휘 #62-4)

h20d 와 달라진 두 가지가 시간에 그대로 들어온다.
  1) **교차모델 캐시가 없다** → 같은 문제를 다른 모델이 다시 물으면 다시 생성한다.
     리더보드형(고유 10문제가 4모델에 걸쳐 14 인스턴스)에서 생성량이 h20d 보다 늘어난다.
  2) **모델을 직접 로드한다** → KV 캐시 크기가 모델마다 다르다. 토큰당 KB 는
     Skywork 56 · Qwen3.5 128 · DeepSeek 144 · **Olmo-3-7B-Think 512**(kv헤드 32 = MHA).
     96 GB 워커에서 Olmo 는 한 배치에 다 못 넣어 **그룹으로 쪼개진다** → 고정비가 그룹마다 더 붙는다.

시간 모델은 워커 실측 4점으로 적합한 ms(B,L) = 1.15 x (15 + 0.375B + 7.21e-4 B L)
(`analysis/ms_per_step_model.py`). 로드는 워커 웜캐시 실측 2~8초를 호출마다 8초로 보수 가정.

사용:  .venv/Scripts/python.exe analysis/h20e_time.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solutions" / "h20e"))
import h20e_lib as L                                    # noqa: E402

LOAD_S = 8.0
SOFT = 3300.0
GPU_GB = 96.0
# (이름, 가중치 GB(bf16), KV KB/토큰) — 로컬 config 에서 계산한 값 (analysis 로그 참조)
MODELS = [("Qwen3.5-4B", 9.3, 128.0), ("Skywork-7B", 15.2, 56.0),
          ("Olmo-3-7B-Think", 14.6, 512.0), ("DeepSeek-8B", 16.4, 144.0)]


def slots_for(weights_gb, kv_kb):
    """로드 후 여유 VRAM 으로 쓸 수 있는 '배치 x cap' 토큰 칸 수 (제출물의 token_slots 와 같은 식)."""
    free_kb = max(0.0, GPU_GB - weights_gb) * 1024 * 1024
    return max(1, int(free_kb * L.KV_SAFE / kv_kb))


def scenario(label, seq, effort="default"):
    """seq = [(MODELS 인덱스, 문제 수)] — 호출마다 모델이 다르므로 캐시 재사용은 없다."""
    spent, rows = 0.0, []
    for i, (mi, n) in enumerate(seq):
        name, w, kv = MODELS[mi]
        budget = (SOFT - spent) / (len(seq) - i)
        budget = max(0.0, budget - LOAD_S)
        slots = slots_for(w, kv)
        got = L.plan(n, budget, 1.0, effort, slots,
                     L.K_RICH if (n <= L.N_FEW and budget > 900) else L.K_DEFAULT)
        if got is None:
            rows.append((name, n, 0, None, 0, 0.0))
            spent += LOAD_S
            continue
        cap, k, g = got
        groups = [min(g, n - j * g) for j in range(-(-n // g))]
        sec = sum(L.seconds_for(x * k, cap, 1.0) for x in groups)
        spent += sec + LOAD_S
        rows.append((name, n, k, cap, len(groups), sec))
    print(f"  {label} · effort {effort}")
    print(f"    {'모델':16s} {'문제':>4s} {'k':>2s} {'cap':>6s} {'그룹':>4s} {'예상s':>7s} {'train AUC':>10s}")
    for r in rows:
        auc = L.grid_auc(r[2], r[3]) if r[3] else 0.0
        print(f"    {r[0]:16s} {r[1]:4d} {r[2]:2d} {str(r[3]):>6s} {r[4]:4d} {r[5]:7.0f} {auc:10.3f}"
              + ("   ← 전부 절단(임계=cap), 정보 없음" if r[3] and auc < 0.65 else ""))
    print(f"    합계 {spent:.0f}s = {spent / 60:.1f}분 (로드 {LOAD_S:.0f}s x {len(seq)} 포함) · 여유 {3600 - spent:.0f}s\n")
    return spent


if __name__ == "__main__":
    print(f"h20e 예상 시간 (SOFT {SOFT:.0f}s · 로드 {LOAD_S:.0f}s/호출 · k 기본 {L.K_DEFAULT} · "
          f"GPU {GPU_GB:.0f} GB · KV 여유 몫 {L.KV_SAFE})\n")
    print("  모델별 '배치 x cap' 칸 수와 cap 4096 에서 한 배치에 들어가는 시퀀스 수")
    for name, w, kv in MODELS:
        s = slots_for(w, kv)
        print(f"    {name:16s} KV {kv:5.0f} KB/토큰 · 칸 {s:9,d} · cap4096 시퀀스 {s // 4096:4d} · cap8192 {s // 8192:4d}")
    print()
    scenario("A 리더보드형 4호출 (Small 14 인스턴스, 고유 10문제)", [(0, 5), (1, 3), (2, 3), (3, 3)])
    scenario("B 4호출 x 10문제", [(0, 10), (1, 10), (2, 10), (3, 10)])
    scenario("C 최종셋형 4호출 x 35문제", [(0, 35), (1, 35), (2, 35), (3, 35)])
    scenario("C 최종셋형 (effort low)", [(0, 35), (1, 35), (2, 35), (3, 35)], effort="low")
    scenario("D 최종셋형 4호출 x 40문제", [(0, 40), (1, 40), (2, 40), (3, 40)])

# 메모: 제출물의 호출예산은 H20_CALLS(기본 4)로 나눈다. Small 트랙은 모델 4종 x effort default = 호출 4회라 맞는다.
# Main 은 gpt-oss-120b 가 effort 3종으로 와서 호출이 더 늘 수 있다 — 그때는 4회까지 예산을 쓰고 나머지는
# '남은 전부'가 되어 cap 이 작아진다. Small 제출본이므로 이 표는 4호출 가정이다.
