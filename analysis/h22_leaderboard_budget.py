"""H22 가 리더보드·최종셋에서 **실제로** 생성까지 도는가. GPU 0. (지휘 #LG-RP 추가확인)

예산은 확정됐다 — 세 출처가 일치한다(추측 아님, 전부 원문):
  1. 공식 사이트 FAQ 8: "Each submission has 3600 seconds for the entire prediction run,
     covering all models and all problems together"
  2. getting-started README 'Evaluation environment':
     "**Time limit:** 3600 seconds per evaluation run … The limit covers the *entire* prediction
      run -- every model batch and every problem together, **not one hour per problem**."
     "**GPU:** one NVIDIA RTX PRO 6000 (Blackwell …) with 96 GB of VRAM. A single card is exposed
      to the evaluation container, so `torch.cuda.device_count()` returns 1"
  3. Discord 2026-09-09 주최측(JosefKuchar): "eval time limit is 1 hour"
→ 논문의 '문제당 1시간 · 8-GPU 노드' 는 **주최측이 라벨을 만들 때 쓴 예산**이고 제출 예산이 아니다.
  (라벨 생성은 문제당 수백 샘플이 필요하니 그 규모가 맞다.)

그래서 두 경로를 따로 센다:
  (A) **HF 패딩 배치** — 제출물이 지금 쓰는 경로. 한 배치가 cap 스텝을 전부 돈다.
      시간 모델은 워커 4측정점 적합 ms(B,L) = 1.15 × (15 + 0.375·B + 7.21e-4·B·L), 오차 ±0.6 ms.
  (B) **vLLM 동봉** — 연속 배칭이라 평균 토큰에서 끝난다. ZIP 크기 제한은 없고 계정 합산 15 GB 라
      이론상 넣을 수 있다. 처리율은 **RunPod H100 PCIe 실측 1,900 tok/s**(①, 526시퀀스 cap 24576)를 쓴다.
      **워커(RTX PRO 6000)에서 재지 않았으므로 이건 추정이다** — 같은 급이라고 가정한 값이다.

평균 생성 토큰은 실측을 쓴다: H14 원본 평균 11,483(cap 16384) · ① 은 cap 24576 에서 526/526 완주.
보수적으로 10,000 토큰/시퀀스로 잡는다.

사용:  .venv/Scripts/python.exe analysis/h22_leaderboard_budget.py
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "h22_leaderboard_budget.txt"

A_, B_, C_, MULT, PROMPT = 15.0, 0.375, 7.21e-4, 1.15, 400.0
VLLM_TPS = 1900.0          # ① 실측 (H100 PCIe, 526시퀀스, cap 24576)
AVG_TOK = 10000.0          # 실측 기반 보수값
BUDGET = 3600.0
RESERVE = 300.0            # 로드·채점 여유
KV = {"Skywork": 56, "DeepSeek": 144, "Olmo": 512, "Qwen3.5": 128}
WGT = {"Skywork": 15.2, "DeepSeek": 16.4, "Olmo": 14.6, "Qwen3.5": 9.3}

SHAPES = [("리더보드 Small 실측형 (호출 5,3,3,3)", [5, 3, 3, 3]),
          ("최종셋형 4모델 × 35문제", [35] * 4)]
CFG = [("완전형 N4·K6·S2", 4, 6, 2), ("N2·K6·S2", 2, 6, 2), ("N2·K6·S1", 2, 6, 1),
       ("N2·K4·S1", 2, 4, 1), ("N2·K2·S1", 2, 2, 1)]
CAPS = [2048, 4096, 8192, 12288, 16384, 24576]


def seconds_hf(batch, cap):
    lin = (A_ + B_ * batch + C_ * batch * PROMPT) * cap / 1000.0
    quad = C_ * batch * cap * cap / 2000.0
    return (lin + quad) * MULT


def slots(m, safe=0.55):
    return int((96.0 - WGT[m]) * 1024 * 1024 * safe / KV[m])


def hf_time(n_problems, per, cap, model):
    seqs = n_problems * per
    bmax = max(1, min(seqs, slots(model) // cap))
    g = max(1, math.ceil(seqs / bmax))
    return g * seconds_hf(math.ceil(seqs / g), cap)


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("H22 가 리더보드·최종셋에서 실제로 생성까지 도는가 (예산 = 합산 3600s · 1 GPU)")
    out()
    out(f"HF 경로: 워커 4측정점 적합 · vLLM 경로: ① 실측 {VLLM_TPS:.0f} tok/s · 시퀀스당 평균 {AVG_TOK:.0f} 토큰")
    out()
    for tag, calls in SHAPES:
        per_call = (BUDGET - RESERVE) / len(calls)
        out(f"## {tag} — 호출 {len(calls)}회 · 호출당 {per_call:.0f}s")
        out()
        out("| 설계 | 시퀀스/문제 | (A) HF 최대 cap | (B) vLLM 최대 cap | vLLM 소요 |")
        out("|---|---:|---:|---:|---:|")
        for name, N, K, S in CFG:
            per = N + K * S
            hf_ok = [c for c in CAPS
                     if max(hf_time(calls[0], per, c, m) for m in KV) <= per_call]
            # vLLM 은 cap 이 아니라 **실제 생성 토큰**이 비용이다 → cap 을 올려도 평균 토큰이 상한이다.
            seqs_total = sum(n * per for n in calls)
            v_s = seqs_total * AVG_TOK / VLLM_TPS
            v_ok = (v_s <= BUDGET - RESERVE)
            out(f"| {name} | {per} | "
                f"{('**' + str(max(hf_ok)) + '**') if hf_ok else '**불가**'} | "
                f"{'**24576**' if v_ok else '**불가**'} | {v_s:,.0f}s |")
        out()
    out("## 판정")
    out()
    out("* **(A) 지금 제출물(HF 경로)은 생성을 못 산다 — 두 이유가 같은 곳에 도착한다.**")
    out("  (i) 사다리 바닥이 cap 8192 인데 호출당 예산으로는 리더보드에서 **cap 2048~4096** 까지만 산다")
    out("      → `plan()` 이 None 을 돌려주고 **텍스트 폴백(n_sym 순위)만 돌아간다**.")
    out("  (ii) 바닥을 2048 로 낮춰도 소용없다. 올림피아드 풀이는 실측 평균 **11,483 토큰**(H14)이라")
    out("      2048 에서 끝나는 샘플이 사실상 없고, 절단 샘플은 **판정에서 제외**되므로(지휘 #H22-c)")
    out("      전부 중립 0.5 가 되어 **다시 텍스트 폴백**으로 떨어진다. 낮은 cap 은 신호가 아니라 침묵이다.")
    out("  즉 지금 올리면 h15A(리더보드 0.50)와 사실상 같은 제출물이다.")
    out("  HF 가 비싼 이유는 패딩 배치가 **제일 긴 시퀀스를 기다려 cap 스텝을 전부 돌기** 때문이다 —")
    out("  33문제 중 하나가 24576 까지 가면 나머지 525개도 그 시간을 함께 쓴다.")
    out()
    out("* **(B) vLLM 을 ZIP 에 동봉하면 리더보드는 들어가고 최종셋은 K 를 2 까지 깎아야 들어간다.**")
    out("  연속 배칭은 평균 토큰에서 끝나므로 비용이 cap 이 아니라 실제 생성량으로 정해진다.")
    out("  다만 K=2 는 '섭동 과반' 을 셀 수 없다(2개 중 과반 = 1개) → **정의 재현이 깨진다**.")
    out("  그리고 이 추정은 워커가 아니라 **H100 실측**이다. 워커(RTX PRO 6000)에서 재본 적이 없고,")
    out("  인터넷이 차단된 컨테이너에 vLLM 을 동봉해 import 까지 되는지도 확인한 적이 없다.")
    out()
    out("* **결론: H22 는 제출물이 아니라 연구 결과다.** 정의 재현이 되는지는 RunPod 에서 오프라인으로")
    out("  답할 수 있고(①), 그 답이 '된다' 여도 **제출 예산 안에서는 못 쓴다**. 이건 이 프로젝트의")
    out("  중심 문장을 한 번 더 확인하는 것이다 — **추론 예산이 방법을 고른다.**")
    out("  리포트에는 '주최측 라벨 정의는 모델 행동에서 재현되지만, 주최측이 정한 추론 예산 안에서는")
    out("  재현할 수 없다' 는 형태로 쓴다. 이게 negative result 가 아니라 **구조적 관찰**이다.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
