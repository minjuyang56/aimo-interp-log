"""#SUB-GEN2048 시간 견적 — 3600초 안에 드는가. GPU 0. (지휘 #SUB-GEN2048 (1))

## 쓰는 시간 모형 (추측 아님, 워커 실측 적합)
  ms(B, L) = 1.15 × (15 + 0.375·B + 7.21e-4·B·L)      (`analysis/ms_per_step_model.py`)
네 측정점 오차 ±0.6 ms: 워커 probe2 배치 8/32/64 에서 25.0 / 54.4 / 95.5 ms/step ·
h20 실측 배치 10 · 16k 문맥 79.2 ms/step. **디코딩은 배치에 거의 선형이고 문맥에 크게 의존한다.**

## 비용의 세 조각
1. **모델 로드** — 워커 warm cache 로 4~8B 약 2초(README). 보수적으로 30s/모델.
2. **생성** — 그룹 수 × MAXNEW 스텝 × ms(B, L). L 은 프롬프트 + 생성 절반(문맥이 자란다).
   배치 B 는 **VRAM 으로 제한**된다: 토큰당 KV 가 Skywork 56 KB · DeepSeek 144 · **Olmo 512**(MHA)
   이므로 같은 문제 수에서 Olmo 가 9배 무겁다. 96 GB 에서 가중치 15 GB 를 빼고 0.6 을 곱해 센다.
3. **특징 prefill** — 문제마다 (프롬프트 + 2048) 토큰을 한 번 forward. 생성 1스텝이 토큰 1개를
   내는 데 비해 prefill 은 전체를 한 번에 처리하므로, **보수적으로 생성 20스텝치**로 센다.

## 판정
합계가 3600s 를 넘으면 **창을 줄이지 않는다**(학습 창과 어긋나면 가중치가 무의미해진다) —
제출물이 `plan` 단계에서 생성을 포기하고 **h15C 폴백**으로 떨어진다(지휘 지정).

사용:  .venv/Scripts/python.exe analysis/sub2048_time.py
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAXNEW = 2048
PROMPT_TOK = 260                 # 지시문 58 + 올림피아드 문제 중앙 ~52단어 → 보수적으로 260
LOAD_S = 30.0
VRAM_GB = 96.0
WEIGHT_GB = {"Skywork": 15.2, "Olmo": 14.6, "DeepSeek": 16.4, "Qwen3.5": 9.3}
KV_KB = {"Skywork": 56, "Olmo": 512, "DeepSeek": 144, "Qwen3.5": 128}
HAS_W = ("Skywork", "Olmo", "DeepSeek")          # 가중치 있는 모델(Qwen3.5 는 폴백)


def ms_per_step(B, L):
    return 1.15 * (15 + 0.375 * B + 7.21e-4 * B * L)


def max_batch(model, n):
    free = (VRAM_GB - WEIGHT_GB[model]) * 0.6 * 1e9
    per = KV_KB[model] * 1024 * (PROMPT_TOK + MAXNEW)
    return max(1, min(n, int(free // per)))


def cost(model, n):
    """반환: (총초, 설명)"""
    if model not in HAS_W:
        return 0.0, f"{model}: 가중치 없음 → h15C 폴백 (생성 0 · ~0s)"
    B = max_batch(model, n)
    groups = (n + B - 1) // B
    L = PROMPT_TOK + MAXNEW / 2
    ms = ms_per_step(B, L)
    gen = groups * MAXNEW * ms / 1000.0
    pre = n * 20 * ms_per_step(1, PROMPT_TOK + MAXNEW) / 1000.0
    tot = LOAD_S + gen + pre
    return tot, (f"{model}: n={n} · 배치 {B}(KV 한도) · 그룹 {groups} · {ms:.1f} ms/step → "
                 f"생성 {gen:.0f}s + prefill {pre:.0f}s + 로드 {LOAD_S:.0f}s = **{tot:.0f}s**")


SHAPES = [
    ("리더보드형 Small (고유 문제 10 · 그룹 5·3·3·3)", [("Qwen3.5", 5), ("Skywork", 3), ("Olmo", 3), ("DeepSeek", 3)]),
    ("최종셋형 4모델 × 35문제", [(m, 35) for m in ("Qwen3.5", "Skywork", "Olmo", "DeepSeek")]),
    ("최종셋형 4모델 × 40문제", [(m, 40) for m in ("Qwen3.5", "Skywork", "Olmo", "DeepSeek")]),
    ("Main 5모델 × 21문제 (gpt-oss 는 폴백)", [(m, 21) for m in ("Qwen3.5", "Skywork", "Olmo", "DeepSeek")]),
    ("8모델 × 40문제 (논문 규모 상한)", [(m, 40) for m in ("Qwen3.5", "Skywork", "Olmo", "DeepSeek")] * 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/results/sub2048_time.txt"))
    a = ap.parse_args()
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    say(f"#SUB-GEN2048 시간 견적 · 창 {MAXNEW} · ms(B,L) = 1.15×(15 + 0.375B + 7.21e-4·B·L)")
    say(f"프롬프트 {PROMPT_TOK}토큰 가정 · 로드 {LOAD_S:.0f}s/모델 · VRAM {VRAM_GB:.0f}GB · "
        f"prefill = 생성 20스텝치(보수적)")
    say("")
    for name, groups in SHAPES:
        tot = 0.0
        say(f"## {name}")
        for m, n in groups:
            c, why = cost(m, n)
            tot += c
            say(f"   {why}")
        pct = tot / 3600 * 100
        say(f"   **합계 {tot:.0f}s = {tot / 60:.1f}분 · 3600s 의 {pct:.0f}%** → "
            f"{'들어간다' if tot <= 3300 else ('빡빡하다(SOFT 3300 초과)' if tot <= 3600 else '초과 → 폴백')}")
        say("")

    say("## 읽는 법")
    say("   SOFT=3300s 에서 제출물은 호출마다 `예산 = 남은시간 / 남은예상호출수` 를 재고,")
    say("   생성 추정이 그 예산의 80% 를 넘으면 **생성을 시작하지 않고** h15C 로 떨어진다.")
    say("   즉 견적이 틀려도 전량 무효는 없고 최악이 'h15C 와 같은 제출물' 이다.")
    say("")
    say("   병목은 **Olmo**다 — 토큰당 KV 512 KB(MHA)라 같은 문제 수에서 배치가 9분의 1로 쪼개지고")
    say("   그룹 수가 그만큼 늘어난다. 35~40문제에서 그 값이 전체 시간을 정한다.")
    say("")
    say("   이 모형은 **RTX PRO 6000 워커 실측 적합**이지만 창 2048 · 배치 수십의 조합은")
    say("   그 워커에서 직접 재 본 적이 없다(측정점 넷은 2k·16k 문맥, 배치 8~64).")
    say("   따라서 ±30% 오차를 가정하고 읽어야 하며, 그 불확실성이 폴백 설계의 이유다.")

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
