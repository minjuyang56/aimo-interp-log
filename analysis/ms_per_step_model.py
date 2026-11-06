"""ms/step 모델 적합 — 배치 크기 B 와 평균 문맥 길이 L 로. (지휘 #61-1)

측정점 (전부 워커 실측)
  probe2 (짧은 문맥, 2048스텝, 프롬프트 ~190):  B=8 → 25.0 · B=32 → 54.4 · B=64 → 95.5 ms/step
  h20 1차 (cap 16384, 5문제 × k2 = B 10):       1297s / 16384스텝 = 79.2 ms/step
형태: ms = a + b·B + c·B·L   (고정 오버헤드 + 가중치 읽기 + 어텐션의 B×L)
  짧은 문맥(평균 L ≈ 190 + 2048/2 ≈ 1214)에서 ms ≈ 15 + 1.25·B 로 거의 선형 → a=15, b + 1214c = 1.25
  긴 문맥(B=10, 평균 L ≈ 190 + 16384/2 ≈ 8382)에서 79.2 → b + 8382c = 6.42
  두 식에서 c = 7.21e-4, b = 0.375
보수적으로 **×1.15** 를 곱해 상위 추정을 쓴다(워커 로그의 60~97 tok/s 산포를 감안).

한 배치의 소요(초) = ∫0..C [a + bB + cB(P + t)] dt / 1000
                  = C(a + bB + cBP)/1000 + cB·C²/2000      (P = 프롬프트 길이, C = cap)
배치를 쪼개면 g배의 고정비 a 가 더 붙으므로 **한 배치가 항상 유리**하다.

사용:  .venv/Scripts/python.exe analysis/ms_per_step_model.py
"""

A, B_COEF, C_COEF, MARGIN_MULT, PROMPT = 15.0, 0.375, 7.21e-4, 1.15, 200.0


def ms_per_step(batch, ctx):
    return (A + B_COEF * batch + C_COEF * batch * ctx) * MARGIN_MULT


def seconds_for(batch, cap, prompt=PROMPT):
    """cap 스텝을 배치로 생성할 때의 예상 초 (문맥이 자라는 것을 적분)."""
    lin = (A + B_COEF * batch + C_COEF * batch * prompt) * cap / 1000.0
    quad = C_COEF * batch * cap * cap / 2000.0
    return (lin + quad) * MARGIN_MULT


def max_cap(batch, budget_s, caps=(4096, 6144, 8192, 12288, 16384), prompt=PROMPT):
    ok = [c for c in caps if seconds_for(batch, c, prompt) <= budget_s]
    return max(ok) if ok else None


if __name__ == "__main__":
    print("적합 확인 (보수 배수 1.15 제외한 원식)")
    for b, ctx, obs in ((8, 1214, 25.0), (32, 1214, 54.4), (64, 1214, 95.5), (10, 8382, 79.2)):
        pred = A + B_COEF * b + C_COEF * b * ctx
        print(f"  B={b:3d} L={ctx:5.0f} · 실측 {obs:5.1f} · 예측 {pred:5.1f} ms/step · 오차 {pred - obs:+.1f}")
    print()
    print("배치×cap 별 예상 소요(초, 보수 1.15 포함)")
    print(f"{'B':>4s} " + " ".join(f"{c:>8d}" for c in (4096, 6144, 8192, 12288, 16384)))
    for b in (4, 6, 10, 20, 30, 40, 60, 80):
        print(f"{b:4d} " + " ".join(f"{seconds_for(b, c):8.0f}" for c in (4096, 6144, 8192, 12288, 16384)))
    print()
    print("시나리오별 예상 (호출예산 = 3300/4 = 825s 가정)")
    for tag, calls in (("A 리더보드형 4호출 (5,3,3,3) · 고유 10문제", [5, 3, 3, 3]),
                       ("B 최종셋형 4호출 × 10문제 (고유 40)", [10, 10, 10, 10])):
        print(f"  {tag}")
        for k in (2, 3):
            tot, uniq = 0.0, 0
            for i, n in enumerate(calls):
                new = n if tag.startswith("B") else (n if i == 0 else max(0, n - 2))
                if new == 0:
                    continue
                batch = new * k
                cap = max_cap(batch, 825)
                sec = seconds_for(batch, cap) if cap else 0
                tot += sec
                uniq += new
            print(f"    k={k}: 고유 {uniq}문제 · 총 {tot:6.0f}s = {tot / 60:4.1f}분 · "
                  f"호출1 cap {max_cap(calls[0] * k, 825)}")
