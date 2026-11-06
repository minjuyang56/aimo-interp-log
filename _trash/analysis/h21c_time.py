"""h21c 예상 시간 — 워커(ms(B,L) 실측 모델)에서 문제당 비용을 쌓아 본다. GPU 0. (지휘 #68)

h21c 의 비용은 h20 계열과 **종류가 다르다**. 긴 생성이 없다:
  문제당 ① 즉답 8토큰 greedy(배치 1) ② 원본 teacher forcing forward 1회 ③ 변형 forward 1회
프리필은 프롬프트 길이에 선형이고, 디코딩은 8스텝뿐이라 ms(B,L) 의 2차항이 거의 없다.

워커 실측 기준값
  ms/step = 1.15 × (15 + 0.375·B + 7.21e-4·B·L)   (`analysis/ms_per_step_model.py`, 4점 적합 ±0.6ms)
  forward 1회(프리필 L토큰, 배치 1) ≈ ms/step 한 번에 L 토큰을 넣는 것과 같으므로
  보수적으로 **한 스텝 비용 × 1** 로 잡고, 거기에 KV 쓰기·오버헤드를 감안해 ×2 여유를 둔다.
로드는 워커 웜캐시 실측 2~8초 → 8초로 보수 가정.

사용:  .venv/Scripts/python.exe analysis/h21c_time.py
"""

A, B_, C_, MULT = 15.0, 0.375, 7.21e-4, 1.15
LOAD_S = 8.0
PROMPT = 400.0          # 올림피아드 문제 + 지시문 + 채팅 템플릿 (단어 중앙 52 → 토큰 300~500)
ANS = 8
SAFETY = 2.0            # forward 비용 상위 추정 배수


def ms_step(batch, ctx):
    return (A + B_ * batch + C_ * batch * ctx) * MULT


def per_problem_s():
    """① 즉답 8스텝 + ② ③ forward 2회."""
    decode = sum(ms_step(1, PROMPT + t) for t in range(ANS)) / 1000.0
    fwd = 2 * ms_step(1, PROMPT + ANS) * SAFETY / 1000.0
    return decode, fwd, decode + fwd


if __name__ == "__main__":
    d, f, tot = per_problem_s()
    print(f"문제당: 즉답 8스텝 {d * 1000:.0f} ms + forward 2회 {f * 1000:.0f} ms = **{tot * 1000:.0f} ms**")
    print(f"프롬프트 {PROMPT:.0f} 토큰 가정 · forward 안전배수 ×{SAFETY}\n")
    print(f"{'시나리오':34s} {'호출':>4s} {'문제/호출':>8s} {'측정':>6s} {'생성·forward':>12s} {'로드':>6s} {'합계':>8s} {'여유':>7s}")
    for tag, calls, n in (("리더보드형 Small (고유 10, 14 인스턴스)", 4, [5, 3, 3, 3]),
                          ("최종셋형 4호출 × 35문제", 4, [35] * 4),
                          ("최종셋형 4호출 × 40문제", 4, [40] * 4),
                          ("Main 5모델 × 21문제", 5, [21] * 5)):
        probs = sum(n)
        work = probs * tot
        load = calls * LOAD_S
        total = work + load
        print(f"{tag:34s} {calls:4d} {str(n):>8s} {probs:6d} {work:12.0f}s {load:6.0f}s {total:8.0f}s "
              f"{3600 - total:7.0f}s")
    print()
    print("캐시는 (model_id, effort, 문제) 단위라 모델이 바뀌면 다시 잰다 — 위 표는 그 전제(재사용 없음)다.")
    print("h20e(생성 길이)는 같은 규모에서 2,927s 를 썼고 그중 두 모델은 상한 포화로 측정이 무효였다.")


# ---- h21combo: 엔트로피(1024토큰) 비용을 더한 판 (지휘 #68 2순위) ----
def combo_rows():
    """엔트로피 생성은 배치 디코딩이라 ms(B,L) 의 2차항이 산다. 문제 수 n 을 한 배치로 본다."""
    rows = []
    for n in (5, 10, 21, 35, 40):
        lin = (A + B_ * n + C_ * n * PROMPT) * 1024 / 1000.0
        quad = C_ * n * 1024 * 1024 / 2000.0
        ent = (lin + quad) * MULT
        drop = n * per_problem_s()[2]
        rows.append((n, drop, ent, drop + ent))
    return rows


if __name__ == "__main__":
    print()
    print("h21combo (drop + 1024토큰 엔트로피) · 한 모델 호출 기준")
    print(f"{'문제':>4s} {'drop':>8s} {'엔트로피':>10s} {'합계':>8s}   4호출 합계(로드 포함)")
    for n, d2, e2, t2 in combo_rows():
        print(f"{n:4d} {d2:8.1f}s {e2:10.0f}s {t2:8.0f}s   {4 * t2 + 4 * LOAD_S:8.0f}s")
    print("  엔트로피 예산이 모자라면 drop 만으로 판정한다(= h21c 동작, 호출당 ENT_MIN_S=90s 문턱).")


# ---- h21cK: 변형 K개 (지휘 #69) ----
def ck_rows(K=4):
    rows = []
    for n in (5, 10, 21, 35, 40):
        decode = n * sum(ms_step(1, PROMPT + t) for t in range(ANS)) / 1000.0
        fwd = n * (1 + K) * ms_step(1, PROMPT + ANS) * SAFETY / 1000.0
        rows.append((n, decode, fwd, decode + fwd))
    return rows


if __name__ == "__main__":
    for K in (4, 8, 16):
        print()
        print(f"h21cK (변형 {K}개 평균) · 한 모델 호출 기준 · 문제당 forward {1 + K}회")
        print(f"{'문제':>4s} {'즉답':>8s} {'forward':>9s} {'합계':>8s}   4호출 합계(로드 포함)")
        for n, d2, f2, t2 in ck_rows(K):
            print(f"{n:4d} {d2:8.1f}s {f2:9.1f}s {t2:8.1f}s   {4 * t2 + 4 * LOAD_S:8.0f}s")
