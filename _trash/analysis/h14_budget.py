"""H14 예산 (지휘 #43-4). 실측 기반, 착수 전 출력.

실측 근거 (p13b 커널, Skywork · vLLM · T4x2 · max_new 16384 · max_num_seqs 16):
  26프롬프트 × k=4 = 104샘플 · 4874s · **157.8 tok/s** → 샘플당 평균 **7,394 토큰**
  vLLM 로드 266s

H14 규모: 문제 33 × (원본 + rename 2종) = 99 프롬프트 × k
캐글 잔여(내 집계, 웹 콘솔 값 아님): 30h 중 사용 추정 18.6h → **약 11.4h**
사용:  .venv/Scripts/python.exe analysis/h14_budget.py
"""

from pathlib import Path

R, TOKS, LOAD = 157.8, 7394, 266
LEFT_H = 11.4


def main():
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    out(f"H14 예산 — R={R} tok/s · 샘플당 {TOKS} 토큰 · 로드 {LOAD}s · 캐글 잔여 약 {LEFT_H}h (내 집계)")
    out(f"{'구성':28s} {'샘플':>5s} {'시간':>7s} {'3h 이내?':>9s}")
    for kk in (2, 1):
        n = 99 * kk
        h = (LOAD + n * TOKS / R) / 3600
        out(f"{'문제 33 × 3종 × k=' + str(kk):28s} {n:5d} {h:6.2f}h {'예' if h <= 3 else '아니오':>9s}")
    out()
    out("선택: k=2 (2.63h) — 지휘 기준 3h 이내. k=1 은 1.35h 이지만 문제당 샘플 1개라 길이 분산이 커진다.")
    out()
    out("제출 시점 예산 (테스트타임 생성이 필요한 설계라 별도 계산):")
    out(f"  문제당 3회 생성 × 최악 16384토큰 = 49,152 토큰. 워커(RTX PRO 6000 bf16)에서 vLLM 합산 1,000 tok/s 가정 시 문제당 약 49s.")
    out(f"  **문제당 기준이면** 3600s 안에 충분히 들어간다(49s ≪ 3600s).")
    out(f"  **배치 합산 기준이면** 240 인스턴스 × 49,152 = 11.8M 토큰 → 1,000 tok/s 에서 11,800s로 **3600s 초과**.")
    out("  → 시간제한의 해석(문제당 vs 합산)에 따라 제출 가능성이 갈린다. Discord #qa 9/28 답변 2번은 '문제당'이라고 기록돼 있으나")
    out("    CLAUDE.md 다른 절에는 '3600초는 배치 전체 합산'이라고 적혀 있어 **상충한다. 주최측 재확인 필요.**")
    out("  합산 기준이라면 실제 평균 7,394토큰·완주 조기종료를 감안해도 240×3×7,394 = 5.3M → 5,300s 로 여전히 초과.")
    (Path(__file__).resolve().parent.parent / "experiments" / "results" / "h14_budget.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
