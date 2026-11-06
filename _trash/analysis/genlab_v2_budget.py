"""genlab v2 예산 (지휘 #33-4). 착수 전 출력용.

토큰 수요 = 선별(N × k) + 변형(채택 × 3 × k) + 궤적(채택 × 1)  각 × 평균 생성 토큰
실측 기반: vLLM 합산 **144.7 tok/s**(1단계) · **287.7 tok/s**(짧은 프롬프트) → 보수적으로 **150 tok/s** 사용.
평균 생성 토큰: MATH level 3~4 에서 2,773. AIME 는 더 길어 **3,500** 으로 가정(보수적).
채택률: 대역 0.25~0.75 에 드는 비율. AIME 에서 7~8B 수학모델이면 **0.25** 가정(모르는 값 → 표로 범위 제시).

주간 한도 30h · 지금까지 사용 추정: H10-B 4.7 + 파일럿 1.16 + 프로브 1.2 + vLLM 스모크 0.3
                                   + Skywork run 2.3 + Skywork l5 0.8 + 실패 커널 5개 약 0.7 = **11.2h** → 잔여 약 **18.8h**
사용:  .venv/Scripts/python.exe analysis/genlab_v2_budget.py
"""

from pathlib import Path

R = 150.0          # tok/s (보수적)
TOK = 3500         # 샘플당 평균 생성 토큰 (AIME 가정)
LOAD = 300         # vLLM 로드 + HF 궤적 로드
USED_H, WEEKLY_H = 11.2, 30.0


def hours(n, k, keep_rate):
    kept = n * keep_rate
    gens = n * k + kept * 3 * k + kept
    return (LOAD + gens * TOK / R) / 3600, int(kept)


def main():
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    left = WEEKLY_H - USED_H
    out(f"genlab v2 예산 — R={R} tok/s · 샘플당 {TOK} 토큰 · 잔여 약 {left:.1f}h")
    out(f"{'구성':30s} {'채택수':>6s} {'모델당':>8s} {'2모델':>8s} {'4모델':>8s} {'잔여 안?':>9s}")
    for n in (90, 150):
        for k in (4, 3):
            for keep in (0.25, 0.40):
                h, kept = hours(n, k, keep)
                tag = f"N={n} · k={k} · 채택률 {keep:.0%}"
                ok = "4모델" if 4 * h <= left else ("2모델" if 2 * h <= left else "1모델" if h <= left else "불가")
                out(f"{tag:30s} {kept:6d} {h:7.2f}h {2 * h:7.2f}h {4 * h:7.2f}h {ok:>9s}")
    out()
    out("권고: **N=90(AIME 전량) · k=4** 로 Skywork 먼저 — 채택률 25%면 모델당 4.3h, 4모델 17.3h(잔여 18.8h에 간신히), 라벨 후보 22개.")
    out("      라벨 후보가 30개 미만이면 MATH level 5 를 source 에 더해 N 을 늘린다(같은 문제집합을 모델 간 공유).")
    out("      k=4 는 정확도 해상도가 0/0.25/0.5/0.75/1 뿐이다 — 대역 0.25~0.75 는 사실상 '1~3개 맞힘'을 뜻한다.")
    out("주의: 채택률은 미지수다. Skywork 선별 단계 결과(정확도 분포)를 보고 나머지 모델의 N 을 정한다.")
    (Path(__file__).resolve().parent.parent / "experiments" / "results" / "genlab_v2_budget.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
