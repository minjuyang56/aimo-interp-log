"""자체 라벨 생성 비용·수확 추정 (지휘 #70 계획서용). GPU 0.

실측 입력값만 쓴다 (추정은 전부 여기서 파생):
  vLLM Skywork T4×2 fp16: 1단계(원본) **144.7 tok/s** · 2단계(변형) **287.7 tok/s**  — 9/30 스모크 실측
  샘플당 토큰: MATH level 3~4 에서 **2,773**(max_new 6144, f=0.45) 실측 → level 5/AIME 는 더 길어 **3,500** 보수 가정
  채택률: genlab v2 1차 선별(Skywork, AIME+level5 60문제, k=4, 미완주=오답 규칙)에서
          base_acc 분포 0.0×17 · 0.25×5 · 0.5×3 · 0.75×9 · 1.0×26 → 대역 [0.5, 1.0] 채택 **38/60 = 63%**
          그중 라벨이 갈린 비율은 v1 본실행에서 2% 였다(쉬운 문제) → level 5/AIME 로 올리고 **중간대역 제외율 30%** 가정
  Kaggle: 주 30h GPU · 동시 2세션 · 남은 26일 ≈ 3.7주 → **약 110 GPU-시간**
  로컬 GPU: 4bit 생성 약 10 tok/s → 하루 0.86M 토큰. 생성에는 못 쓴다(프로브 forward 전용)

사용:  .venv/Scripts/python.exe analysis/labelgen_plan.py
"""

TOK_S = 150.0          # 보수 처리율 (실측 144.7~287.7 중 낮은 쪽)
TOK_SAMPLE = 3500.0    # 샘플당 생성 토큰 (보수)
ACCEPT = 0.63          # 선별 채택률 (실측)
MIDBAND_DROP = 0.30    # max_drop 중간대역(0.11~0.24) 제외율 가정
KAGGLE_H = 110.0       # 남은 기간 Kaggle GPU 시간


def hours(tokens):
    return tokens / TOK_S / 3600.0


def scenario(M, models, N_var, k):
    """선별(원본 k샘플) → 채택분만 변형 N개 × k샘플."""
    screen = M * models * k * TOK_SAMPLE
    kept = M * ACCEPT
    var = kept * models * N_var * k * TOK_SAMPLE
    rows = kept * models
    labels = rows * (1 - MIDBAND_DROP)
    return dict(M=M, models=models, N=N_var, k=k, screen_h=hours(screen), var_h=hours(var),
                total_h=hours(screen + var), kept=kept, rows=rows, labels=labels)


if __name__ == "__main__":
    print(f"가정: {TOK_S:.0f} tok/s · 샘플 {TOK_SAMPLE:.0f} 토큰 · 채택률 {ACCEPT:.0%} · "
          f"중간대역 제외 {MIDBAND_DROP:.0%} · Kaggle 잔여 {KAGGLE_H:.0f} GPU-h")
    print()
    hdr = f"{'시나리오':28s} {'선별h':>7s} {'변형h':>7s} {'합계h':>7s} {'채택문제':>8s} {'(모델,문제)행':>12s} {'라벨':>7s} {'예산내':>6s}"
    print(hdr)
    for M in (100, 200, 400):
        for models, N, k in ((4, 4, 4), (2, 4, 4), (4, 6, 3)):
            r = scenario(M, models, N, k)
            ok = "O" if r["total_h"] <= KAGGLE_H else "X"
            tag = f"M={M} 모델{models} 변형{N} k{k}"
            print(f"{tag:28s} {r['screen_h']:7.1f} {r['var_h']:7.1f} "
                  f"{r['total_h']:7.1f} {r['kept']:8.0f} {r['rows']:12.0f} {r['labels']:7.0f} {ok:>6s}")
    print()
    p = scenario(20, 1, 4, 4)
    print(f"파일럿 (M=20 · 1모델 · 변형4 · k4): 선별 {p['screen_h']:.2f}h + 변형 {p['var_h']:.2f}h = "
          f"**{p['total_h']:.2f}h** · 채택 {p['kept']:.0f}문제 · 라벨 {p['labels']:.0f}개")
    print()
    print("현재 라벨 33개(문제 단위)와 비교한 배수:")
    for M, models, N, k in ((100, 4, 4, 4), (200, 4, 4, 4), (400, 2, 4, 4)):
        r = scenario(M, models, N, k)
        print(f"  M={M} 모델{models}: 라벨 {r['labels']:.0f}개 = 현재의 {r['labels'] / 33:.1f}배 "
              f"(모델×문제 행이므로 모델별 라벨 차이가 **처음으로 생긴다**)")
