"""자체 라벨 생성 Kaggle 예산 — **vLLM 실측 반영 (지휘 #30, 2026-09-30 01:20).**

측정: `aimo-genlab-skywork-or1-math-7b-vllm` (Skywork · T4×2 fp16 · vLLM tensor_parallel=2 · max_num_seqs 48 · MATH level 3~4 · max_new 6144)
  vLLM 로드                 267 s
  1단계 원문제 16개          307 s · **144.7 tok/s** · 44,370 토큰 → 문제당 2,773 토큰 (f = 2773/6144 = 0.45)
  2단계 변형 53건            416 s · **287.7 tok/s** → 변형당 2,258 토큰
  완주 12/16 · 정답 12/16 → **완주한 것 중 12/12 정답** (level 3~4 + 6144 토큰으로 정답률 문제 해결)
  라벨 집계에서 KeyError 로 죽어 궤적 단계는 못 돌았다 → plan 없는 문제 처리 고쳐서 본실행에 반영.

이전 HF 경로 대비: 16.1 → 144.7 tok/s (**9배**). 변형 단계는 프롬프트가 짧아 더 빨라 287.7.

시간(모델당) ≈ 로드 267 + N × 2773/144.7 + N × p × V × 2258/287.7 + 궤적(로드 ~150 + 문제당 ~3 s)

사용:  .venv/Scripts/python.exe analysis/genlab_budget.py
"""

import argparse
from pathlib import Path

USED_H = 7.4          # H10-B 4.7 + 파일럿 1.16 + 프로브 1.2 + vLLM 스모크 0.3
WEEKLY_H = 30.0
R1, R2 = 144.7, 287.7          # tok/s
TOK1, TOK2 = 2773, 2258        # 문제당 · 변형당 생성 토큰
P_CORRECT = 0.75               # 완주·정답 비율 (12/16)


def hours(n, v, p=P_CORRECT):
    sec = 267 + n * TOK1 / R1 + n * p * v * TOK2 / R2 + 150 + n * 3
    return sec / 3600


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", type=int, default=4); a = ap.parse_args()
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    left = WEEKLY_H - USED_H
    out(f"예산 (vLLM 실측) — R1={R1} R2={R2} tok/s · 정답률 {P_CORRECT} · 주간 {WEEKLY_H}h 중 {USED_H}h 사용, 잔여 {left:.1f}h")
    out(f"{'구성':34s} {'모델당':>8s} {'2모델':>8s} {'4모델':>8s} {'라벨 수/모델':>12s} {'잔여 안?':>9s}")
    for n in (100, 150, 200, 300):
        for v in (5, 3):
            h = hours(n, v)
            tag = f"N={n} · 공식3+비공식2" if v == 5 else f"N={n} · 공식2+비공식1"
            out(f"{tag:34s} {h:7.2f}h {2 * h:7.2f}h {4 * h:7.2f}h {int(n * P_CORRECT):12d} {'4모델 가능' if 4 * h <= left else ('2모델 가능' if 2 * h <= left else '불가'):>9s}")
    out()
    out("선택: **N=150 · 공식3+비공식2** → 모델당 약 1.2h, 4모델 4.8h, 모델당 라벨 약 112개 (현재 v2 33개의 3.4배).")
    out("검정력 메모(analysis/power_33.py): n=33 에서는 참 AUC 0.75도 검출력이 낮다. n≈112 면 0.70 전후부터 판정이 가능해진다.")
    out("본실행 커널: [author-kaggle]/aimo-genlab-skywork-or1-math-7b-run · [author-kaggle]/aimo-genlab-deepseek-r1-0528-qwen3-8b-run (N=150).")
    out("남은 2모델(Olmo-3-7B-Think · Qwen3.5-4B)은 GPU 슬롯이 비면 올린다. Qwen3.5는 vLLM 미지원이면 HF 폴백.")
    (Path(__file__).resolve().parent.parent / "experiments" / "results" / "genlab_budget.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
