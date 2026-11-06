"""#KAGGLE-0 (2) 견적 — Kaggle 무료 GPU 로 k>=16 새 라벨을 만들 때의 문제당 시간·N·주 수. $0.

**이 견적의 전제는 `power_vs_n.py` + `power_select.py` 두 측정이다** (결과를 보고 쓴 것이 아니라
이 견적의 입력이다):
  * 현 파이프라인(StandardScaler→PCA24→LR)은 d=4096 에서 **n=300 까지 늘려도** 8σ 신호를 못 찾는다.
  * 폴드 안 단변량 선별(S)로 바꾸면 n=33 0.701 → **n=100 0.818** → n=200 0.972 로 뚫린다.
→ 즉 '라벨을 더 만든다' 는 **파이프라인을 S 로 바꿀 때만** 가치가 있고, 그때 목표 n 은 **약 100/모델**이다.

실측 단가(전부 커밋된 로그): Skywork · Kaggle T4x2 · cap 16384 · k=4
  선별 231 s/문제 (144.1 tok/s) · 변형 160 s/건 (174.7 tok/s) · 변형/문제 2.8 · 문제당 총 636 s
토큰당 KV (CLAUDE.md 표): Skywork 56 · Qwen3.5 128 · DeepSeek 144 · Olmo 512 KB.
T4x2 32 GB 에서 가중치를 빼면 KV 풀이 약 12.2 GB (#LG-32k 실측).

사용: .venv/Scripts/python.exe analysis/kaggle_labelgen_est.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "experiments/results/kaggle_labelgen_est.txt"

# 실측 단가 (k=4 · cap 16384 · Skywork)
SCREEN_S_K4 = 231.0
VAR_S_K4 = 160.0
VAR_PER_PROB = 2.8
KV_KB = {"Skywork": 56, "Qwen3.5-4B": 128, "DeepSeek": 144, "Olmo": 512}
WEIGHT_GB = {"Skywork": 15.2, "Qwen3.5-4B": 9.3, "DeepSeek": 16.4, "Olmo": 14.6}
KV_POOL_GB = 12.2          # T4x2 실측
QUOTA_H = 30.0             # 주간 GPU 시간 (CLAUDE.md 기록, API 로는 노출되지 않는다)
SESSION_H = 9.0            # 세션 상한
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    out("#KAGGLE-0 (2) 견적 — k>=16 새 라벨 생성, Kaggle T4x2")
    out(f"   주간 쿼터 {QUOTA_H} GPU-h · 세션 상한 {SESSION_H} h · 실측 단가는 k=4·cap 16384·Skywork")
    out()

    out("## (ㄱ) 모델별 실현 가능성 — cap 16384 에서 동시 시퀀스 수가 전부를 정한다")
    out(f"   {'모델':<12}{'가중치GB':>9}{'KV/토큰':>9}{'시퀀스당KV':>11}{'동시':>6}  판정")
    feasible = []
    for m, kv in KV_KB.items():
        per_seq = kv * 16384 / 1e6          # GB
        conc = int(KV_POOL_GB / per_seq)
        ok = conc >= 4
        if ok:
            feasible.append(m)
        note = "쓸 수 있다" if ok else f"**동시 {conc} → 사실상 불가**"
        out(f"   {m:<12}{WEIGHT_GB[m]:>9.1f}{kv:>8} KB{per_seq:>10.1f}G{conc:>6}  {note}")
    out()
    out("   → Olmo 는 토큰당 KV 512 KB(MHA)라 cap 16384 에서 동시 1 시퀀스다. 배치가 성립하지 않아")
    out("     벽시계가 비선형으로 늘어난다(#LG-32k 에서 cap 32768 은 시작조차 못 했다).")
    out("   → Qwen3.5 는 vLLM 이 아키텍처를 못 읽어 HF 폴백이고 실측 9배 느리다(지휘 #33-A).")
    out(f"   → **쓸 수 있는 모델: {', '.join(m for m in feasible if m != 'Qwen3.5-4B')}** (2종)")
    out()

    out("## (ㄴ) k 별 문제당 시간 (선별 k + 변형 2.8유형 x k, 실측 선형 환산)")
    out(f"   {'k':>4}{'선별 s':>9}{'변형 s':>9}{'문제당 s':>10}{'문제당 h':>10}{'주당 문제':>10}")
    per_prob = {}
    for k in (4, 8, 16, 32):
        sc = SCREEN_S_K4 * k / 4
        va = VAR_S_K4 * k / 4 * VAR_PER_PROB
        tot = sc + va
        per_prob[k] = tot
        out(f"   {k:>4}{sc:>9.0f}{va:>9.0f}{tot:>10.0f}{tot/3600:>10.3f}{QUOTA_H*3600/tot:>10.1f}")
    out()

    out("## (ㄷ) 목표 N = 100/모델 (power_select 에서 S 가 0.818 로 뚫리는 지점)")
    out(f"   {'k':>4}{'모델':>4}{'문제':>6}{'GPU-h':>9}{'주 수':>7}{'세션 수':>8}  판정")
    for k in (8, 16):
        for nmod in (1, 2):
            n = 100 * nmod
            h = n * per_prob[k] / 3600
            wk = h / QUOTA_H
            sess = h / SESSION_H
            fits = "이번 주 안" if h <= QUOTA_H else f"**{wk:.1f}주 필요**"
            out(f"   {k:>4}{nmod:>4}{n:>6}{h:>9.1f}{wk:>7.1f}{sess:>8.1f}  {fits}")
    out()

    out("## (ㄹ) 남은 달력 — 이게 결정적이다")
    out("   오늘 2026-10-18 · 대회 phase 종료 **2026-11-01** (2.0주) · 리포트 마감 2026-11-15 (4.0주)")
    k16_2 = 200 * per_prob[16] / 3600
    out(f"   k=16 · 2모델 · N=100 = **{k16_2:.0f} GPU-h = {k16_2/QUOTA_H:.1f}주** → phase 종료를 넘고")
    out(f"   리포트 마감({4.0:.1f}주)도 넘는다. 라벨이 도착해도 쓸 시간이 없다.")
    k8_2 = 200 * per_prob[8] / 3600
    out(f"   (참고) k=8 로 내리면 {k8_2:.0f} GPU-h = {k8_2/QUOTA_H:.1f}주 — 지휘가 k>=16 으로 못 박았고,")
    out("    #ORG-LABEL 이 'k=8 격자 0.125 는 주최측 0.01 의 12배 거칠다' 를 이미 쟀다.")
    out()

    out("## 판정")
    out("   **이번 주 할당(30 GPU-h) 안에 들어가지 않는다** → 지휘 (3) 대로 **계획만 기록하고")
    out("   커널은 띄우지 않는다.** 근거 셋:")
    out(f"     1. k=16·2모델·N=100 = {k16_2:.0f} GPU-h = 쿼터의 {k16_2/QUOTA_H:.1f}배")
    out("     2. 쓸 수 있는 모델이 2종뿐이다(Olmo 동시 1 · Qwen3.5 HF 폴백 9배 느림)")
    out("        → 지휘가 요구한 '모델 간 라벨 차이' 를 4모델로 못 잰다")
    out("     3. 달력이 더 짧다 — phase 종료가 2.0주 뒤다")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
