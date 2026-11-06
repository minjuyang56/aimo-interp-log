"""#FLIPENT-ZIP 시간 견적 — 3300s 안에 드나. **`analysis/ttp_budget.py` 의 함수를 import 해서** 쓴다.

지휘가 "기존 ttp_budget 숫자로 3300초 예산을 확인하라" 고 했으므로 시간모델을 다시 쓰지 않고
그 파일의 `gen_seconds`(= ms(B,L) 실측 적합 + KV 표 + 패딩 배치 그룹화)를 그대로 부른다.
**재현 게이트**: 같은 함수로 ttp 의 기록값(K=8 · cap 1024 · Small 4모델x40문제 = **2,176s**)을
재현하지 않으면 멈춘다 — 그러면 import 경로가 어긋난 것이다.

#FLIPENT-ZIP 은 섭동이 없다 → **문제당 시퀀스 1개**(K=0), cap 1024 고정, 강제 즉답 없음.
그래서 ttp 의 K=8 대비 생성 시퀀스가 1/9 이다.

사용: .venv/Scripts/python.exe analysis/flipent_time.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import ttp_budget as TB  # noqa: E402

REPORT = ROOT / "experiments/results/flipent_time.txt"
CAP = 1024
WORKER_VRAM = 96.0
#  Small 트랙 4모델. gpt-oss-120b 는 Main 전용이고 우리 가중치가 없어 **생성 0**(길이 폴백)이다.
SMALL = ["Skywork", "Qwen3.5", "DeepSeek", "Olmo"]
LOAD_S = 3.0        # warm cache 로드 실측 약 2초(CLAUDE.md 워커 절) — 모델당 보수적으로 3초
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    out("#FLIPENT-ZIP 시간 견적 — ttp_budget.gen_seconds 를 import · cap %d · 문제당 시퀀스 1개" % CAP)
    out("   시간모델 ms(B,L) = 1.15 x (15 + 0.375B + 7.21e-4 B L) · 프롬프트 %d 토큰 · 워커 96 GB"
        % TB.PROMPT)
    out()

    #  -- 재현 게이트: ttp 기록값(K=8 cap 1024, Small 4x40 = 2176s) --
    tot = 0.0
    for m in SMALL:
        s, _, _ = TB.gen_seconds(m, 40 * 9, CAP, WORKER_VRAM)
        a = TB.ans_seconds(m, 40 * 9, CAP, WORKER_VRAM)
        tot += s + a
    ok = abs(tot - 2176) <= 15
    out("## 재현 게이트 — 같은 함수로 ttp K=8 cap 1024 (Small 4모델x40문제) 를 다시 낸다")
    out("   계산 %.0fs vs ttp_budget.txt 기록 2176s → %s" % (tot, "일치" if ok else "**불일치**"))
    if not ok:
        REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
        raise SystemExit("재현 게이트 실패 — ttp_budget 숫자를 재현하지 못했다")
    out()

    #  -- flipent 본 견적 --
    out("## 제출 환경 — 호출마다 그 모델 하나만 열고 문제 전부를 한 배치로 생성 (합산 안전 3300s)")
    out("   %-26s %9s %9s %9s %9s %10s %8s" % ("규모", "Skywork", "Qwen3.5", "DeepSeek", "Olmo", "합계", "3300s 대비"))
    rows = [("리더보드형 Small (5,3,3,3)", [5, 3, 3, 3]),
            ("최종셋형 4모델x35문제", [35] * 4),
            ("최종셋형 4모델x40문제", [40] * 4),
            ("8모델x40문제 (논문 규모)", [40] * 4)]
    for label, nprobs in rows:
        per, tot = [], 0.0
        for m, n in zip(SMALL, nprobs):
            s, groups, slots = TB.gen_seconds(m, n, CAP, WORKER_VRAM)
            s += LOAD_S
            per.append((m, s, groups, slots))
            tot += s
        if label.startswith("8모델"):
            tot *= 2            # 8모델이면 같은 네 묶음이 두 번 (논문 규모 상한 가정)
        out("   %-26s %9.0f %9.0f %9.0f %9.0f %10.0f %7.1f%%"
            % (label, per[0][1], per[1][1], per[2][1], per[3][1], tot, 100 * tot / 3300))
    out()
    gl = []
    for m in SMALL:
        s, groups, slots = TB.gen_seconds(m, 40, CAP, WORKER_VRAM)
        gl.append("%s 칸 %d → 그룹 %d" % (m, slots, groups))
    out("   배치 그룹화(4모델x40문제): " + " · ".join(gl))
    out()
    out("## 읽는 법")
    out("   * 섭동이 없어 **문제당 시퀀스가 1개**다 → ttp(K=8, 2,176s)의 약 1/9 이고 예산에 여유가 크다.")
    out("   * cap 을 **깎지 않는다**: 점수가 '앞 25%' 라 T 가 바뀌면 재는 구간이 바뀌어 고정 임계가")
    out("     무의미해진다. 그래서 제출물은 cap 1024 를 못 사면 **생성을 시작하지 않고 h15C 로 떨어진다**")
    out("     (최악 = h15C 와 같은 제출물, 전량 무효 없음).")
    out("   * 병목은 Olmo(토큰당 KV 512 KB · MHA)이고 40문제까지는 한 배치에 들어간다.")
    out("   * 모형이 ±30% 틀려도 무효가 없다 — 제출물이 생성 전에 `추정 > 예산x0.8` 을 보고 폴백한다.")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(REPORT))


if __name__ == "__main__":
    main()
