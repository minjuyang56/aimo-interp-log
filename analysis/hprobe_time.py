"""hprobe 시간표 — 생성이 없으니 비용은 **문제당 forward 1회**뿐이다. GPU 0. (지휘 #78-1)

입력은 **실측 두 개**만 쓴다. 추정 배수를 끼워 넣지 않는다:
  * 문제당 forward = **263 ms** — 로컬 RTX 2000 Ada 8 GB, Skywork 7B **4bit**, 공식 프롬프트(약 400토큰),
    4문제 연속 측정 (scratchpad/hprobe_e2e.py 로그). 첫 1회는 워밍업이라 2.07s 였고 그 뒤 0.26~0.28s.
  * 모델 로드 = **2 s** — 워커 warm cache 실측(CLAUDE.md 환경 제약 절). 로컬 4bit 는 37 s 였다.

워커는 RTX PRO 6000 96 GB · bf16 이라 로컬 4bit 보다 **빠를 수밖에 없다**(4bit 역양자화가 없고 대역폭이 크다).
그 배수를 모르므로 **곱하지 않는다** — 아래 표는 '워커가 내 노트북만큼 느리다면' 의 상한이다.
h20e 가 같은 규모에서 2,927 s 를 쓰고 그중 두 모델이 상한 포화로 무효였던 것과 비교하면 차수가 다르다.

사용:  .venv/Scripts/python.exe analysis/hprobe_time.py
"""

from pathlib import Path

MS_PER_PROBLEM = 263.0          # 실측(로컬 4bit)
LOAD_S = 2.0                    # 실측(워커 warm cache)
BUDGET = 3600.0
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "hprobe_time.txt"

SHAPES = [
    ("리더보드 Small 실측형 (5,3,3,3)", [5, 3, 3, 3]),
    ("최종셋형 4모델 × 35문제", [35] * 4),
    ("4모델 × 40문제", [40] * 4),
    ("Main 5모델 × 21문제", [21] * 5),
    ("논문 규모 8모델 × 40문제", [40] * 8),
]


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("hprobe 시간표 — 문제당 forward 1회(생성 없음)")
    out()
    out(f"실측 입력: 문제당 {MS_PER_PROBLEM:.0f} ms (로컬 4bit Skywork 7B) · 로드 {LOAD_S:.0f} s/호출 (워커 warm cache)")
    out("워커 bf16 속도 배수는 모르므로 곱하지 않았다 — 아래는 '워커가 로컬 4bit 만큼 느릴 때' 의 상한이다.")
    out()
    out(f"| 규모 | 호출 | 문제 합 | forward | 로드 | 합계 | 3600s 중 |")
    out("|---|---:|---:|---:|---:|---:|---:|")
    for tag, ns in SHAPES:
        probs = sum(ns)
        fw = probs * MS_PER_PROBLEM / 1000.0
        ld = len(ns) * LOAD_S
        tot = fw + ld
        out(f"| {tag} | {len(ns)} | {probs} | {fw:.0f}s | {ld:.0f}s | **{tot:.0f}s** | {tot / BUDGET:.1%} |")
    out()
    out("비교: h20e(모델별 생성 길이)는 리더보드형에서 **2,927 s** 를 썼고 그중 2모델은 토큰 상한 포화로 측정이 무효였다.")
    out("hprobe 는 생성이 0 이므로 포화라는 실패 양식이 원리상 없다 — 대신 신호가 가중치에 있어야 한다.")
    out()
    out("캐시: 점수는 (model_id, 문제) 단위로 캐시한다. 특징이 모델마다 다르므로 모델 간 재사용은 없다")
    out("(= 위 표의 forward 합은 캐시 적중 0 을 가정한 값이다).")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
