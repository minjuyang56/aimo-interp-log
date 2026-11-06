"""#GEN-PROBE 단계 0(b) — 풀이 텍스트 재생성 비용. GPU 0. 실측만 쓴다(추정 배수는 출처를 적는다).

## 왜 필요한가
#GEN-PROBE 의 특징은 '모델 자신의 원문제 풀이 도중 벡터'다. 그런데 k=8 라벨 생성 실행은
`--no-traj` 로 돌았고(지휘 #72: 파일럿2 에서 궤적 단계 OOM 10건) `greedy_text` 는 그 분기에서만
저장된다 → **풀이 텍스트가 어느 산출물에도 없다**(label/screen JSON 에 정확도 집계만 있다).
그래서 특징을 만들려면 원문제를 **다시 생성**해야 한다. 이 스크립트가 그 비용을 낸다.

## 모델
비용 = 토큰 ÷ 처리율. 둘 다 실측에서 온다:
  * seq당 토큰 = 그 모델 선별 단계의 (초 × tok/s) ÷ (40문제 × k)  — 같은 모델·같은 cap·같은 풀
  * 처리율 = 그 선별의 tok/s 에 **저동시성 배수**를 곱한다. 배수의 출처는 Olmo 선측정
    (5문제 × k4 = 20seq → 338.2 tok/s) 대비 같은 모델 320seq 선별(571.6 tok/s) = 0.59.
    k=1 재생성은 모델당 23~41 시퀀스라 320seq 보다 동시성이 낮다 → 처리율이 떨어진다.
"""
RATIO = 338.2 / 571.6          # 저동시성(20seq) / 고동시성(320seq) 실측 비
PRICE = 2.89                   # H100 PCIe $/h (원장 실측 단가)
LOAD = 200.0                   # vLLM 로드 1회 실측 범위 93~267s 중 보수값
SETUP = 600.0                  # 파드 생성·pip·가중치 내려받기 (실측 여유분)

# (모델, 재생성할 문제 수, 선별 seq당 토큰, 선별 tok/s@320seq)
JOBS = [
    ("Skywork  off40 라벨 30", 30, 8364, 1785.5),
    ("Olmo     off40 라벨 31", 31, 8890, 571.6),
    ("DeepSeek off40 라벨 23", 23, 15735, 764.9),
    ("DeepSeek off80 라벨 18", 18, 14834, 749.2),
]


def table(caps):
    for cap in caps:
        print(f"\n## 생성 상한 cap {cap:,} (seq당 토큰을 그 값으로 자른다)")
        tot = 0.0
        for name, n, per, thr in JOBS:
            tk = n * min(per, cap)
            s = tk / (thr * RATIO)
            tot += s
            print(f"   {name:24s} {n:3d}seq · {tk/1e6:5.2f}M tok · {thr*RATIO:6.0f} tok/s → {s:7.0f}s")
        # 모델은 3종(DeepSeek 두 슬라이스는 한 번 로드)
        over = 3 * LOAD + SETUP
        h = (tot + over) / 3600
        print(f"   생성 합 {tot:,.0f}s + 로드3회·설치 {over:,.0f}s = {h:.2f}h → **${h*PRICE:.2f}**")
        print(f"   수명 상한을 {h*1.6:.1f}h 로 잡으면 최악 ${h*1.6*PRICE:.2f}")


if __name__ == "__main__":
    print(__doc__.split("## 모델")[0].strip())
    print(f"\n저동시성 배수 {RATIO:.2f} (실측 338.2/571.6) · 단가 ${PRICE}/h")
    table([32768, 8192, 4096])
