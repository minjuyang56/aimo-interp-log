"""Ġ 디토큰화 버그로 오염된 **DeepSeek 관련 RunPod 지출** 집계. (지휘, 2026-10-11) GPU 0.

왜 '오염' 과 '손실' 과 '복구' 를 나누나: 셋은 성격이 다르고 섞어 한 숫자로 내면 과장된다.
  * **오염** = Ġ 손상 텍스트 위에서 산출물(라벨·선별)을 만든 지출. 산출물이 쓸 수 없게 됐다.
  * **복원됨** = 손상됐지만 왕복 재디코드로 되살린 지출(genprobe 의 greedy 텍스트).
    텍스트를 저장했기 때문에 살았다 — 상설 규칙 ①의 금액 근거다.
  * **딴 이유 손실** = DeepSeek 지출이지만 Ġ 와 무관하게 날아간 것(회수 누락·수명 상한).
  * **복구** = #DS-RELABEL 재생성 지출. 사고의 대가지만 '오염된 지출' 은 아니다.

파드별 모델은 **remote.log 에서 읽는다**(원장에는 모델이 없다). 혼합 파드(genprobe, 3모델)는
로그에 찍힌 **모델별 로드+생성 초**로 배분한다 — 파드 셋업 시간은 모델이 공유하므로
그 비율을 파드 실비 전체에 적용한다(따로 떼어 낼 근거가 없다).

사용: .venv/Scripts/python.exe analysis/dsbug_cost.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "runpod"))
from launch import HARD_CAP_USD, ledger_runs, spent_usd  # noqa: E402

OUT = ROOT / "experiments" / "results" / "dsbug_cost.txt"

# 파드 분류 — 근거는 각 파드의 remote.log 다(모델 줄·산출 줄을 직접 읽었다).
# 'why' 는 그 지출이 무엇을 만들었는지, 'use' 는 그 산출물이 어떤 결론에 쓰였는지.
CONTAM = {                                     # Ġ 손상 산출물을 낸 DS 단독 파드
    "67elv2je7wzefw": ("DS off40 k=8 선별 40문제", "손상된 base_acc → 채택(>=0.5) 선별이 왜곡"),
    "luvw8yu3mkxsza": ("DS off40 k=8 변형·라벨 23건", "#77 DS 팔 · #GEN-PROBE 주묶음 · 외부전이 0.792"),
    "1f7jul0dgk7o9s": ("DS off80 k=8 선별·변형·라벨 18건", "#DS-REP 학습셋"),
    "x04ghnc52b6dll": ("DS off80 k=4 라벨 26건", "프로브 학습엔 안 썼다(라벨 품질표에만 등장)"),
}
RECOVERED = {                                  # 손상됐으나 텍스트를 저장해 되살린 혼합 파드
    "oi8q4v8ia3ojq0": ("genprobe greedy 3모델", {"Skywork": 80 + 386, "DeepSeek": 85 + 710,
                                                 "Olmo": 74 + 820}),
}
OTHER_LOSS = {
    "p2x24f1mlossk5": ("DS off80 k=4 선별 3,097초", "회수가 작업 종료 후에만 돌아 산출물 유실 "
                                                   "(Ġ 와 무관한 운영 사고)"),
}
REMEDY = {                                     # #DS-RELABEL 재생성 (사고의 대가, 오염 지출은 아니다)
    "mbpc29h3v7um64": "DS off40 k=8 재생성",
    "9svqsw8dzrtjsv": "DS off80 k=8 재생성",
}
lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def main():
    runs = {r.get("pod_id"): r for r in ledger_runs() if r.get("pod_id")}

    def cost(pid):
        r = runs.get(pid)
        if r is None:
            return None, None
        c = r.get("cost_usd")
        return (float(c), "실비") if c is not None else (float(r.get("worst_usd", 0)), "미정산(최악)")

    out("Ġ 디토큰화 버그 — DeepSeek 관련 RunPod 지출 집계 (원장 실비 · GPU 0)")
    out("파드별 모델은 remote.log 에서 읽었고, 혼합 파드는 모델별 로드+생성 초로 배분했다.")
    out()

    out("## (A) 오염 — Ġ 손상 텍스트 위에서 산출물을 만든 지출 (복원 불가)")
    a = 0.0
    for pid, (what, use) in CONTAM.items():
        c, kind = cost(pid)
        if c is None:
            out(f"   {pid}  원장에 없음")
            continue
        a += c
        out(f"   {pid}  ${c:>6.2f} {kind:<12} {what}")
        out(f"   {'':16}         └ 쓰인 곳: {use}")
    out(f"   **(A) 합계 ${a:.2f}**")
    out()

    out("## (B) 손상됐으나 복원됨 — 텍스트를 저장했기 때문에 살았다")
    b = 0.0
    for pid, (what, shares) in RECOVERED.items():
        c, kind = cost(pid)
        if c is None:
            continue
        tot = sum(shares.values())
        ds = shares["DeepSeek"] / tot
        b += c * ds
        out(f"   {pid}  파드 ${c:.2f} {kind} · {what}")
        out(f"   {'':16}  모델별 초 " + " · ".join(f"{k} {v}" for k, v in shares.items())
            + f" (합 {tot})")
        out(f"   {'':16}  DS 비율 {ds:.3f} → **DS 배분 ${c * ds:.2f}**")
        out(f"   {'':16}  왕복 재디코드로 마커 0 복원 → 특징은 되살렸다. 라벨은 텍스트가 없어 못 살렸다.")
    out(f"   **(B) DS 배분 합계 ${b:.2f}**")
    out()

    out("## (C) DS 지출이지만 Ġ 와 무관하게 날아간 것")
    c_tot = 0.0
    for pid, (what, why) in OTHER_LOSS.items():
        c, kind = cost(pid)
        if c is None:
            continue
        c_tot += c
        out(f"   {pid}  ${c:>6.2f} {kind:<12} {what}")
        out(f"   {'':16}         └ {why}")
    out(f"   **(C) 합계 ${c_tot:.2f}** — 이 건은 Ġ 오염 집계에 넣지 않는다")
    out()

    out("## (D) 복구 — #DS-RELABEL 재생성 (사고의 대가, 오염 지출은 아니다)")
    d = 0.0
    for pid, what in REMEDY.items():
        c, kind = cost(pid)
        if c is None:
            out(f"   {pid}  아직 원장에 없음 — {what}")
            continue
        d += c
        out(f"   {pid}  ${c:>6.2f} {kind:<12} {what}")
    out(f"   **(D) 합계 ${d:.2f}**")
    out()

    s = spent_usd()
    unsettled = [pid for pid in REMEDY if runs.get(pid) and runs[pid].get("cost_usd") is None]
    out("## 합계")
    if unsettled:
        out(f"   !! 주의: 재생성 파드 {len(unsettled)}개가 **미정산**이라 (D) 는 실비가 아니라")
        out(f"      수명 상한 × 시간당 가격의 **최악값**이다 (파드 {', '.join(unsettled)}).")
        out("      파드가 끝나면 이 스크립트를 다시 돌려 (D) 와 총액을 실비로 바꿀 것.")
    out(f"   **Ġ 로 오염된 DeepSeek 지출 = (A) + (B) = ${a + b:.2f}**")
    out(f"     그중 복원 불가(라벨·선별) (A) ${a:.2f} · 복원됨(greedy 텍스트) (B) ${b:.2f}")
    out(f"   참고: Ġ 무관 DS 손실 (C) ${c_tot:.2f} · 재생성 (D) ${d:.2f}")
    out(f"   사고가 만든 총 비용 = (A) + (D) = **${a + d:.2f}** "
        f"(오염된 산출물 + 그걸 다시 만드는 값)")
    out(f"   원장 누적 ${s:.2f} / 하드캡 ${HARD_CAP_USD:.2f} · 남음 ${HARD_CAP_USD - s:.2f}")
    out(f"   오염 지출이 누적에서 차지하는 비중 {(a + b) / max(1e-9, s):.0%}"
        + ("  (누적에 미정산 예약이 섞여 있어 분모가 과대 → 비중은 하한)" if unsettled else ""))
    out()
    out("## 이 숫자가 말하는 것")
    out("   (A) 는 **집계만 저장했기 때문에** 복원 불가가 됐다. (B) 는 같은 버그를 맞았지만")
    out("   텍스트를 저장해서 재디코드로 살았다 — 같은 사고, 다른 결과. 상설 규칙 ①(원문 저장)의")
    out("   값이 그 차이이고, 이번에는 (A) + (D) 로 청구서가 나왔다.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
