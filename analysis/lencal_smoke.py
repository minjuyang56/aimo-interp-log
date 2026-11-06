"""#LEN-CAL 제출물 스모크 — 사전등록이 요구한 검증 6개를 **전부 숫자로** 찍는다.

1. 점수식이 h15C 와 같은 값인가 (최대절대차 0)
2. train 33문제에서 True 가 정확히 20개인가 (교정 목표 달성)
3. 미지 `model_id`·빈 문자열·문제 1개·긴 문제에서 **순수 bool** 인가
4. 옛 2-인자 호출(`are_robust(model_id, problems)`)도 받는가
5. 예외 경로가 상수 True 로 떨어지는가
6. **실측 시간** — 최종셋형(4모델 × 40문제) 등 규모별 벽시계

`analysis/smoke_submission.py` 는 채점기와 같은 분기로 따로 부른다(여기서는 중복하지 않는다).

사용: .venv/Scripts/python.exe analysis/lencal_smoke.py
"""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOL = ROOT / "solutions/lencal"
sys.path.insert(0, str(SOL))
sys.path.insert(0, str(ROOT / "experiments"))
REPORT = ROOT / "experiments/results/lencal_smoke.txt"
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    sol = importlib.import_module("solution")
    lib = importlib.import_module("lencal_lib")
    from h21_uncertainty import problems
    rows = problems()
    txt = [r["problem"] for r in rows]
    y = np.array([r["y"] for r in rows])
    ok = True

    art = lib.load_art()
    out("#LEN-CAL 스모크")
    out(f"   아티팩트 임계 {art['thr']:+.6f} · 목표 True 비율 {art['target_true_rate']:.4f} · "
        f"달성 {art['achieved_true_rate']:.4f} · 동점 {art['tie_at_boundary']}")
    out()

    # 1. 점수식 동일성
    sys.path.insert(0, str(ROOT / "solutions/h15C"))
    import h15_lib
    mad = float(np.abs(np.array(lib.scores(txt)) - np.array(h15_lib.scores(txt, "C"))).max())
    g = mad == 0.0
    ok &= g
    out(f"1. 점수식 == h15C : 최대절대차 {mad:.3e}  {'OK' if g else '**FAIL**'}")

    # 2. 교정 목표
    pred = sol.are_robust("Skywork/Skywork-OR1-Math-7B", "default", txt)
    nt = sum(pred)
    g = nt == art["k"] and len(pred) == len(txt)
    ok &= g
    out(f"2. train {len(txt)}문제 True {nt}개 (목표 {art['k']}) · 정확도 "
        f"{float((np.array(pred) == (y == 1)).mean()):.4f}  {'OK' if g else '**FAIL**'}")

    # 3. 경계 입력
    cases = [
        ("미지 model_id", "some/unknown-model", ["Find $x$ if $x+2=5$."]),
        ("빈 model_id", "", ["Find $x$ if $x+2=5$."]),
        ("문제 1개(긴)", "allenai/Olmo-3-7B-Think", [" ".join(["word"] * 400)]),
        ("문제 1개(짧)", "Qwen/Qwen3.5-4B", ["Compute 2+2."]),
        ("빈 리스트", "Qwen/Qwen3.5-4B", []),
        ("공백만", "Qwen/Qwen3.5-4B", ["   "]),
    ]
    for nm, mid, ps in cases:
        r = sol.are_robust(mid, "default", ps)
        g = isinstance(r, list) and len(r) == len(ps) and all(type(x) is bool for x in r)
        ok &= g
        out(f"3. {nm:<16} → {r if len(r) <= 3 else f'{len(r)}개'}  순수bool {g}  "
            f"{'OK' if g else '**FAIL**'}")

    # 4. 옛 2-인자 호출
    r2 = sol.are_robust("Skywork/Skywork-OR1-Math-7B", txt[:5])
    r3 = sol.are_robust("Skywork/Skywork-OR1-Math-7B", "default", txt[:5])
    g = r2 == r3 and all(type(x) is bool for x in r2)
    ok &= g
    out(f"4. 옛 2-인자 호출 == 3-인자 : {g}  {'OK' if g else '**FAIL**'}")

    # 5. 예외 경로
    bad = sol.are_robust("m", "default", [None])          # log_words(None) → AttributeError
    g = bad == [sol.FALLBACK]
    ok &= g
    out(f"5. 예외 → 폴백 {bad} (기대 [{sol.FALLBACK}])  {'OK' if g else '**FAIL**'}")

    # 6. 실측 시간
    out()
    out("6. 실측 벽시계 (모델 로드 0 · 생성 0 → 문자열 처리뿐)")
    base = txt * 2
    for label, nmodel, nprob in [("리더보드형 Small (5,3,3,3)", 4, None),
                                 ("최종셋형 4모델×35문제", 4, 35),
                                 ("최종셋형 4모델×40문제", 4, 40),
                                 ("Main 5모델×21문제", 5, 21),
                                 ("8모델×40문제 (논문 규모)", 8, 40)]:
        counts = [5, 3, 3, 3] if nprob is None else [nprob] * nmodel
        t0 = time.perf_counter()
        tot = 0
        for c in counts:
            tot += len(sol.are_robust("Skywork/Skywork-OR1-Math-7B", "default", base[:c]))
        dt = time.perf_counter() - t0
        out(f"   {label:<28} 인스턴스 {tot:>4} · {dt * 1000:8.2f} ms · "
            f"3600s 의 {dt / 3600 * 100:.5f}%")

    out()
    out("## " + ("**전부 통과**" if ok else "**실패 — 제출물이 아니다**"))
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
