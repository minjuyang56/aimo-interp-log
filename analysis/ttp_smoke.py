"""#TT-PERTURB 로컬 스모크 — GPU 0. **주장하지 않고 계측한다.**

검사 7개:
1. 섭동 앞 6종이 `solutions/h22/h22_lib.rule_perturbations` 와 **전 문제에서 글자 그대로 같은가**
   (베껴 적어 어긋난 사고가 이 저장소에 여러 번 있었다)
2. 섭동 결정성 — 두 번 불러 같은가
3. **정답 불변 안전검사** — 섭동이 수식 안 숫자 다중집합을 보존하는가 (깨지면 '흔들렸다' 의 정의가 무너진다)
4. 섭동 개수 분포 (K 가 조용히 줄면 분해능이 떨어진다)
5. `score_one` 경계 케이스
6. h15C 폴백이 h15C 제출물과 같은 판정인가
7. `are_robust` 경계 입력 — 미지 model_id·빈 문자열·빈 리스트에서 순수 bool (모델 로드 없이)

사용: .venv/Scripts/python.exe analysis/ttp_smoke.py
"""

import importlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solutions/ttp"))
sys.path.insert(0, str(ROOT / "solutions/h22"))
sys.path.insert(0, str(ROOT / "solutions/h15C"))
sys.path.insert(0, str(ROOT / "experiments"))
REPORT = ROOT / "experiments/results/ttp_smoke.txt"
L = []
NUM_IN_MATH = __import__("re").compile(r"\d+")


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    ttp = importlib.import_module("ttp_lib")
    h22 = importlib.import_module("h22_lib")
    sol = importlib.import_module("solution")
    from h21_uncertainty import problems
    rows = problems()
    txt = [r["problem"] for r in rows]
    ok = True
    out(f"#TT-PERTURB 스모크 — 공식 train-main-v2 문제단위 {len(txt)}문제로 검사")
    out()

    # 1. h22 와의 동일성 (앞 6종)
    bad = []
    for t in txt:
        a = [(k, v) for k, v in ttp.perturbations(t) if not k.startswith(("reorder", "notation"))]
        b = h22.rule_perturbations(t)
        if a != b:
            bad.append(t[:60])
    g = not bad
    ok &= g
    out(f"1. 앞 6종 == h22.rule_perturbations : 불일치 {len(bad)}/{len(txt)}  {'OK' if g else '**FAIL**'}")
    if bad:
        out(f"   예: {bad[:2]}")

    # 2. 결정성
    g = all(ttp.perturbations(t) == ttp.perturbations(t) for t in txt)
    ok &= g
    out(f"2. 섭동 결정성 (두 번 호출 동일) : {g}  {'OK' if g else '**FAIL**'}")

    # 3. 정답 불변 — 수식 구간 안 숫자 다중집합 보존
    #    (distract 는 수식 밖에 문장을 붙이므로 수식 안 숫자는 그대로여야 하고,
    #     notation 은 수식 **밖** 숫자만 단어로 바꾸므로 역시 수식 안은 그대로여야 한다)
    def math_nums(s):
        return Counter(n for m in ttp.MATH_SPAN.finditer(s) for n in NUM_IN_MATH.findall(m.group(0)))
    viol = []
    for t in txt:
        base = math_nums(t)
        for k, v in ttp.perturbations(t):
            if math_nums(v) != base:
                viol.append((k, t[:40]))
    g = not viol
    ok &= g
    out(f"3. 수식 안 숫자 다중집합 보존 : 위반 {len(viol)}  {'OK' if g else '**FAIL**'}")
    if viol:
        out(f"   위반 유형 {Counter(k for k, _ in viol).most_common()}")

    # 4. 개수 분포
    cnt = Counter(len(ttp.perturbations(t)) for t in txt)
    kinds = Counter(k for t in txt for k, _ in ttp.perturbations(t))
    g = min(cnt) >= ttp.K_MIN
    ok &= g
    out(f"4. 섭동 개수 분포 {dict(sorted(cnt.items()))} · 최소 {min(cnt)} (K_MIN {ttp.K_MIN})  "
        f"{'OK' if g else '**FAIL**'}")
    out(f"   유형별 적용 횟수 {dict(kinds)}")

    # 5. score_one
    cases = [
        ("원본답 없음", (None, [1, 1, 1, 1, 1, 1]), (None, 0)),
        ("전부 일치", (7, [7, 7, 7, 7, 7, 7]), (0.0, 6)),
        ("절반 갈림", (7, [7, 7, 7, 1, 2, 3]), (0.5, 6)),
        ("전부 갈림", (7, [1, 2, 3, 4, 5, 6]), (1.0, 6)),
        ("쓸 수 있는 섭동 3개", (7, [7, 7, 7, None, None, None]), (None, 3)),
        ("쓸 수 있는 섭동 4개", (7, [7, 7, 7, 1, None, None]), (0.25, 4)),
    ]
    for nm, args, want in cases:
        got = ttp.score_one(*args)
        g = (got[0] is None and want[0] is None and got[1] == want[1]) or \
            (got[0] is not None and want[0] is not None and abs(got[0] - want[0]) < 1e-12 and got[1] == want[1])
        ok &= g
        out(f"5. score_one {nm:<18} → {got}  기대 {want}  {'OK' if g else '**FAIL**'}")

    # 6. h15C 폴백 복제
    import h15_lib
    a = ttp.h15c_decide(txt)
    b = [bool(s >= h15_lib.load_art()["thr"]["C"]) for s in h15_lib.scores(txt, "C")]
    g = a == b
    ok &= g
    out(f"6. h15C 폴백 복제 : {sum(1 for x, y in zip(a, b) if x == y)}/{len(txt)} 일치  "
        f"{'OK' if g else '**FAIL**'}")

    # 7. 경계 입력 (모델 로드 없음 → 폴백 경로)
    for nm, mid, ps in [("미지 model_id", "some/unknown", ["Find $x$ if $x+2=5$."]),
                        ("빈 model_id", "", ["Find $x$ if $x+2=5$."]),
                        ("빈 리스트", "Qwen/Qwen3.5-4B", []),
                        ("공백만", "", ["   "])]:
        r = sol.are_robust(mid, "default", ps)
        g = isinstance(r, list) and len(r) == len(ps) and all(type(x) is bool for x in r)
        ok &= g
        out(f"7. {nm:<14} → {r}  순수bool {g}  {'OK' if g else '**FAIL**'}")
    r2 = sol.are_robust("", txt[:4])                     # 옛 2-인자
    r3 = sol.are_robust("", "default", txt[:4])
    g = r2 == r3
    ok &= g
    out(f"7. 옛 2-인자 == 3-인자 : {g}  {'OK' if g else '**FAIL**'}")

    out()
    out("## " + ("**전부 통과**" if ok else "**실패 — 제출물이 아니다**"))
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
