"""결정적 상수 변형 K개 생성 (지휘 #69 사전등록). 난수 없음 — 같은 문제면 항상 같은 K개가 나온다.

왜 K개인가: h21c 는 변형 1개로 drop 을 쟀다. 주사위를 한 번 던진 값이다.
주최측 라벨은 **여러 변형에서의 정확도 하락**이므로 여러 번 던져 평균을 낸다.

두 규칙을 **따로** 만든다(섞지 않는다 — 효과를 분리해 보고해야 한다):
  rule="v1"  h21c 가 쓴 규칙의 확장. 숫자 후보에 **지수·첨자도 포함**한다($x^2+y^2$ → $x^2+y^3$ 가 생긴다).
  rule="v2"  개선 규칙. `^` 또는 `_` 뒤(중괄호 포함) 위치를 **후보에서 제외**해 지수·첨자를 건드리지 않는다.

변형 i 는 (위치, 증분) 격자에서 나온다: 위치는 등장 순서, 증분은 DELTAS 순서.
새 값이 2 미만이면 +1 로 대체한다(0/1 은 수식 구조를 바꾸기 쉽다).

자체검사:  .venv/Scripts/python.exe experiments/variants_constants.py
"""

import re

NUM_RE = re.compile(r"(?<![\w.])([2-9]|[1-9]\d{1,3})(?![\w.])")
DELTAS = (1, -1, 2, 3, -2, 4, 5, -3, 6, 7, -4, 8, 9, -5, 10, 11)


def _in_script(text, start):
    """그 숫자가 지수·첨자 안인가. 바로 앞을 거슬러 `^`/`_`(+공백/중괄호)를 찾는다."""
    i = start - 1
    while i >= 0 and text[i] in " \t{":
        i -= 1
    return i >= 0 and text[i] in "^_"


def sites(text, rule="v2"):
    """변형 가능한 숫자 위치(등장 순서). rule='v2' 면 지수·첨자 위치를 뺀다."""
    out = []
    for m in NUM_RE.finditer(text):
        if rule == "v2" and _in_script(text, m.start()):
            continue
        out.append(m)
    return out


def make_variants(text, k, rule="v2"):
    """결정적 변형 k개. 자리가 부족하면 같은 자리에 다른 증분을 쓴다. 중복·무변화는 버린다.
    반환: [(라벨, 변형문), ...] — 길이는 k 이하(만들 수 없으면 더 적다)."""
    ms = sites(text, rule)
    if not ms:
        return []
    out, seen = [], {text}
    for d_idx in range(len(DELTAS)):
        for s_idx, m in enumerate(ms):
            if len(out) >= k:
                return out
            v = int(m.group(1))
            nv = v + DELTAS[d_idx]
            if nv < 2:
                nv = v + 1
            cand = text[:m.start()] + str(nv) + text[m.end():]
            if cand in seen:
                continue
            seen.add(cand)
            out.append((f"{rule}:s{s_idx}d{DELTAS[d_idx]:+d}", cand))
    return out


if __name__ == "__main__":
    import difflib
    tests = [
        "Find the remainder when $7^{2024}$ is divided by 100.",
        "Let $x+y=12$ and $xy=35$. Compute $x^2+y^2$.",
        "A box holds 24 red and 36 blue balls. How many ways to pick 5?",
        "What is the value of $a$?",
    ]
    for t in tests:
        print("=" * 100)
        print("원문:", t)
        for rule in ("v1", "v2"):
            vs = make_variants(t, 4, rule)
            print(f"  {rule} 자리 {len(sites(t, rule))}개 · 변형 {len(vs)}개")
            for lab, v in vs:
                diff = [x for x in difflib.ndiff(t.split(), v.split()) if x.startswith(("+", "-"))]
                print(f"    {lab:14s} {' '.join(d.strip() for d in diff)[:80]}")
    print("=" * 100)
    print("결정성 확인:", all(make_variants(t, 8, "v2") == make_variants(t, 8, "v2") for t in tests))
    t = tests[1]
    print("v2 가 지수를 피하는가:", all("y^2" in v or "y^{2}" in v for _, v in make_variants(t, 4, "v2")))
