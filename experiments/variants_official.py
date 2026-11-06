"""공식 유형 변형 3종 (rename · typos · distract) — 대회 정의와 같은 유형. (지휘 #29, 2026-09-29)

왜 이제 쓰나: 공식 규칙 Data 절 [Clarification 19 Sep] 이 "추가 라벨 데이터 금지" 규칙을 **폐지**했다.
금지는 평가 데이터 유출뿐이므로, 대회가 실제로 쓰는 변형 유형으로 라벨을 만드는 게 정의상 가장 가깝다.
`experiments/gen_labels.py` 의 구현을 그대로 옮긴 것(시드 고정, 모델 없음). 셋 다 정답 불변.

  rename    수식 안의 기호만 치환(대문자 15칸 시프트 · 소문자 짝 교환). LaTeX 명령어는 보호.
  typos     수식 밖 5글자 이상 단어 k개에서 인접 두 글자를 바꿈.
  distract  문제 끝에 정답과 무관한 한 문장을 덧붙임.
"""

import re

MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
DISTRACTORS = [
    "Note that the problem was originally posed at a regional competition in 2019.",
    "A friend of the author suggested this problem after a long hike.",
    "The committee that wrote this problem also enjoys chess on weekends.",
]


def edit_rename(text, rng=None):
    def chunk(c):
        c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + UPPER_SHIFT) % 26 + 65), c)
        return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: LOWER_PAIRS.get(m.group(1), m.group(1)), c)

    def span(m):
        s, out, pos = m.group(0), [], 0
        for cmd in LATEX_CMD.finditer(s):
            out.append(chunk(s[pos:cmd.start()])); out.append(cmd.group(0)); pos = cmd.end()
        out.append(chunk(s[pos:]))
        return "".join(out)
    new = MATH_SPAN.sub(span, text)
    return new if new != text else None


def edit_typos(text, rng, k=3):
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    words = [m for m in re.finditer(r"[A-Za-z]{5,}", text) if not any(a <= m.start() < b for a, b in spans)]
    if not words:
        return None
    out = list(text)
    for i in sorted(rng.choice(len(words), size=min(k, len(words)), replace=False), reverse=True):
        m = words[i]
        j = m.start() + 1 + int(rng.integers(m.end() - m.start() - 3))
        out[j], out[j + 1] = out[j + 1], out[j]
    new = "".join(out)
    return new if new != text else None


def edit_distract(text, rng):
    return text.rstrip() + " " + DISTRACTORS[int(rng.integers(len(DISTRACTORS)))]


# --- rename 두 번째 변형 (지휘 #43): 다른 시프트·다른 짝. rename 과 겹치지 않는 기호 배치를 쓴다.
UPPER_SHIFT2 = 7
LOWER_PAIRS2 = {"x": "s", "s": "x", "y": "v", "v": "y", "z": "q", "q": "z", "n": "h", "h": "n",
                "k": "g", "g": "k", "a": "e", "e": "a", "b": "d", "d": "b", "c": "f", "f": "c"}


def edit_rename2(text, rng=None):
    def chunk(c):
        c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + UPPER_SHIFT2) % 26 + 65), c)
        return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: LOWER_PAIRS2.get(m.group(1), m.group(1)), c)

    def span(m):
        t, out, pos = m.group(0), [], 0
        for cmd in LATEX_CMD.finditer(t):
            out.append(chunk(t[pos:cmd.start()])); out.append(cmd.group(0)); pos = cmd.end()
        out.append(chunk(t[pos:]))
        return "".join(out)
    new = MATH_SPAN.sub(span, text)
    return new if new != text else None


OFFICIAL = {"rename": edit_rename, "typos": edit_typos, "distract": edit_distract}


def official_variants(text, seed=0, kinds=None):
    """반환: [(kind, new_text)] — 적용 가능한 것만. rng는 문제마다 seed 고정."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for k in (kinds or OFFICIAL):
        v = OFFICIAL[k](text, rng)
        if v and v != text:
            out.append((k, v))
    return out


if __name__ == "__main__":
    s = "Alice has 12 apples and $x + y = 7$. Bob computes the remainder when $N$ is divided by $9$. How many apples remain?"
    for k, v in official_variants(s):
        print(f"{k:9s}: {v}")
