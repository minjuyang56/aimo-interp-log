"""비공식 변형 3종 — 정답 불변, 주최측 공식 변형 목록에 없는 유형만. (지휘 #28, 사전등록 2026-09-29)

규칙 근거: 금지되는 것은 "최종 검증셋과 **같은 변형 유형**의 추가 라벨 데이터". 공식 목록은
  rephrase · rename · domain · distract · typos · expert_* 이다.
여기 세 변형은 그 어디에도 속하지 않는다:
  (a) reorder  조건 문장들의 **순서**만 바꾼다. 단어·이름·소재·군더더기 모두 그대로 → rephrase/rename/domain/distract/typos 아님.
  (b) notation **표기**만 바꾼다(숫자 12 → twelve, \\frac{a}{b} → a/b, \\% → percent). 문장 구조·어휘 불변 → rephrase 아님.
  (c) format   문제는 손대지 않고 **답 형식 지시문**만 바꾼다(ANSWER: → FINAL =). 문제 텍스트 변형이 아예 아님.

정답 보존 안전장치 (1차 검사에서 실패한 두 가지를 고친 것)
  * reorder는 **절차/참조 문장**이 하나라도 있으면 포기한다. "합을 7에 더한다 → 총합을 4로 나눈다" 처럼 순서가 곧 계산이면
    순서를 바꾸는 순간 답이 바뀐다. 참조어(the total, the result, second, then, remaining…)·명령형·질문형을 전부 차단.
  * notation은 \\frac{1}{2} 안의 숫자를 단어로 바꾸지 않는다(그러면 "one/two"가 되어 값이 깨진다). 보호 구간을 먼저 잡고 변환한다.

사용:  .venv/Scripts/python.exe experiments/variants_nonofficial.py      (자체 검사 출력)
"""

import re

MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
SENT_END = re.compile(r"(?<=[.!?])\s+")
# 순서를 바꾸면 뜻이 변하는 신호: 앞 문장의 결과를 가리키거나 절차를 잇는 표현
UNSAFE = re.compile(r"\b(the total|the result|the sum of (these|those|them)|this (number|value|result|sum|total)|these|those|"
                    r"first|second|third|next|then|after (that|this)|afterwards|finally|previous|above|remaining|it is|they are|"
                    r"thus|therefore|hence|so that|resulting)\b", re.I)
IMPERATIVE = re.compile(r"^(find|compute|what|how|determine|calculate|evaluate|let|suppose|express|give|state|write|if)\b", re.I)
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def _protected(text):
    """수식 span + \\frac{...}{...} 구간 (여기 숫자는 건드리지 않는다)."""
    out = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    out += [(m.start(), m.end()) for m in FRAC.finditer(text)]
    return out


def _inside(i, spans):
    return any(a <= i < b for a, b in spans)


def num_to_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return TENS[t * 10] + ("-" + ONES[r] if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return ONES[h] + " hundred" + (" " + num_to_words(r) if r else "")
    return None


def sentences(text):
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    parts, last = [], 0
    for m in SENT_END.finditer(text):
        if _inside(m.start(), spans):
            continue
        parts.append(text[last:m.start()].rstrip()); last = m.end()
    parts.append(text[last:])
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------- (a) 조건 문장 순서 재배열
def reorder(text):
    """마지막 문장(질문)은 고정, 앞의 조건 문장 순서를 뒤집는다.
    문장 4개 이상 + 조건 문장에 참조어·명령형·질문이 하나도 없을 때만. 아니면 None."""
    parts = sentences(text)
    if len(parts) < 4:
        return None
    body, tail = parts[:-1], parts[-1]
    for p in body:
        if UNSAFE.search(p) or IMPERATIVE.match(p.strip()) or "?" in p:
            return None
    new = " ".join(list(reversed(body)) + [tail])
    return new if new != text else None


# ---------------------------------------------------------------- (b) 표기 변환
def notation(text):
    """수식 span·\\frac 안을 뺀 1~999 정수를 영어 단어로, \\frac{a}{b} → a/b, \\% → percent. 바뀐 게 없으면 None."""
    spans = _protected(text)
    res, pos, changed = [], 0, False
    for m in re.finditer(r"(?<![\w.$\\/])(\d{1,3})(?![\w.$/])", text):
        if _inside(m.start(), spans):
            continue
        w = num_to_words(int(m.group(1)))
        if w is None:
            continue
        res.append(text[pos:m.start()]); res.append(w); pos = m.end(); changed = True
    res.append(text[pos:])
    out = "".join(res)
    out2 = FRAC.sub(lambda m: f"{m.group(1)}/{m.group(2)}", out).replace("\\%", " percent")
    changed = changed or out2 != out
    return out2 if changed and out2 != text else None


# ---------------------------------------------------------------- (c) 답 형식 지시문
SOLVE_PROMPT = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer "
                "on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN "
                "if the given problem does not have a solution.")
SOLVE_PROMPT_ALT = ("You are a precise math problem solver. Work through the given problem step by step, then write your final "
                    "answer on the very last line using exactly this form:\nFINAL = <your answer>\nThe value should be an integer, "
                    "or NaN when the problem has no solution.")
SOLVE_PROMPT_ALT2 = ("Solve the following mathematics problem carefully, showing your reasoning. When you are done, end your reply "
                     "with one line in precisely this shape:\nRESULT -> <your answer>\nUse an integer, or NaN if no solution exists.")
ANSWER_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+)", re.I)
ANSWER_ALT_RE = re.compile(r"(?:FINAL\s*=|RESULT\s*->)\s*\$?(-?[\d,]+)", re.I)


def extract_answer(text, alt=False):
    hits = (ANSWER_ALT_RE if alt else ANSWER_RE).findall(text or "")
    if not hits:
        hits = re.findall(r"\\boxed\{\s*(-?[\d,]+)\s*\}", text or "")
    if not hits:
        return None
    try:
        return int(hits[-1].replace(",", ""))
    except ValueError:
        return None


def variants_for(text):
    """문제당 변형 **정확히 2개**. reorder·notation 중 되는 것을 먼저 쓰고, 모자라면 format(c)·format2로 채운다.
    반환: [(kind, problem_text, alt_prompt)] — alt_prompt True면 답 형식 지시문을 바꾼 것(format2는 RESULT -> 형식)."""
    out = []
    for kind, fn in (("reorder", reorder), ("notation", notation)):
        v = fn(text)
        if v:
            out.append((kind, v, False))
    for kind in ("format", "format2"):
        if len(out) >= 2:
            break
        out.append((kind, text, True))
    return out[:2]


# ---------------------------------------------------------------- 자체 검사
if __name__ == "__main__":
    samples = [
        ("바뀌면 안 되는 절차형", "The sum of \\frac{1}{2} and \\frac{1}{3} is added to 7. A second number equals 20. The total is divided by 4. What is the result?"),
        ("안전한 조건형", "Alice has 12 apples. Bob has three times as many apples as Alice. Carol has $5$ fewer apples than Bob. How many apples does Carol have?"),
        ("짧은 문제", "Let $f(x) = x^2 + 3x$. Find $f(4)$."),
        ("분수 보호", "A recipe uses \\frac{3}{4} cup of sugar and 25 grams of salt. The baker triples the recipe. How much sugar is used?"),
    ]
    for tag, s in samples:
        print(f"[{tag}] {s}")
        print("  reorder :", reorder(s) or "(적용 불가)")
        print("  notation:", notation(s) or "(적용 불가)")
        print("  → 채택 :", [(k, t[:60] + ("…" if len(t) > 60 else ""), alt) for k, t, alt in variants_for(s)])
        print()
    assert reorder(samples[0][1]) is None, "절차형은 반드시 거부해야 한다"
    assert "one/two" not in (notation(samples[0][1]) or ""), "\\frac 안 숫자는 단어로 바꾸면 안 된다"
    assert notation(samples[3][1]).count("3/4") == 1 and "twenty-five" in notation(samples[3][1])
    print("답 추출:", extract_answer("blah\nANSWER: -12\n"), extract_answer("FINAL = 1,234", alt=True), extract_answer("so \\boxed{42}"))
    print("자체 검사 통과")
