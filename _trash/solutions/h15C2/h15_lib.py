"""H15 제출 공용 라이브러리 — 텍스트만, GPU 0, 생성 없음. 사전등록 2026-10-01 (지휘 #49).

점수 세 가지 (모두 **높을수록 robust**)
  A  score = −n_sym                      n_sym = edit_rename 이 바꾸는 문자 수 (수식 안 기호·고유명사)
  B  score = −(n_sym − (a·log단어수 + b)) 길이 잔차화. a·b 는 공식 train 33문제에서 **고정**(아티팩트)
  C  score = −log(단어수)                 대조군. 리더보드에서 '기호 효과'와 '길이 효과'를 가르기 위한 것

판정: score >= 임계값 이면 True(robust). 임계값은 **train 두 클래스 중앙값의 중점**으로 사전 고정했다
      (train AUC 를 최대화하는 값이 아니다 — 리더보드·검증셋을 보고 고르지 않겠다는 뜻).
      배치 내 중앙값 분할은 쓰지 않는다(최종셋이 불균형일 수 있다는 9/28 답변).
모델별 차이 없음: 네 모델 모두 같은 규칙을 쓴다(라벨이 문제 단위라는 H2 관측에 따름).
미지 model_id 도 같은 규칙이라 KeyError 가 날 수 없다.
"""

import json
import math
import re
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts" / "h15.json"
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT = 15
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}


def _rename(text):
    """experiments/variants_official.edit_rename 과 같은 치환 (수식 span 안, LaTeX 명령어는 보호)."""
    def chunk(c):
        c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + UPPER_SHIFT) % 26 + 65), c)
        return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: LOWER_PAIRS.get(m.group(1), m.group(1)), c)

    def span(m):
        s, out, pos = m.group(0), [], 0
        for cmd in LATEX_CMD.finditer(s):
            out.append(chunk(s[pos:cmd.start()])); out.append(cmd.group(0)); pos = cmd.end()
        out.append(chunk(s[pos:]))
        return "".join(out)
    return MATH_SPAN.sub(span, text)


def n_sym(text):
    rn = _rename(text)
    return sum(1 for a, b in zip(text, rn) if a != b) if len(rn) == len(text) else 0


def log_words(text):
    return math.log(max(1, len(text.split())))


def load_art():
    return json.loads(ART.read_text(encoding="utf-8"))


def scores(problems, mode):
    """mode in {A, B, C}. 반환: 문제별 점수 (높을수록 robust)."""
    art = load_art()
    out = []
    for p in problems:
        ns, lw = n_sym(p), log_words(p)
        if mode == "A":
            out.append(-float(ns))
        elif mode == "B":
            out.append(-float(ns - (art["slope"] * lw + art["intercept"])))
        else:
            out.append(-float(lw))
    return out


def decide(problems, mode):
    thr = load_art()["thr"][mode]
    return [bool(s >= thr) for s in scores(problems, mode)]
