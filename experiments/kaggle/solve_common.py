"""커널들이 공유하는 풀이 프롬프트와 답 추출. (2026-10-05, H20-M 재push 때 분리)

왜 분리하나: `genlab_h14.py` 는 `SOLVE_PROMPT` · `extract_answer` 를 **다른 파일에서 inline 되어 들어온다고 가정**하고
`# noqa: F821` 로 넘겨 놨다. H20-M 1차 push 에서 `--body genlab_h14.py` 만 줬다가
`NameError: name 'edit_rename' is not defined` 로 vLLM 로드까지 끝낸 뒤 죽었다(모델당 4분 낭비).
가정을 파일로 고정해 두면 `--body genlab_h14.py,solve_common.py,../variants_official.py` 로 항상 같이 올라간다.

정의는 `experiments/gen_labels.py` 의 것과 **같다**(라벨 생성 때 쓴 것과 답 추출 규칙을 다르게 두면 비교가 깨진다).
"""

import re as _re

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
ANSWER_RE = _re.compile(r"ANSWER:\s*([^\n<]+)")
BOXED_RE = _re.compile(r"\\boxed\{([^{}]*)\}")
THINK_RE = _re.compile(r"<think>.*?</think>", _re.S)


def extract_answer(text):
    """ANSWER: 가 있으면 마지막 것, 없으면 \\boxed{} 의 마지막 것. 없으면 None."""
    text = THINK_RE.sub("", text)
    m = ANSWER_RE.findall(text)
    hit = [m[-1]] if m else BOXED_RE.findall(text)
    if not hit:
        return None
    a = hit[-1].strip().rstrip(".").replace(",", "").replace("$", "").strip()
    return a or None
