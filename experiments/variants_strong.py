"""센 섭동 — Domain · Rephrase. 재작성기 모델로 만들고 **2단계로 정답 보존을 검증한다**. (지휘 #STRONG)

왜 필요한가 (측정된 근거)
  주최측은 과반 규칙으로 `train-main-v2` 에서 spurious **27%**(19/71)를 얻는데, 우리 3유형
  (rename·typos·distract)으로는 **0~15%** 다. k=8 Skywork 에서는 '3유형 중 2개 이상 깨진 문제' 가
  **0건**이어서 과반 규칙이 아예 적용 불가였다. 즉 **규칙이 아니라 섭동이 약하다.**
  논문의 섭동은 6유형(Rephrase · Rename · Domain · Distract · Typos · Expert)이고, 특히 **Domain** 이
  빠져 있다 — MATH 137문제에서 **짧은 문제 50/52 를 무너뜨린** 그 유형이다(CLAUDE.md 라벨 길이 의존 절).

설계 (지휘 #STRONG)
  * **Domain ×2**: 수학 구조·수치·정답은 그대로, **이야기/맥락만 다른 분야로** 다시 쓴다.
  * **Rephrase ×1**: 문장 구조를 전면 재작성한다(수치·기호는 그대로).
  → 기존 rename·typos·distract 와 합쳐 **6유형**(논문과 같은 수).
  * 재작성기는 **피험 모델이 아닌 더 큰 instruct 모델**을 쓴다. 피험 모델이 자기 섭동을 만들면
    그 모델에 유리하게 편향될 수 있고, 모델마다 다른 섭동이 생겨 모델 간 비교가 깨진다.
    기본값 `Qwen/Qwen2.5-32B-Instruct` — 공식 모델 카드에서 **`license: apache-2.0`** 확인(2026-10-10).
  * **재작성은 한 번만 하고 세 모델이 같은 섭동을 쓴다.** 모델별로 다시 만들면 비용이 3배이고
    (더 중요하게) 모델 간 비교가 깨진다.
  * temperature 0 — 재현돼야 한다.

정답 보존 2단계 검증 (지휘 #STRONG-3)
  1. **숫자 multiset 보존**: 원본과 재작성본의 숫자 다중집합이 같아야 한다. 하나라도 다르면 버린다.
  2. **재작성기 자기 판정**: 같은 모델에게 두 문제를 보여주고 "수학적으로 같은 문제이고 정답이 같은가"
     를 묻는다. **YES 만 채택.** 순서를 바꿔 두 번 묻고 **둘 다 YES 여야** 통과한다
     (한 방향만 물으면 모델이 기계적으로 YES 를 내는 쪽으로 쏠린다).
  탈락률을 반드시 보고한다 — **탈락률이 0 에 가까우면 검증이 무력하다는 뜻**이므로 그 사실을 적는다.

사용: 이 파일은 프롬프트와 검증 규칙만 담는다(모델 없음). 실행은
      `experiments/runpod/jobs/strongvar.py` 가 vLLM 으로 한다.
"""

import re
from collections import Counter

NUM_RE = re.compile(r"\d+")

#   Domain 재작성 v2 (지휘 #DOMAIN-FIX) — v1 은 채택 2/10 이었고 탈락 기제가 명확했다:
#   수식을 이야기로 **풀어쓰다 숫자 다중집합이 깨진다**(1단계 탈락 6/8).
#   그래서 v2 는 **수식 블록을 글자 그대로 복사하라고 강제**하고 few-shot 1개를 붙인다.
#   수식 밖 서술만 다른 분야로 바꾸게 하면 정답 보존과 분야 전환이 양립할 수 있다.
_FEWSHOT = (
    "Example.\n"
    "ORIGINAL: A bakery has $12$ trays and the number of pastries $p$ satisfies $3p + 5 = 41$. "
    "How many pastries are on each tray?\n"
    "GOOD REWRITE: A shipping depot has $12$ pallets and the number of parcels $p$ satisfies "
    "$3p + 5 = 41$. How many parcels are on each pallet?\n"
    "Notice: every LaTeX expression ($12$, $p$, $3p + 5 = 41$) is copied character for character; "
    "only the nouns outside the math changed.\n\n"
)
_DOMAIN_RULES = (
    "You rewrite mathematics competition problems so that the SETTING comes from a different "
    "real-world domain, while the mathematics stays identical.\n"
    "HARD RULES - breaking any of them makes the rewrite useless:\n"
    "1. Copy EVERY LaTeX expression character for character. Anything between $...$ or \\[...\\] "
    "must appear in your output exactly as in the original, in the same order.\n"
    "2. Never turn a formula into words. Never introduce a new number anywhere.\n"
    "3. Change only the prose OUTSIDE the math: the objects, the people, the scenario.\n"
    "4. If the problem is purely symbolic and has almost no prose, change only the few words that "
    "name objects and leave everything else as is. A small change is better than a broken one.\n"
    "5. Do not solve the problem. Output only the rewritten problem as one paragraph.\n\n"
)
DOMAIN_PROMPTS = [
    ("domain0", _DOMAIN_RULES + _FEWSHOT
     + "Use a logistics / warehouse / shipping setting.\n\nORIGINAL: "),
    ("domain1", _DOMAIN_RULES + _FEWSHOT
     + "Use a laboratory / scientific measurement setting (different from logistics).\n\nORIGINAL: "),
]
REPHRASE_PROMPTS = [
    ("rephrase0",
     "You rewrite mathematics competition problems. Rewrite the problem below changing the sentence "
     "structure completely: reorder the given information, change passive to active voice, split or "
     "merge sentences. Keep the same domain and story. Every number and every mathematical "
     "expression must appear unchanged and the answer must stay the same. Do not solve it. "
     "Output only the rewritten problem as one paragraph.\n\n"),
]
STRONG_PROMPTS = DOMAIN_PROMPTS + REPHRASE_PROMPTS

#   자기 판정 — 짧은 답만 받는다. 이유를 쓰게 하면 길어지고 판정이 흐려진다.
JUDGE_PROMPT = (
    "You are checking whether two mathematics problems are equivalent.\n"
    "Answer with exactly one word: YES if solving problem B gives the SAME numerical answer as "
    "solving problem A and the mathematical content is the same, or NO if the answer could differ "
    "in any way (changed quantities, changed relationship, missing information, ambiguity).\n"
    "Be strict: if you are not certain the answers are identical, answer NO.\n\n"
    "Problem A:\n{a}\n\nProblem B:\n{b}\n\nOne word (YES or NO):"
)


def numbers_preserved(a, b):
    """숫자 다중집합이 같은가. 1단계 검증."""
    return Counter(NUM_RE.findall(a)) == Counter(NUM_RE.findall(b))


def clean_rewrite(raw, original):
    """재작성 출력에서 문제 본문만 꺼낸다. 모델이 사족을 붙이는 경우가 있다."""
    t = (raw or "").strip()
    # 사고 블록을 뚫고 나온 경우 잘라낸다
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()
    # 첫 한 단락만 (지시대로 한 단락이어야 한다)
    t = t.split("\n\n")[0].strip()
    # 흔한 머리말 제거
    for pref in ("Rewritten problem:", "Problem:", "Here is the rewritten problem:",
                 "Rewritten:", "Restated problem:"):
        if t.lower().startswith(pref.lower()):
            t = t[len(pref):].strip()
    return t


def judge_ok(ans):
    """판정 출력에서 YES 를 읽는다. 애매하면 **False**(엄격하게)."""
    t = (ans or "").strip().upper()
    if "</THINK>" in t:
        t = t.split("</THINK>", 1)[1].strip()
    first = re.sub(r"[^A-Z]", "", t[:20])
    return first.startswith("YES")


def acceptable(original, rewrite, judgements):
    """2단계를 모두 통과했나. `judgements` 는 (A→B, B→A) 두 판정의 원문."""
    if not rewrite or len(rewrite) < 25 or rewrite.strip() == original.strip():
        return False, "빈/짧은/동일 출력"
    if not numbers_preserved(original, rewrite):
        return False, "숫자 다중집합 불일치"
    if not all(judge_ok(j) for j in judgements):
        return False, "재작성기 자기 판정 NO"
    return True, "통과"


if __name__ == "__main__":
    # 모델 없이 되는 자체 검사 — 규칙이 의도대로 거르는지
    o = "Alice has 12 apples and $x + y = 7$. How many remain after she gives away 4?"
    cases = [
        ("통과해야 함", "A warehouse holds 12 crates and $x + y = 7$. How many remain after 4 ship out?",
         ["YES", "yes"]),
        ("숫자 바뀜 → 탈락", "A warehouse holds 13 crates and $x + y = 7$. How many remain after 4 ship out?",
         ["YES", "YES"]),
        ("판정 NO → 탈락", "A warehouse holds 12 crates and $x + y = 7$. How many remain after 4 ship out?",
         ["YES", "NO"]),
        ("한쪽만 YES → 탈락", "A warehouse holds 12 crates and $x + y = 7$. How many remain after 4 ship out?",
         ["YES", "maybe"]),
        ("동일 출력 → 탈락", o, ["YES", "YES"]),
    ]
    for tag, rw, js in cases:
        ok, why = acceptable(o, rw, js)
        print(f"  {'OK  ' if (ok == tag.startswith('통과')) else 'FAIL'} {tag}: {ok} ({why})")
    print("  판정 파서:", [judge_ok(x) for x in ("YES", " yes.", "NO", "Yes, same", "I think NO", "")])
