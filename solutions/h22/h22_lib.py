"""H22 — **주최측 라벨 정의를 테스트타임에 그대로 재현한다.** 사전등록 2026-10-10 (지휘 #H22).

무엇이 달라졌나 (왜 H21 계열을 떠나는가)
  논문(arXiv 2607.13899) 확인으로 라벨 정의가 우리가 쓰던 것과 **두 군데**에서 달랐다:
    (1) 판정은 **하락폭 임계값이 아니라 다수결**이다 — 원본을 반복 샘플링에서 안정적으로 맞히고,
        **섭동의 과반에서 틀리면 spurious**. (우리는 max_drop ≤0.10 / ≥0.25 로 역추정해 썼다.)
    (2) 섭동은 **상수만이 아니다** — Rephrase / Rename / Domain / Distract / Typos / Expert 혼합이다.
        H21c 는 상수 변형 1개(K=1)로만 쟀고, 그게 H21cK 에서 ±0.09 흔들리는 '운 좋은 주사위'였다.
  즉 H21c 의 재현 실패(자체 88문제 0.594, null 안)는 신호가 없어서가 아니라 **정의와 어긋나서**일 수 있다.
  리더보드 상위 4명이 동시에 86%(12/14)에 몰려 있는 것도 '정의를 재현한 사람들' 로 읽힌다.

점수 (사전등록, 부호 고정 — 결과 보고 뒤집지 않는다)
      score_spurious = P(원본 안정) × (섭동 과반 오답 비율)
  **높을수록 spurious.** 두 항을 곱하는 것이 정의 그대로다: 원본을 못 푸는 문제는 애초에 라벨 대상이 아니고
  (주최측은 그런 샘플을 **제외**했다), 원본을 푸는데 섭동에서 무너지는 것만 spurious 다.

  판정은 **절대 임계값 0.5**. 이건 이 프로젝트에서 처음으로 절대 임계값이 **정당한** 경우다 —
  점수가 임의 통계량이 아니라 **라벨 정의 자체의 확률**이라서 0.5 가 '과반' 과 정확히 같은 뜻이 된다.
  (호출 내 중앙값 분할을 쓰지 않는 이유: 최종셋이 균형이 아닐 수 있다는 9/28 공지. h20e·h21c 는
   중앙값 분할이었고 균형 가정에 기대고 있었다.)

테스트타임의 유일한 대체 — **의사정답(pseudo-gold)**
  실전에서는 정답을 모른다. 그래서 'P(원본 정답)' 을 그대로는 못 센다.
  원본 N 샘플의 **다수결 답**을 의사정답으로 쓰고, P(원본 안정) = 그 다수결에 동의한 샘플 비율로 읽는다.
  이건 **정확도가 아니라 자기일관성(self-consistency)** 이다. 정직하게 적어 둔다:
  모델이 일관되게 틀리는 문제에서는 이 항이 1.0 이 되어 과대평가된다. 다만 주최측 라벨도 '원본을
  안정적으로 맞히는 문제' 만 대상으로 하므로, 그런 문제는 애초에 라벨 모집단 밖일 가능성이 크다.

섭동은 **정답 불변 유형만** 쓴다 (설계 판단, 기록해 둔다)
  논문의 6유형(Rephrase·Rename·Domain·Distract·Typos·Expert)은 **전부 정답 보존 재서술**이다.
  그래서 의사정답이 섭동본에도 그대로 전이되고, '섭동 답 ≠ 의사정답' 이 곧 '무너졌다' 가 된다.
  **상수 변형은 primary 점수에서 뺀다**: 상수를 바꾸면 정답도 바뀌어야 하는데 올림피아드 문제의 새 정답을
  테스트타임에 계산할 방법이 없다(풀이기가 없다). 대신 H21c 가 검증한 방향으로 **보조항**만 남겼다 —
  상수를 바꿨는데 답이 **안 바뀌면** 이해가 아니라 암기 쪽 증거다(`H22_CONST_W`, 기본 0 = 꺼짐).

  K=8 의 구성: rename · rename2 · typos×2 · distract×2 · rephrase×2.
  rephrase 는 규칙으로 못 만들어서 **모델 자신에게 바꿔쓰게** 하고 **숫자 보존 검사**를 통과한 것만 쓴다
  (숫자 다중집합이 원본과 같아야 한다 — 하나라도 바뀌면 정답이 바뀌었을 수 있으므로 버린다).

예산 (이 설계의 생사가 걸린 자리 — `analysis/h22_budget.py`)
  * 공식 FAQ 8 해석(배치 전체 합산 3600s): **어떤 조합도 안 들어간다.** 최소형(N2·K2·S1)·cap 4096 조차
    전 인스턴스 합계 15,537s = 432% 다. h20 계열을 죽인 그 벽이고 이번엔 더 두껍다.
  * 논문 기술 해석(문제당 1시간 × 8 GPU): 완전형 N8·K8·S4·cap 16384 가 88% 로 들어간다.
  → 어느 쪽인지 확인되기 전까지 **예산 적응형**으로 만든다. 남은 시간을 재서 아래 사다리를 위에서부터
    내려오고, 최소형조차 못 사면 생성을 포기하고 **텍스트 규칙(n_sym)** 으로 떨어진다.
    그래서 (i) 이 맞아도 무효가 아니고, (ii) 가 맞으면 완전형으로 돈다.

미완주 처리
  상한에 걸려 답을 못 낸 샘플은 **오답으로 센다**(개정 #37 과 같은 규칙: 주최측 라벨도 정해진 예산 안의
  정확도이고, 답을 못 내놓은 실행은 결측이 아니라 실패다). 이게 cap 을 신호의 일부로 만들지만,
  그 성질은 주최측 라벨도 똑같이 갖고 있다.
"""

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

T0 = time.time()
SOFT = float(os.environ.get("H22_SOFT", 3300))
EXPECTED_CALLS = int(os.environ.get("H22_CALLS", 4))
RESERVE = float(os.environ.get("H22_RESERVE", 30))
THRESH = float(os.environ.get("H22_THRESH", 0.5))
NEUTRAL = 0.5                                             # 중립 확률 (판정은 텍스트 폴백으로 보낸다)
MIN_FIN_ORIG = int(os.environ.get("H22_MIN_FIN_ORIG", 2))  # 원본에서 이만큼 완주해야 과반을 센다
MIN_PERT = int(os.environ.get("H22_MIN_PERT", 3))          # 쓸 수 있는 섭동이 이만큼은 있어야 한다
CONST_W = float(os.environ.get("H22_CONST_W", 0.0))       # 상수 변형 보조항 가중치. 기본 0 = 꺼짐
REPHRASE_CAP = int(os.environ.get("H22_REPHRASE_CAP", 640))
# 사고 과정을 건너뛰고(`</think>`) **재서술 머리말을 미리 박아** 풀이로 직행하는 것을 막는다.
REPHRASE_FORCE = "</think>\nRestated problem: "
CACHE = Path(os.environ.get("H22_CACHE", "/tmp/h22_cache.json"))
STATE = Path(os.environ.get("H22_STATE", "/tmp/h22_state.json"))
FALLBACK = True

# (N 원본샘플, K 섭동수, S 섭동당샘플, cap) — 위에서부터 내려온다. cap 8192 미만은 측정이 포화해 무정보라
# 마지막 한 칸(4096)은 '그래도 아무것도 안 하는 것보다 낫다' 용이고 정상 경로가 아니다.
#   (N 원본샘플, K 섭동수, S 섭동당샘플, cap) — 위에서부터 내려오며 **처음 들어가는 칸**을 쓴다.
#   **cap 우선 정렬이다 (지휘 #H22-c)**: 자체 라벨에서 spurious 를 결정한 변형 실패의 69% 가 오답이 아니라
#   cap 16384 절단이었다. 절단은 문제의 성질이 아니라 예산이므로 cap 을 먼저 산다.
#   시간은 cap 에 **제곱**으로 드니(ms = a + b·B + c·B·L, L 이 cap 과 함께 자람) cap 을 24576 으로 올리면
#   같은 예산에서 K6·S2 까지만 들어간다. 그 교환을 받아들인 것이다 — K=6 은 논문의 6유형을 각 1개씩
#   덮으므로 '섭동 과반' 의 정의는 유지된다. 완전형(N8·K8·S4·cap 16384, 허용의 88%)을 쓰려면
#   H22_LADDER="8,8,4,16384" 로 덮어쓴다. 근거 표: analysis/h22_budget.py
LADDER = [
    (4, 6, 2, 24576),      # 기본: cap 최대 · 6유형 전부 · 최악 모델 6,116s (허용의 85%)
    (4, 6, 4, 16384),      # cap 내려가면 S 를 올린다 · 4,597s (64%)
    (4, 6, 2, 16384),
    (4, 4, 2, 16384),
    (2, 4, 2, 16384),
    (2, 2, 1, 16384),
    (2, 2, 1, 8192),       # 바닥. 8192 미만은 측정이 포화해 무정보라 내려가지 않는다
]
if os.environ.get("H22_LADDER"):
    # **스모크 전용 훅.** 로컬 8 GB 4bit 로는 cap 4096 조차 못 돌려서 생성 경로를 계측할 방법이 없다.
    # 형식 "N,K,S,cap[;N,K,S,cap…]". 제출 시에는 설정하지 않으므로 기본 사다리가 그대로 쓰인다.
    LADDER = [tuple(int(x) for x in rung.split(",")) for rung in os.environ["H22_LADDER"].split(";")]
A_, B_, C_, MULT, PROMPT = 15.0, 0.375, 7.21e-4, 1.15, 400.0      # 워커 실측 적합 (±0.6 ms, 4점)
KV_SAFE = float(os.environ.get("H22_KV_SAFE", 0.55))
EFFORT_MULT = {"low": 0.8, "default": 1.0, "medium": 1.0, "high": 1.1}

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
# **문제를 풀지 말라고 명시해야 한다.** 수학 추론 모델은 지시를 무시하고 풀이로 직행한다 —
# 로컬 실측에서 네 번 다 풀이를 냈고(숫자 보존 검사가 전부 걸러냈다) rephrase 가 0개였다.
# 그래서 (a) "Do NOT solve" 를 넣고 (b) 아래 REPHRASE_FORCE 로 **재서술 머리말을 미리 박는다**.
REPHRASE_PROMPTS = [
    "Rewrite the following mathematics problem in different words. Do NOT solve it. Do NOT explain "
    "anything. Do NOT give an answer. Output one single paragraph containing only the rewritten "
    "problem statement. You MUST keep every number and every mathematical expression exactly as it "
    "appears, so that the answer stays the same.\n\n",
    "Restate the following mathematics problem using more formal, examiner-style phrasing. Do NOT "
    "solve it. Do NOT explain anything. Do NOT give an answer. Output one single paragraph containing "
    "only the restated problem. Every number and every mathematical expression must appear unchanged, "
    "so that the answer stays the same.\n\n",
]

MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
UPPER_SHIFT, UPPER_SHIFT2 = 15, 7
LOWER_PAIRS = {"x": "t", "t": "x", "y": "u", "u": "y", "z": "w", "w": "z", "n": "m", "m": "n",
               "k": "j", "j": "k", "a": "p", "p": "a", "b": "q", "q": "b", "c": "r", "r": "c"}
LOWER_PAIRS2 = {"x": "s", "s": "x", "y": "v", "v": "y", "z": "q", "q": "z", "n": "h", "h": "n",
                "k": "g", "g": "k", "a": "e", "e": "a", "b": "d", "d": "b", "c": "f", "f": "c"}
DISTRACTORS = [
    "Note that the problem was originally posed at a regional competition in 2019.",
    "A friend of the author suggested this problem after a long hike.",
    "The committee that wrote this problem also enjoys chess on weekends.",
]
NUM_ALL = re.compile(r"\d+")
ANS_RE = re.compile(r"ANSWER\s*:\s*(-?\d+)")
BOXED_RE = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")
CONST_RE = re.compile(r"(?<![\w.])([2-9]|[1-9]\d{1,3})(?![\w.])")
_m = {"model": None, "tok": None, "id": None, "failed": set()}


def log(msg):
    print(f"[h22 {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def remaining():
    return SOFT - (time.time() - T0)


def jload(p, d):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return d


def jsave(p, o):
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(o), encoding="utf-8")
    except Exception:
        pass


def key_of(t, model_id="", effort=""):
    """캐시 키에 model_id·effort 를 넣는다 — 모델별 라벨이 다르므로 교차모델 재사용은 금지."""
    return hashlib.sha1(f"{model_id}|{effort}|{t.strip()}".encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 섭동 (정답 불변, 전부 결정적)

def _chunk(c, shift, pairs):
    c = re.sub(r"[A-Z]", lambda m: chr((ord(m.group(0)) - 65 + shift) % 26 + 65), c)
    return re.sub(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", lambda m: pairs.get(m.group(1), m.group(1)), c)


_ALONE = re.compile(r"(?<![A-Za-z])([a-z])(?![A-Za-z])")


def _safe_pairs(span_text, pairs):
    """이 수식 구간에서 **기호 충돌을 일으키지 않는** 치환만 남긴다.

    왜 필요한가 (2026-10-10, 자체 라벨에서 발견): 치환 규칙은 **홀로 선 한 글자**만 바꾼다.
    그런데 `\\frac{kc+l}{mc+n}` 에서 `mc` 의 `m` 은 두 글자로 붙어 있어 안 바뀌고, 홀로 선 `n` 은
    `m` 으로 바뀐다 → `mc+m` 이 되어 **서로 다른 두 기호가 한 기호로 뭉친다**. 정답이 보존되지 않는다.
    실제로 자체 라벨 88건 중 이런 문제가 있고, 그 변형이 var_acc 를 떨어뜨려 **spurious 라벨을 만들었다**.

    규칙: 홀로 선 글자 L 을 M 으로 바꾸는 것은, M 이 이 구간에 아예 없거나 M 도 홀로만 서서 함께
    바뀌어 나갈 때만 안전하다. (pairs 는 대합이라 L↔M 이 같이 바뀌면 단순 교환이고 안전하다.)
    `variants_official.py` 는 고치지 않는다 — 이미 생성한 라벨과의 비교 가능성이 깨진다."""
    alone = set(_ALONE.findall(span_text))
    any_l = set(re.findall(r"[a-z]", span_text))
    return {L: M for L, M in pairs.items()
            if L in alone and (M not in any_l or M in alone)}


def _rename(text, shift, pairs):
    """수식 안의 기호만 치환한다. LaTeX 명령어(\\frac 등)는 보호 — 안 하면 수식이 깨져 정답이 안 보존된다.
    치환표는 구간마다 `_safe_pairs` 로 걸러 기호 충돌을 막는다."""
    def span(m):
        s, out, pos = m.group(0), [], 0
        segs = []
        for cmd in LATEX_CMD.finditer(s):
            segs.append(s[pos:cmd.start()])
            pos = cmd.end()
        segs.append(s[pos:])
        p = _safe_pairs("".join(segs), pairs)       # 충돌 판정은 **구간 전체**를 보고 한다
        pos = 0
        for cmd in LATEX_CMD.finditer(s):
            out.append(_chunk(s[pos:cmd.start()], shift, p))
            out.append(cmd.group(0))
            pos = cmd.end()
        out.append(_chunk(s[pos:], shift, p))
        return "".join(out)
    new = MATH_SPAN.sub(span, text)
    return new if new != text else None


def _typos(text, seed, k=3):
    """수식 **밖** 5글자 이상 단어 k개에서 인접 두 글자를 바꾼다. 수식 안은 건드리면 정답이 바뀐다.
    난수 대신 seed 로 결정적으로 고른다(제출물은 재현돼야 한다)."""
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    words = [m for m in re.finditer(r"[A-Za-z]{5,}", text)
             if not any(a <= m.start() < b for a, b in spans)]
    if not words:
        return None
    pick = sorted({(seed * 7 + i * 3) % len(words) for i in range(min(k, len(words)))}, reverse=True)
    out = list(text)
    for i in pick:
        m = words[i]
        span_len = m.end() - m.start()
        j = m.start() + 1 + ((seed + i) % max(1, span_len - 3))
        if j + 1 < m.end():
            out[j], out[j + 1] = out[j + 1], out[j]
    new = "".join(out)
    return new if new != text else None


def _distract(text, idx):
    return text.rstrip() + " " + DISTRACTORS[idx % len(DISTRACTORS)]


def rule_perturbations(text):
    """규칙 기반 정답 불변 섭동 6종. (kind, text) 리스트. 적용 불가한 것은 빠진다."""
    cands = [
        ("rename", _rename(text, UPPER_SHIFT, LOWER_PAIRS)),
        ("rename2", _rename(text, UPPER_SHIFT2, LOWER_PAIRS2)),
        ("typos0", _typos(text, 0)),
        ("typos1", _typos(text, 1)),
        # **숫자가 없는 distractor 만 쓴다.** variants_official.py 의 첫 문장은 "…in 2019." 로 숫자를
        # 주입하는데, 연도가 답에 쓰이는 문제에서는 정답이 바뀔 수 있다. 섭동은 정답 불변이어야 한다.
        ("distract0", _distract(text, 1)),
        ("distract1", _distract(text, 2)),
    ]
    out, seen = [], {text}
    for k, v in cands:
        if v and v not in seen:
            out.append((k, v))
            seen.add(v)
    return out


def numbers_preserved(a, b):
    """숫자 다중집합이 같은가. rephrase 가 정답을 바꿨는지 거르는 유일한 검사다."""
    return Counter(NUM_ALL.findall(a)) == Counter(NUM_ALL.findall(b))


def const_perturbations(text, k=2):
    """상수 변형 — **primary 점수 밖**이다(새 정답을 테스트타임에 모른다). H21c 와 같은 규칙."""
    ms = list(CONST_RE.finditer(text))
    out, seen = [], {text}
    for d in (1, -1, 2, 3):
        for m in ms:
            if len(out) >= k:
                return out
            v = int(m.group(1))
            nv = v + d if v + d >= 2 else v + 1
            cand = text[:m.start()] + str(nv) + text[m.end():]
            if cand not in seen:
                out.append((f"const{len(out)}", cand))
                seen.add(cand)
    return out


# ---------------------------------------------------------------- 텍스트 폴백 (생성 못 살 때)

def n_sym(text):
    """edit_rename 이 실제로 바꾸는 문자 수 — H15 주지표(train AUC 0.760). 많을수록 spurious."""
    tot = 0
    for m in MATH_SPAN.finditer(text):
        s, pos = m.group(0), 0
        for cmd in LATEX_CMD.finditer(s):
            seg = s[pos:cmd.start()]
            tot += sum(1 for ch in seg if ch.isupper() or (ch.islower() and ch in LOWER_PAIRS))
            pos = cmd.end()
        seg = s[pos:]
        tot += sum(1 for ch in seg if ch.isupper() or (ch.islower() and ch in LOWER_PAIRS))
    return tot


def rank_split(values, higher_is_spurious=True):
    """호출 내 중앙값 분할 — **폴백 전용**이다. primary 는 절대 임계값 0.5 를 쓴다."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [True]
    order = sorted(range(n), key=lambda i: (-values[i] if higher_is_spurious else values[i], i))
    n_true = n - (n // 2)                       # 상위 절반을 spurious 로 = 하위 ceil(n/2) 가 True
    res = [True] * n
    for rank, i in enumerate(order):
        res[i] = rank >= (n - n_true)
    return res


# ---------------------------------------------------------------- 모델·예산

def kv_kb_per_token(model):
    """토큰당 KV KB. Olmo-3 는 MHA(kv헤드 32)라 Skywork 의 9배 — 배치 한도가 모델마다 다르다."""
    cfg = getattr(model, "config", None)
    t = getattr(cfg, "text_config", None) or cfg
    L = getattr(t, "num_hidden_layers", None)
    kv = getattr(t, "num_key_value_heads", None) or getattr(t, "num_attention_heads", None)
    hd = getattr(t, "head_dim", None)
    if not hd:
        hs, nh = getattr(t, "hidden_size", 0), getattr(t, "num_attention_heads", 0)
        hd = hs // nh if (hs and nh) else 0
    if not (L and kv and hd):
        return 512.0                            # 못 읽으면 가장 나쁜 쪽(MHA)으로 가정
    return 2 * L * kv * hd * 2 / 1024


def token_slots(model):
    """'배치 × cap' 으로 쓸 수 있는 토큰 칸 수. 측정 실패 시 None = 제한 없음."""
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return None
    return max(1, int(free / 1024 * KV_SAFE / kv_kb_per_token(model)))


def seconds_for(batch, cap, scale=1.0):
    lin = (A_ + B_ * batch + C_ * batch * PROMPT) * cap / 1000.0
    quad = C_ * batch * cap * cap / 2000.0
    return (lin + quad) * MULT * scale


def plan(n_problems, budget, slots, scale=1.0, effort="default"):
    """사다리를 위에서부터 내려와 **시간과 VRAM 을 둘 다** 만족하는 첫 칸을 고른다.
    반환 (N, K, S, cap, batch, est) 또는 None(= 생성 포기, 텍스트 폴백).

    쪼개는 단위는 **시퀀스**다. h20e 는 길이 순위를 재서 한 문제의 시퀀스가 같은 배치에 있어야 했지만
    (배치마다 끊기는 스텝이 달라 길이 비교가 깨졌다), H22 는 **답을 비교**하므로 배치 경계가
    비교 가능성을 건드리지 않는다. 그래서 VRAM 칸에 맞춰 자유롭게 자를 수 있다."""
    budget = budget * EFFORT_MULT.get(effort, 1.0)
    for N, K, S, cap in LADDER:
        total_seq = n_problems * (N + K * S)
        bmax = total_seq if not slots else max(1, min(total_seq, slots // cap))
        groups = max(1, -(-total_seq // bmax))
        batch = -(-total_seq // groups)                 # 그룹에 고르게 나눈 실제 배치 크기
        est = groups * seconds_for(batch, cap, scale)
        if est <= budget:
            return N, K, S, cap, batch, est
    return None


def get_model(model_id):
    """**판정 대상 모델 자신**을 bf16 으로 연다. 다른 모델이 오면 앞 모델을 놓아준다. 못 열면 (None, None)."""
    want = str(model_id or "").strip()
    if not want or want in _m["failed"]:
        return None, None
    if _m["id"] == want and _m["model"] is not None:
        return _m["tok"], _m["model"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    if _m["model"] is not None:
        log(f"  앞 모델 {_m['id']} 해제")
        _m["model"], _m["tok"], _m["id"] = None, None, None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    kw = dict(local_files_only=True, low_cpu_mem_usage=True, device_map="auto", dtype=torch.bfloat16,
              attn_implementation="sdpa")
    if os.environ.get("H22_QUANT") == "4bit":
        from transformers import BitsAndBytesConfig
        kw.pop("dtype")
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    for name in [want] + ([want.split(":")[0]] if ":" in want else []):
        t = time.time()
        try:
            try:                                # DeepSeek 은 AutoTokenizer 가 옛 LlamaTokenizer 로 열어 인코딩이 달라진다
                tok = PreTrainedTokenizerFast.from_pretrained(name, local_files_only=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            tok.padding_side = "left"
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            try:
                model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
            except Exception:
                k2 = dict(kw)
                k2.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(name, **k2).eval()
        except Exception as e:
            log(f"  {name} 로드 실패: {type(e).__name__} {str(e)[:120]}")
            continue
        log(f"  {name} 로드 {time.time() - t:.1f}s")
        _m["tok"], _m["model"], _m["id"] = tok, model, want
        return tok, model
    _m["failed"].add(want)
    return None, None


# ---------------------------------------------------------------- 생성·답 추출

def answer_of(text):
    """생성문에서 정수 답을 뽑는다. `ANSWER: n` 우선, 없으면 마지막 \\boxed{n}. 못 뽑으면 None(=오답)."""
    ms = ANS_RE.findall(text)
    if ms:
        return int(ms[-1])
    ms = BOXED_RE.findall(text)
    if ms:
        return int(ms[-1])
    return None


def is_degenerate(text, window=48, need=6):
    """무한반복 감지 (지휘 #H22-c-3). 끝부분에서 같은 조각이 need 번 이상 되풀이되면 망가진 생성이다.
    생성을 **멈추지는 않는다**(StoppingCriteria 는 배치 전체를 끊어 다른 시퀀스를 망친다) —
    대신 사후에 표시해 **판정에서 제외**한다(절단과 같은 처리)."""
    tail = text[-4000:]
    if len(tail) < window * need:
        return False
    probe = tail[-window:]
    return tail.count(probe) >= need


def generate(texts, n_samples, cap, deadline, model_id, prompt=None, sample=True, suffix=""):
    """한 배치 생성 → 각 입력마다 n_samples 개의 (텍스트, 완주여부) 쌍.

    **sample 과 n_samples 는 다른 축이다.** 평평한 시퀀스 설계에서는 같은 프롬프트를 행으로 복제하고
    n_samples=1 로 부르는데, 전에는 `do_sample = n_samples > 1` 이라 그 경로가 **greedy** 가 되어
    복제한 행이 전부 같은 출력을 냈다(= stab 이 항상 1.0, 샘플링 설계가 무너짐). 로컬 end-to-end 가 잡았다.

    완주 판정: 생성 토큰 안에 EOS 가 있으면 완주. 없으면 cap 에 걸린 **절단**이다.
    절단 샘플은 오답이 아니라 **판정 제외** 대상이다 (지휘 #H22-c-2)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    tok, model = get_model(model_id)
    if model is None:
        raise RuntimeError("model unavailable")

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return time.time() > deadline

    sys_p = SOLVE_PROMPT if prompt is None else prompt
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{sys_p}\n\n{t}"}],
                                       tokenize=False, add_generation_prompt=True) + suffix for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    gkw = dict(max_new_tokens=cap, pad_token_id=tok.pad_token_id,
               num_return_sequences=n_samples, do_sample=bool(sample),
               stopping_criteria=StoppingCriteriaList([Deadline()]))
    if sample:
        gkw.update(temperature=0.6, top_p=0.95)
    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, **gkw)
    new = out[:, enc["input_ids"].shape[1]:]
    dec = tok.batch_decode(new, skip_special_tokens=True)
    eos = {i for i in ([tok.eos_token_id] + list(getattr(model.generation_config, "eos_token_id", None)
                       if isinstance(getattr(model.generation_config, "eos_token_id", None), (list, tuple))
                       else [getattr(model.generation_config, "eos_token_id", None)])) if i is not None}
    rows = []
    for r in range(new.shape[0]):
        ids = new[r].tolist()
        fin = bool(eos & set(ids)) or len(ids) < cap
        rows.append((dec[r], fin and not is_degenerate(dec[r])))
    n_fin = sum(1 for _, f in rows if f)
    log(f"  생성 {len(texts)}×{n_samples} cap {cap} · {time.time() - t0:.0f}s · 완주 {n_fin}/{len(rows)}")
    return [rows[i * n_samples:(i + 1) * n_samples] for i in range(len(texts))]


def rephrase(texts, cap, deadline, model_id):
    """모델 자신에게 바꿔쓰게 하고 **숫자 보존 검사**를 통과한 것만 돌려준다.
    통과 못 하면 그 섭동은 버린다(K 가 줄어든다) — 정답이 바뀐 섭동을 쓰면 '무너졌다'의 정의가 깨진다."""
    out = [[] for _ in texts]
    for p_i, sysp in enumerate(REPHRASE_PROMPTS):
        if time.time() > deadline:
            break
        try:
            # 바꿔쓰기는 **greedy** 로 받고(재현돼야 한다) `</think>` 를 **강제**해 사고 과정을 건너뛴다.
            # 안 하면 추론 모델이 긴 사고를 먼저 뱉어 cap 안에 재서술이 안 나온다 — 로컬 e2e 에서
            # cap 220 으로 통과 0/2 였다(= 실전에서 K 가 조용히 줄어든다). h21c 의 즉답 강제와 같은 수법.
            got = generate(texts, 1, cap, deadline, model_id, prompt=sysp, sample=False,
                           suffix=REPHRASE_FORCE)
        except Exception as e:
            log(f"  rephrase {p_i} 실패: {type(e).__name__}")
            break
        kept = 0
        for i, cands in enumerate(got):
            cand, fin = cands[0] if cands else ("", False)
            # **첫 한 단락만** 쓴다. 지시를 어기고 풀이를 덧붙이면 뒤쪽은 버려야 한다.
            cand = (cand or "").strip().split("\n\n")[0].strip()
            if not fin or len(cand) < 20 or cand == texts[i] or not numbers_preserved(texts[i], cand):
                continue                         # 숫자가 바뀌었으면 정답도 바뀌었을 수 있다 → 버린다
            out[i].append((f"rephrase{p_i}", cand))
            kept += 1
        log(f"  rephrase {p_i}: 숫자보존 통과 {kept}/{len(texts)}")
    return out


# ---------------------------------------------------------------- 점수

def score_one(orig_answers, pert_answers, const_answers=None):
    """정의 그대로: P(원본 안정) × (섭동 과반 오답 비율). 높을수록 spurious.

    **절단·무한반복 샘플은 오답이 아니라 판정에서 제외한다 (지휘 #H22-c-2).** 입력은
    (답, 완주여부) 쌍의 리스트다. 근거: 자체 라벨에서 spurious 를 결정한 변형 실패의 69% 가
    절단이었고, 그건 문제의 성질이 아니라 예산이다. 완주한 샘플이 너무 적으면 숫자를 만들지 않고
    **중립(0.5)** 을 돌려준다 — 적은 완주 샘플로 과반을 세면 그게 바로 cap 아티팩트다.

    (여기서 개정 #37 의 '미완주 = 오답' 규칙과 갈린다. #37 은 **오프라인 라벨 생성**의 규칙이고
     주최측 라벨도 정해진 예산 안의 정확도라는 근거였다. 제출물에서는 우리가 쓰는 cap 이 주최측의
     예산보다 짧을 수 있어 절단이 순수한 측정 잡음이 된다. 두 규칙이 다른 이유를 여기 적어 둔다.)

    반환 (score, 진단). score 가 None 이면 판정 불가 → 호출자가 텍스트 폴백으로 보낸다."""
    def vote(pairs):
        """완주한 샘플만 모아 다수결. 반환 (과반답, 완주수, 전체수). 완주 0 이면 (None, 0, n)."""
        fin = [a for a, f in pairs if f]
        c = Counter(a for a in fin if a is not None)
        return (c.most_common(1)[0][0] if c else None), len(fin), len(pairs)

    gold, n_fin_o, n_o = vote(orig_answers)
    if n_fin_o < MIN_FIN_ORIG:
        return NEUTRAL, {"why": f"원본 완주 {n_fin_o}/{n_o} < {MIN_FIN_ORIG} → 중립",
                         "neutral": True, "n_fin_orig": n_fin_o}
    if gold is None:
        return NEUTRAL, {"why": f"원본이 완주했지만 답을 못 뽑음 ({n_fin_o}/{n_o})",
                         "neutral": True, "n_fin_orig": n_fin_o}
    fin_o = [a for a, f in orig_answers if f]
    stab = sum(1 for a in fin_o if a == gold) / len(fin_o)      # 분모 = **완주한** 원본 샘플

    broken, usable, dropped = 0, 0, 0
    for pairs in pert_answers:
        maj, n_fin, _ = vote(pairs)
        if n_fin == 0:                           # 이 섭동은 전부 절단 → 판정에서 뺀다(오답 아님)
            dropped += 1
            continue
        usable += 1
        if maj != gold:
            broken += 1
    if usable < MIN_PERT:
        return NEUTRAL, {"why": f"쓸 수 있는 섭동 {usable}/{len(pert_answers)} < {MIN_PERT} → 중립",
                         "neutral": True, "n_fin_orig": n_fin_o, "dropped": dropped}
    bfrac = broken / usable
    score = stab * bfrac
    diag = {"gold": gold, "stab": round(stab, 3), "broken": broken, "n_pert": usable,
            "dropped_pert": dropped, "n_fin_orig": f"{n_fin_o}/{n_o}", "bfrac": round(bfrac, 3)}
    if const_answers and CONST_W > 0:            # 보조항(기본 꺼짐): 상수를 바꿨는데 답이 안 바뀌면 암기 쪽
        same = cu = 0
        for pairs in const_answers:
            maj, n_fin, _ = vote(pairs)
            if n_fin == 0:
                continue
            cu += 1
            if maj == gold:
                same += 1
        if cu:
            score = (1 - CONST_W) * score + CONST_W * stab * (same / cu)
            diag["const_same"] = f"{same}/{cu}"
    return score, diag


# ---------------------------------------------------------------- 본체

def decide(problems, model_id="", effort="default"):
    """반환: 문제마다 bool(robust). 예외는 호출자가 잡아 전부 FALLBACK 으로 떨어뜨린다."""
    cache = jload(CACHE, {})
    state = jload(STATE, {"calls": 0})
    state["calls"] = int(state.get("calls", 0)) + 1
    jsave(STATE, state)
    done = int(state["calls"])
    left_calls = max(1, EXPECTED_CALLS - done + 1)
    budget = max(0.0, (remaining() - RESERVE) / left_calls)
    log(f"호출 {done} · 문제 {len(problems)} · model_id {model_id} · effort {effort} · 예산 {budget:.0f}s")

    texts = [str(p or "") for p in problems]
    out = [None] * len(texts)
    todo = []
    n_neutral_cached = 0
    for i, t in enumerate(texts):
        hit = cache.get(key_of(t, model_id, effort))
        if hit is not None and hit.get("score") is not None:
            if (hit.get("diag") or {}).get("neutral"):
                # 이미 '완주 부족 = 중립' 으로 판정된 문제다. 다시 생성해도 같은 결과이므로
                # 예산을 쓰지 않고 그대로 텍스트 폴백으로 보낸다.
                n_neutral_cached += 1
                continue
            out[i] = bool(float(hit["score"]) < THRESH)
        else:
            todo.append(i)
    if n_neutral_cached:
        log(f"캐시된 중립 {n_neutral_cached}건 → 텍스트 폴백")
    if not todo:
        log("전부 캐시 적중")
        return [bool(x) if x is not None else FALLBACK for x in out]

    tok, model = get_model(model_id)
    cfg = None
    if model is not None:
        cfg = plan(len(todo), budget, token_slots(model), 1.0, effort)
    if model is None or cfg is None:
        why = "모델을 못 열었다" if model is None else "예산이 최소형조차 못 산다"
        log(f"텍스트 폴백: {why}")
        vals = [float(n_sym(texts[i])) for i in todo]
        for i, rb in zip(todo, rank_split(vals, higher_is_spurious=True)):
            out[i] = bool(rb)
        return [bool(x) if x is not None else FALLBACK for x in out]

    N, K, S, cap, batch, est = cfg
    log(f"설계 N={N} K={K} S={S} cap={cap} · 배치 {batch} · 예상 {est:.0f}s / 예산 {budget:.0f}s")
    deadline = time.time() + budget

    # --- 섭동 만들기 (rephrase 는 모델이 필요하므로 먼저)
    perts = [rule_perturbations(texts[i])[:max(0, K - 2)] for i in todo]
    try:
        reph = rephrase([texts[i] for i in todo], REPHRASE_CAP,
                        min(deadline, time.time() + 0.15 * budget), model_id)
    except Exception as e:
        log(f"  rephrase 단계 실패: {type(e).__name__} — 규칙 섭동만 쓴다")
        reph = [[] for _ in todo]
    # rephrase 는 모델이 지시를 어기면 0개가 될 수 있다(실측: 숫자 보존 검사가 전부 걸러낸 적 있음).
    # 그때 K 가 조용히 줄면 '섭동 과반' 의 분해능이 떨어진다 → **규칙 섭동으로 K 를 메운다**.
    for j, i in enumerate(todo):
        got_p = perts[j] + reph[j]
        if len(got_p) < K:
            have = {v for _, v in got_p}
            got_p += [(k, v) for k, v in rule_perturbations(texts[i]) if v not in have]
        perts[j] = got_p[:K]
    log(f"섭동 수: {[len(p) for p in perts][:10]}{'…' if len(perts) > 10 else ''} (목표 K={K})")
    consts = [const_perturbations(texts[i], 2) if CONST_W > 0 else [] for i in todo]

    # --- 시퀀스를 평평하게 펼친다. 같은 프롬프트를 샘플 수만큼 **행으로 복제**하고
    #     num_return_sequences=1 로 돌린다 — 행마다 독립적으로 샘플링되므로 N 샘플이 그대로 나오고,
    #     원본(N 샘플)과 섭동(S 샘플)의 샘플 수가 달라도 한 배치에 섞을 수 있다.
    #     (max(N,S) 로 맞춰 돌리고 버리는 방식은 S<N 일 때 섭동 쪽을 N 배 생성해 낭비가 크다.)
    seqs = []                                    # (문제 인덱스, 슬롯 키, 텍스트)
    for j, i in enumerate(todo):
        for _ in range(N):
            seqs.append((i, "orig", texts[i]))
        for q, (kind, v) in enumerate(perts[j]):
            for _ in range(S):
                seqs.append((i, f"p{q}:{kind}", v))
        for q, (kind, v) in enumerate(consts[j]):
            for _ in range(S):
                seqs.append((i, f"c{q}:{kind}", v))
    got = {}                                     # (문제 인덱스, 슬롯 키) -> [(답, 완주여부) …]
    for g0 in range(0, len(seqs), batch):
        part = seqs[g0:g0 + batch]
        if time.time() > deadline:
            log(f"  마감 — 남은 시퀀스 {len(seqs) - g0}개 포기")
            break
        try:                                     # 복제한 행마다 **독립 샘플**이 나와야 한다 → sample=True
            dec = generate([t for _, _, t in part], 1, cap, deadline, model_id, sample=True)
        except Exception as e:
            log(f"  생성 실패: {type(e).__name__} {str(e)[:100]}")
            break
        for (i, slot, _), d in zip(part, dec):
            txt, fin = d[0] if d else ("", False)
            got.setdefault((i, slot), []).append((answer_of(txt), fin))

    # --- 점수: 문제마다 원본 슬롯과 섭동 슬롯을 모아 정의대로 계산
    for j, i in enumerate(todo):
        oa = got.get((i, "orig"))
        if not oa:
            continue                             # 생성이 아예 안 된 문제 → 아래 텍스트 폴백
        pa = [got[(i, f"p{q}:{kind}")] for q, (kind, _) in enumerate(perts[j])
              if (i, f"p{q}:{kind}") in got]
        ca = [got[(i, f"c{q}:{kind}")] for q, (kind, _) in enumerate(consts[j])
              if (i, f"c{q}:{kind}") in got]
        if not pa:
            log(f"  문제{i}: 섭동 결과가 하나도 없음 → 폴백")
            continue
        sc, diag = score_one(oa, pa, ca)
        diag["kinds"] = [k for k, _ in perts[j]]
        if sc is None or diag.get("neutral"):
            # **중립 출력** (지휘 #H22-c-2): 완주 샘플이 모자라 과반을 셀 수 없다. 확률은 0.5 로 기록하고
            # 판정은 동전 던지기 대신 텍스트 폴백(n_sym 순위)으로 보낸다 — 그게 0.5 보다 정보가 많다.
            cache[key_of(texts[i], model_id, effort)] = {"score": NEUTRAL, "diag": diag}
            log(f"  문제{i}: 중립 0.500 ({diag.get('why')}) → 텍스트 폴백")
            continue
        out[i] = bool(sc < THRESH)
        cache[key_of(texts[i], model_id, effort)] = {"score": sc, "diag": diag}
        log(f"  문제{i}: score {sc:.3f} gold {diag['gold']} stab {diag['stab']} "
            f"broken {diag['broken']}/{diag['n_pert']} (제외 섭동 {diag['dropped_pert']}, "
            f"원본완주 {diag['n_fin_orig']}) → robust {out[i]}")
    jsave(CACHE, cache)

    left = [i for i in range(len(texts)) if out[i] is None]
    if left:
        log(f"미판정 {len(left)}건 → 텍스트 폴백")
        vals = [float(n_sym(texts[i])) for i in left]
        for i, rb in zip(left, rank_split(vals, higher_is_spurious=True)):
            out[i] = bool(rb)
    return [bool(x) if x is not None else FALLBACK for x in out]
