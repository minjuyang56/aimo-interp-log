"""#LEN-CAL 제출 라이브러리 — 텍스트만. 모델 로드 0 · 생성 0토큰. 사전등록 2026-10-15 (지휘 #FINAL-PLAN).

점수 = **−log(max(1, 공백분리 단어 수))** — h15C 의 `log_words` 와 **글자 그대로 같은 식**이다.
짧을수록 robust. (공식 train-main-v2 33문제에서 길이 단독 AUC 0.671 이 그 방향이다.)

**바뀐 것은 임계 하나다.** h15C 는 '두 클래스 중앙값의 중점'(−4.0840)을 썼고, 이것은
**사전확률 교정**을 쓴다: train 의 예측 True 비율이 **그 라벨의 robust 비율과 같아지도록** 고정.
  train = train-main-v2 의 주최측 라벨 문제단위 **33개** (robust 20 → 목표 True 비율 0.6061)
  규칙   = 점수 내림차순 20번째와 21번째의 **중점**

**이 제출물의 근거는 순위 능력이 아니라 교정이다.** #ORG-TRAIN 에서 이 점수의 AUC 0.671 은
자기 셔플 null 상단 0.70 을 넘지 못했다 — 순위는 유의하지 않다. 최종셋이 불균형이면 True 비율을
사전확률에 맞추는 것이 정확도 기댓값을 그 사전확률에 붙이고, 균형이면 이득이 0 이다. 그 양쪽을
사전등록에 적어 뒀다.

모델별 차이 **없음** — `model_id` 를 보지 않으므로 미지 모델에서 `KeyError` 가 날 수 없다
(절대 규칙 4). 특징은 문제 텍스트 하나뿐이다(절대 규칙 1·2).
"""

import json
import math
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts" / "lencal.json"


def log_words(text):
    """h15C `h15_lib.log_words` 와 같은 식. 바꾸면 임계가 무의미해진다."""
    return math.log(max(1, len(text.split())))


def load_art():
    return json.loads(ART.read_text(encoding="utf-8"))


def scores(problems):
    """문제별 점수. **높을수록 robust**(= 짧을수록 robust)."""
    return [-log_words(p) for p in problems]


def decide(problems):
    thr = float(load_art()["thr"])
    return [bool(s >= thr) for s in scores(problems)]
