"""제출 A — 공용 계측기 프로브 + 배치 내 중앙값 분할.

설계(사전등록, 2026-09-29):
  계측기   deepseek-ai/DeepSeek-R1-0528-Qwen3-8B 하나. 판정 대상 model_id와 무관하게 이걸 로드한다.
           근거: train-main-v2에서 라벨은 문제 단위(11/11 모델 간 일관). 워커 캐시에 있음.
  점수     문제를 solve 프롬프트로 읽혀 층 18 마지막 토큰 벡터 → 표준화 → w·x + b
  판정     배치(are_robust 한 호출) 안에서 점수 내림차순 상위 ceil(n/2)개만 True. 절대 임계값 없음.
           근거: 검증셋이 모델별·문제별 균형(주최측 공지). 동점은 입력 순서 앞이 우선.
  런타임   로드 1회(warm cache ~2 s) + 문제당 forward 1회(GPU ~0.3 s). 배치 21문제 ≈ 10 s.
  실패 시  solution.py가 상수 False로 폴백.
"""

from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent / "artifacts" / "instrument_L18.npz"
INSTRUMENT = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)


def load_artifact():
    d = np.load(ART)
    return {"layer": int(d["layer"]), "mu": d["mu"], "sd": d["sd"], "w": d["w"], "b": float(d["b"])}


def load_instrument():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    try:                                    # AutoTokenizer는 이 체크포인트를 옛 Llama로 열어 인코딩이 달라진다
        tok = PreTrainedTokenizerFast.from_pretrained(INSTRUMENT, local_files_only=True)
    except (OSError, TypeError, ValueError):
        tok = AutoTokenizer.from_pretrained(INSTRUMENT, local_files_only=True)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        INSTRUMENT, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
        device_map="auto" if cuda else None).eval()
    if not cuda:
        model.to("cpu")
    return tok, model


def scores(problems):
    import torch
    art = load_artifact()
    tok, model = load_instrument()
    out = []
    try:
        for p in problems:
            text = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{p}"}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                h = model(**ids, output_hidden_states=True).hidden_states[art["layer"]][0, -1].float().cpu().numpy()
            out.append(float(np.dot(art["w"], (h - art["mu"]) / art["sd"]) + art["b"]))
    finally:
        del model
        try:
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass
    return out


def median_split(score_list):
    """상위 ceil(n/2)개 True. 동점은 입력 순서 앞이 우선(안정 정렬)."""
    n = len(score_list)
    k = (n + 1) // 2
    order = sorted(range(n), key=lambda i: (-score_list[i], i))
    top = set(order[:k])
    return [bool(i in top) for i in range(n)]
