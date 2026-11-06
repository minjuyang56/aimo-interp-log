"""정지 사진 프로브 — 마지막 프롬프트 토큰의 hidden state 한 장에 선 하나.

런타임 의존성은 torch·transformers·numpy뿐. sklearn 없이 numpy로 판정한다
(로지스틱 회귀 추론은 내적 하나라 sklearn을 실을 이유가 없다 — 실패 지점 하나 줄임).
"""

from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)


# val-sample의 옛 별칭. 본선은 HF 이름을 그대로 넘기지만 로컬 검증엔 이게 온다.
ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}


def artifact_for(model_id: str):
    """model_id에 맞는 프로브. 없으면 None — 호출자가 폴백한다."""
    model_id = ALIASES.get(model_id, model_id)
    name = model_id.split("/")[-1].replace(".", "_")
    path = ART / f"{name}.npz"
    if not path.is_file():
        return None
    d = np.load(path)
    return {"layer": int(d["layer"]), "mu": d["mu"], "sd": d["sd"], "w": d["w"], "b": float(d["b"])}


def load(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # AutoTokenizer는 DeepSeek-R1-0528-Qwen3-8B를 옛 LlamaTokenizer로 열어 인코딩이 달라진다.
    # 업스트림 extraction.py와 같은 우회: Fast 토크나이저를 직접 연다.
    from transformers import PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except (OSError, TypeError, ValueError):
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, local_files_only=True,
        device_map="auto" if device == "cuda" else None, low_cpu_mem_usage=True,
    ).eval()
    if device == "cpu":
        model.to("cpu")
    return tok, model


def last_token_hidden(tok, model, problem: str, layer: int) -> np.ndarray:
    import torch

    text = tok.apply_chat_template(
        [{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{problem}"}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    return hs[layer][0, -1].float().cpu().numpy()


def predict(model_id: str, problems: list[str]) -> list[bool]:
    art = artifact_for(model_id)
    if art is None:
        raise LookupError(f"no probe for {model_id}")
    tok, model = load(ALIASES.get(model_id, model_id))   # 배치당 1회
    try:
        out = []
        for p in problems:
            h = last_token_hidden(tok, model, p, art["layer"])
            score = float(np.dot(art["w"], (h - art["mu"]) / art["sd"]) + art["b"])
            out.append(bool(score > 0.0))
        return out
    finally:
        del model
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
