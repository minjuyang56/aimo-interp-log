"""동영상 프로브 — CoT를 생성하는 동안 hidden state가 그리는 경로의 기하.

토큰마다 층 L의 hidden state를 하나씩 받으면 D차원 공간에 점 T개가 순서대로 찍힌다.
그 경로에서 뽑는 숫자들:

  path_len        점 사이 거리의 합. 얼마나 멀리 돌아다녔나
  net_disp        첫 점→끝 점 직선거리
  tortuosity      path_len / net_disp. 1이면 직진, 클수록 헤맴
  mean_step       평균 보폭 · step_cv 보폭의 변동계수
  mean_turn       연속 세 점이 만드는 꺾임각 평균(라디안). 곡률의 대용
  turn_late_ratio 후반 절반 꺾임각 / 전반 절반 꺾임각. 후반에 잠잠해지면 <1
  early_conv      끝 점 방향과 코사인 유사도가 0.9를 넘는 첫 시점 / T. 일찍 답 근처면 작다
  gyration        점들의 무게중심 반경. 퍼져 있으면 크다
  final_drift     마지막 10% 구간의 이동거리 / path_len. 끝나고도 움직이면 크다
  n_tokens        T 자체 (생성 길이)

가설(사용자 정의): 진짜 추론은 탐색하다 수렴(긴 경로·큰 꺾임·늦은 수렴),
패턴 매칭은 처음부터 답 근처(짧은 경로·이른 수렴). 이건 검증할 가설이지 사실이 아니다.
"""

from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
FEATURES = ["path_len", "net_disp", "tortuosity", "mean_step", "step_cv", "mean_turn",
            "turn_late_ratio", "early_conv", "gyration", "final_drift", "n_tokens"]
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)


def geometry(P: np.ndarray) -> dict:
    """P: (T, D) float32. T가 3 미만이면 NaN 대신 0으로 채운다."""
    T = P.shape[0]
    if T < 3:
        return {k: 0.0 for k in FEATURES} | {"n_tokens": float(T)}
    steps = np.diff(P, axis=0)                       # (T-1, D)
    lens = np.linalg.norm(steps, axis=1) + 1e-8
    path_len = float(lens.sum())
    net_disp = float(np.linalg.norm(P[-1] - P[0]) + 1e-8)
    u = steps / lens[:, None]
    cos = np.clip((u[:-1] * u[1:]).sum(1), -1, 1)
    turn = np.arccos(cos)                            # (T-2,)
    half = len(turn) // 2
    early, late = turn[:half].mean() + 1e-8, turn[half:].mean() + 1e-8
    final = P[-1]
    rel = P - P[0]
    fin = final - P[0]
    sim = (rel @ fin) / (np.linalg.norm(rel, axis=1) * np.linalg.norm(fin) + 1e-8)
    hit = np.nonzero(sim > 0.9)[0]
    early_conv = float(hit[0] / T) if len(hit) else 1.0
    tail = max(1, int(0.1 * (T - 1)))
    return {
        "path_len": path_len,
        "net_disp": net_disp,
        "tortuosity": path_len / net_disp,
        "mean_step": float(lens.mean()),
        "step_cv": float(lens.std() / (lens.mean() + 1e-8)),
        "mean_turn": float(turn.mean()),
        "turn_late_ratio": float(late / early),
        "early_conv": early_conv,
        "gyration": float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())),
        "final_drift": float(lens[-tail:].sum() / path_len),
        "n_tokens": float(T),
    }


def to_vector(feat: dict) -> np.ndarray:
    v = np.array([feat[k] for k in FEATURES], dtype=np.float32)
    # 길이·거리류는 로그 스케일이 맞다 (토큰 수에 비례해 자란다)
    for i, k in enumerate(FEATURES):
        if k in ("path_len", "net_disp", "tortuosity", "mean_step", "gyration", "n_tokens"):
            v[i] = np.log1p(max(v[i], 0.0))
    return v


ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}


def artifact_for(model_id: str):
    model_id = ALIASES.get(model_id, model_id)
    name = model_id.split("/")[-1].replace(".", "_")
    path = ART / f"{name}.npz"
    if not path.is_file():
        return None
    d = np.load(path)
    return {"layer": int(d["layer"]), "max_new_tokens": int(d["max_new_tokens"]),
            "mu": d["mu"], "sd": d["sd"], "w": d["w"], "b": float(d["b"])}


def load(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # AutoTokenizer는 DeepSeek-R1-0528-Qwen3-8B를 옛 LlamaTokenizer로 열어 인코딩이 달라진다.
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


def trajectory(tok, model, problem: str, layer: int, max_new_tokens: int) -> np.ndarray:
    """생성하면서 매 스텝 층 L의 새 토큰 hidden state를 모은다. (T, D)"""
    import torch

    text = tok.apply_chat_template(
        [{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{problem}"}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )
    # out.hidden_states: 스텝마다 (층수+1)개의 (1, t, D). 첫 스텝은 프롬프트 전체, 이후는 새 토큰 1개.
    pts = [step[layer][0, -1].float().cpu().numpy() for step in out.hidden_states]
    return np.stack(pts).astype(np.float32)


def predict(model_id: str, problems: list[str]) -> list[bool]:
    art = artifact_for(model_id)
    if art is None:
        raise LookupError(f"no trajectory artifact for {model_id}")
    tok, model = load(ALIASES.get(model_id, model_id))
    try:
        out = []
        for p in problems:
            P = trajectory(tok, model, p, art["layer"], art["max_new_tokens"])
            v = to_vector(geometry(P))
            score = float(np.dot(art["w"], (v - art["mu"]) / art["sd"]) + art["b"])
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
