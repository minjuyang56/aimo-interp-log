"""Count how many times a probe_inference module loads the language model.

Stubs out the artifact, tokenizer and model so no checkpoint is needed, then
calls predict_robustness on a 28-problem batch and reports the load count.
"""

import importlib.util
import sys
from pathlib import Path

import torch

HIDDEN = 8
LAYER = 3
N_PROBLEMS = 28


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(message["content"] for message in messages)

    def __call__(self, prompt, return_tensors=None, truncation=None):
        return {"input_ids": torch.zeros(1, 5, dtype=torch.long)}


class FakeOutput:
    def __init__(self):
        self.hidden_states = tuple(torch.ones(1, 5, HIDDEN) for _ in range(LAYER + 2))


class FakeModel:
    device = torch.device("cpu")

    def __call__(self, **kwargs):
        return FakeOutput()


def run(path: Path, name: str) -> tuple[int, int, bool]:
    module = load_module(path, name)

    artifact = module.ProbeArtifact(
        model_id="fake/model",
        system_prompt="solve it",
        data={
            "best_layer_index": LAYER,
            "recommended_strategy": {
                "name": "best_layer_mean_margin",
                "layer_index": LAYER,
            },
            "groups": [
                {
                    "probes": {
                        LAYER: {
                            "weights": [[0.0] * HIDDEN],
                            "bias": [0.0],
                            "threshold": [0.0],
                        }
                    }
                }
            ],
        },
        kind="pickle",
    )

    calls = {"n": 0}

    def counting_load_model(model_id):
        calls["n"] += 1
        return FakeTokenizer(), FakeModel()

    module._load_artifact = lambda model_id: artifact
    module._load_model = counting_load_model

    problems = [f"problem {index}" for index in range(N_PROBLEMS)]
    results = module.predict_robustness("qwen3-8b:low", problems)

    ok = (
        type(results) is list
        and len(results) == N_PROBLEMS
        and all(type(value) is bool for value in results)
    )
    return calls["n"], len(results), ok


if __name__ == "__main__":
    for label, path in [
        ("원본 (HEAD)", Path(sys.argv[1])),
        ("수정본", Path(sys.argv[2])),
    ]:
        loads, count, ok = run(path, f"pi_{label}")
        print(
            f"{label:14s} 모델 로드 {loads:3d}회 · 예측 {count}개 · "
            f"계약 준수 {'OK' if ok else 'FAIL'}"
        )
