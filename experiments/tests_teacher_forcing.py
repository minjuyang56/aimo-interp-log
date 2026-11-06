"""지휘 #30 핵심 가정 검증: '생성 중 스텝별 hidden' == '프롬프트+생성문 teacher forcing forward 의 같은 자리 hidden'.
인과 마스크라 같아야 한다. 작은 무작위 Qwen2 로 수치 비교.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(r"C:\side\aimo-interp-log")
sys.path.insert(0, str(ROOT / "experiments"))
from extract_hidden import load_tokenizer  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

tok = load_tokenizer("Skywork/Skywork-OR1-Math-7B")
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=96, intermediate_size=192, num_hidden_layers=6,
                  num_attention_heads=6, num_key_value_heads=2, max_position_embeddings=2048)
torch.manual_seed(0)
model = Qwen2ForCausalLM(cfg).eval()
mid = (len(model.model.layers) + 1) // 2      # hidden_states 인덱스 (len(hidden_states)=L+1 의 절반)

prompt = "Solve: Alice has 12 apples and Bob has three times as many. How many in total?"
pid = tok(prompt, return_tensors="pt")["input_ids"]
P = pid.shape[1]
T = 40

# --- (A) 생성 루프: 새 토큰을 넣은 직후의 L//2 hidden (genlab.py 방식)
with torch.no_grad():
    out = model(input_ids=pid, use_cache=True, output_hidden_states=True)
past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
gen, hs_gen = [], []
with torch.no_grad():
    for _ in range(T):
        o = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True, output_hidden_states=True)
        past = o.past_key_values
        gen.append(int(nxt)); hs_gen.append(o.hidden_states[mid][0, -1].clone())
        nxt = o.logits[:, -1].argmax(-1)
A = torch.stack(hs_gen)

# --- (B) teacher forcing: 프롬프트+생성 토큰 전체를 한 번에
ids = torch.cat([pid, torch.tensor([gen])], dim=1)
grab = {}
h = model.model.layers[mid - 1].register_forward_hook(lambda m, a, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
with torch.no_grad():
    model(input_ids=ids)
h.remove()
B = grab["h"][0, P:P + T]

d = (A - B).abs()
print(f"생성 토큰 {T}개 · 벡터 차원 {A.shape[1]}")
print(f"최대 절대차 {d.max():.3e} · 평균 {d.mean():.3e} · A 노름 중앙 {A.norm(dim=1).median():.3f}")
cos = torch.nn.functional.cosine_similarity(A, B, dim=-1)
print(f"코사인 최소 {cos.min():.6f} · 평균 {cos.mean():.6f}")
assert cos.min() > 0.9999, "teacher forcing 이 생성 시점 hidden 과 다르다"
# 청크 평균까지 같은지
CH = 32
ca = torch.stack([A[i:i + CH].mean(0) for i in range(0, T, CH)])
cb = torch.stack([B[i:i + CH].mean(0) for i in range(0, T, CH)])
print(f"청크 {len(ca)}개 코사인 최소 {torch.nn.functional.cosine_similarity(ca, cb, dim=-1).min():.6f}")
print("검증 통과: 두 방식의 벡터가 같다")
