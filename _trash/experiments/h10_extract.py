"""H10-A 재료 추출 (H9 의존 없음): 원문제 forward 1회 → 단어 자리 층별 벡터(균등 가중) · 마지막 토큰 층 궤적 · 층 평균.

단어 4통 목록·단위 분할은 h9_essence.units() 그대로(MATH137만 보고 작성, v2로 조정 금지). 가중치는 균등(imp = 1).
저장: results/h10_layers_<모델>_<set>.npz  — h10_compass.py --src h10 이 읽는다.

사용:  .venv/Scripts/python.exe experiments/h10_extract.py --model M [--set v2|math]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import LOCAL_MODELS, load_model  # noqa: E402
from h9_essence import prompt_of, texts, units  # noqa: E402

RES = HERE / "results"


def extract(model_id, which):
    import torch
    tok = load_tokenizer(model_id); model = load_model(model_id, "4bit")
    rows = texts(which); store = []; layer_sum, layer_cnt = None, 0; t0 = time.time()
    for k, (pidx, text, y) in enumerate(rows, 1):
        us = units(text)
        if len(us) < 4:
            continue
        p0 = prompt_of(tok, text); base = p0.index(text)
        enc = tok(p0, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k2: v.to(model.device) for k2, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        ALL = torch.stack([h[0] for h in hs]).float().cpu()                       # (L+1, T, D)
        uidx = [[i for i, (a, c) in enumerate(offs) if c > base + s and a < base + e] for (s, e, _) in us]
        keep = [j for j, idx in enumerate(uidx) if idx]
        unit_layers = torch.stack([ALL[:, uidx[j]].mean(1) for j in keep]).half()   # (U, L+1, D)
        span = [i for i, (a, c) in enumerate(offs) if c > base and a < base + len(text)]
        layer_sum = layer_sum + ALL[:, span].sum(1) if layer_sum is not None else ALL[:, span].sum(1); layer_cnt += len(span)
        store.append({"pidx": pidx, "y": y, "imp": np.ones(len(keep), np.float32), "buckets": np.array([us[j][2] for j in keep]),
                      "unit_layers": unit_layers.numpy(), "last_layers": ALL[:, -1].half().numpy(), "T": len(offs)})
        if k % 9 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)} units={len(keep)} T={len(offs)}  {time.time() - t0:.0f}s", flush=True)
    out = RES / f"h10_layers_{alias(model_id)}_{which}.npz"
    np.savez_compressed(out, store=np.array(store, dtype=object), layer_mean=(layer_sum / max(1, layer_cnt)).half().numpy(), model=model_id)
    print(f"저장 {out}  {time.time() - t0:.0f}s"); del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--set", default="v2", choices=["v2", "math"])
    a = ap.parse_args()
    for m in ([a.model] if a.model else LOCAL_MODELS):
        extract(m, a.set)
