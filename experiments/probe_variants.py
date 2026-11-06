"""프로브 **벡터 설계** 탐색용 추출 — 원본·rename 변형 × 평균풀링·마지막토큰. (지휘 #PROBE-AUDIT)

**이건 탐색이다. 확증이 아니다.** #77 주지표는 이미 1회 실행으로 확정됐고(미입증),
여기서 나오는 숫자로 그 판정을 고치지 않는다. 목적은 '벡터 설계가 신호를 못 담은 것인가' 를
같은 라벨·같은 nested CV·같은 null 로 재어 보는 것이다.

한 번의 추출로 네 가지를 전부 저장한다 — forward 를 두 번만 돌면 되므로 따로 돌릴 이유가 없다:
  * `h{l}`        원본, 문제 본문 구간 **평균 풀링** (= #77 이 쓴 그 벡터)
  * `hl{l}`       원본, **마지막 입력 토큰** (= 주최측 probing 베이스라인의 위치)
  * `v{l}`        rename 변형, 평균 풀링
  * `vl{l}`       rename 변형, 마지막 입력 토큰
→ 대조 벡터(h − v)와 결합([h, v])은 오프라인에서 만든다.

**rename 은 라벨 생성 때와 같은 함수·같은 시드로 만든다**: `official_variants(problem, seed=pidx)`
를 그대로 불러 `kind == "rename"` 을 집는다. rng 가 유형들 사이에 공유되므로 `edit_rename` 만
따로 부르면 **다른 텍스트가 나온다** — 그러면 라벨과 짝이 안 맞는다.

사용:
  python experiments/probe_variants.py --model <id> --labels <라벨json> --suffix _k8
  (정밀도 기본 4bit — #77 추출과 같게. bf16 비교는 --quant bf16 로 GPU 여유가 있을 때.)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from h21_uncertainty import SOLVE_PROMPT, load                 # noqa: E402
from probe_pilot import official_rows, own_rows                # noqa: E402
from variants_official import official_variants                # noqa: E402

OUT = ROOT / "experiments" / "results" / "probe"


def rename_of(row):
    """라벨 생성 때와 **같은** rename 텍스트. 못 만들면 None."""
    for kind, text in official_variants(row["problem"], seed=row["pidx"]):
        if kind == "rename":
            return text
    return None


def one(tok, model, text, layers):
    """(평균풀링, 마지막토큰) 두 벡터를 한 forward 에서 꺼낸다."""
    import torch
    full = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}],
                                   tokenize=False, add_generation_prompt=True)
    cpos = full.rfind(text[:40])
    s = len(tok(full[:cpos], add_special_tokens=False)["input_ids"]) if cpos > 0 else 0
    enc = tok(full, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    mean, last = {}, {}
    for l in layers:
        hs = out.hidden_states[l][0]
        mean[l] = hs[s:, :].float().mean(0).cpu().numpy()
        last[l] = hs[-1, :].float().cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return mean, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--tags", nargs="*", default=["own"])
    ap.add_argument("--suffix", default="_k8")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    alias = a.model.split("/")[-1].replace(".", "_")
    tok, model = load(a.model, a.quant)
    L = model.config.num_hidden_layers
    layers = sorted({max(1, L // 4), L // 2, (3 * L) // 4, L})
    print(f"[변형프로브] {a.model} · {a.quant} · 층 {L} · 인덱스 {layers}", flush=True)

    src = {"own": lambda: own_rows(a.labels), "official": official_rows}
    for tag in a.tags:
        rows = src[tag]()
        store = {f"h{l}": [] for l in layers}
        store.update({f"hl{l}": [] for l in layers})
        store.update({f"v{l}": [] for l in layers})
        store.update({f"vl{l}": [] for l in layers})
        has_var = []
        t0 = time.time()
        for n, r in enumerate(rows, 1):
            m0, l0 = one(tok, model, r["problem"], layers)
            rn = rename_of(r)
            if rn is None:
                #  rename 을 못 만드는 문제는 **원본 벡터로 채우고 플래그를 내린다**
                #  (0 으로 채우면 차이 벡터가 원본 그대로가 되어 조용히 다른 특징이 된다).
                m1, l1 = m0, l0
                has_var.append(0)
            else:
                m1, l1 = one(tok, model, rn, layers)
                has_var.append(1)
            for l in layers:
                store[f"h{l}"].append(m0[l]); store[f"hl{l}"].append(l0[l])
                store[f"v{l}"].append(m1[l]); store[f"vl{l}"].append(l1[l])
            if n % 10 == 0 or n == len(rows):
                print(f"  [{tag}] {n}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(
            OUT / f"{alias}_{tag}{a.suffix}_var.npz",
            layers=np.array(layers), has_var=np.array(has_var),
            y=np.array([int(r["label"]) for r in rows]),
            y_fin=np.array([-1 if r.get("label_fin") is None else int(r["label_fin"]) for r in rows]),
            y_major=np.array([-1 if r.get("label_major") is None else int(r["label_major"]) for r in rows]),
            gen_k=np.array([0 if r.get("gen_k") is None else int(r["gen_k"]) for r in rows]),
            pidx=np.array([r["pidx"] for r in rows]),
            words=np.array([len(r["problem"].split()) for r in rows]),
            **{k: np.stack(v) for k, v in store.items()})
        print(f"[변형프로브] 저장 {OUT / f'{alias}_{tag}{a.suffix}_var.npz'} · "
              f"rename 가능 {sum(has_var)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
