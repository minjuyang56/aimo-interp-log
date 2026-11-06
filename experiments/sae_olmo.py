"""#SAE-OLMO — Olmo 층 15 공개 SAE 로 '풀이 도중' 활성을 희소 분해한다. 로컬 · 비용 $0.

사전등록: CLAUDE.md `## #SAE-OLMO` (커밋 `69bb2bf`, 실행 전 기록). 설정을 바꿔 다시 돌리면
그 사실과 이유를 기록에 남기고 **탐색**으로 표기한다.

## 단계 (메모리 때문에 반드시 이 순서다)
`--stage hidden` : Olmo 를 **4bit** 로 열어 greedy 풀이 앞 2048 토큰의 `hidden_states[16]` 을
                   **토큰별로** 디스크에 뜬다(31행 × 2048 × 4096 float16 ≈ 520 MB). 그리고 **모델을 해제한다.**
`--stage sae`    : SAE(float32 5.37 GB)를 올려 그 hidden 을 통과시킨다. **동시에 올리면 8 GB 에 안 들어간다.**
                   품질 게이트(분산설명률 · L0)를 먼저 찍고, 활성을 **희소로**(TopK 인덱스·값) 저장한다.

## 층이 맞는 근거 (문서 확인, 추측 아님)
SAE README: *"Hook convention: layer `L` uses `outputs.hidden_states[L + 1]`"* →
`layer_15` = `hidden_states[16]` = 우리 Olmo 특징 층(L=32 · L//2=16). 활성 종류도 같다
(post-layer residual stream). 이 저장소를 여러 번 괴롭힌 off-by-one 이 여기서는 문서로 일치한다.

## 왜 희소로 저장하나
TopK64 라 토큰당 비영이 64개뿐이다 → (인덱스, 값) 만 남기면 31 × 2048 × 64 ≈ 32 MB 다.
밀집이면 31 × 2048 × 163,840 = 10.4G 원소로 불가능하다. 이 저장이 사전 선별과
'대표 토큰 문맥'(사전등록 (4))을 **같은 파일에서** 가능하게 한다.

사용:
  .venv/Scripts/python.exe experiments/sae_olmo.py --stage hidden
  .venv/Scripts/python.exe experiments/sae_olmo.py --stage sae
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
PROBE = ROOT / "experiments" / "results" / "probe"
GEN = ROOT / "experiments" / "results" / "runpod" / "genprobe"
OUT = ROOT / "experiments" / "results" / "sae"
SAE_DIR = Path(r"C:\side\sae\olmo-l15")
MODEL = "allenai/Olmo-3-7B-Think"
ALIAS = "Olmo-3-7B-Think"
LAYER_HS = 16          # hidden_states 인덱스. SAE 의 layer_15 와 같은 지점(README hook convention)
WIN = 2048
TOPK = 64


def log(m):
    print(f"[sae {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- 단계 1: hidden
def stage_hidden(a):
    import torch
    from h21_uncertainty import SOLVE_PROMPT, load

    npz = np.load(PROBE / f"{ALIAS}_gen{WIN}.npz", allow_pickle=True)
    pidx = npz["pidx"].astype(int)
    probs = [str(x) for x in npz["problems"]] if "problems" in npz else None
    rows = json.loads((GEN / "oi8q4v8ia3ojq0" / "out" / f"{ALIAS}_greedy.json").read_text(encoding="utf-8"))
    sol = {int(r["pidx"]): r["greedy_text"] for r in rows if r.get("greedy_text")}
    if probs is None:
        prob_of = {int(r["pidx"]): r["problem"] for r in rows}
        probs = [prob_of[int(q)] for q in pidx]
    log(f"대상 {len(pidx)}행 (특징 npz 와 같은 행·같은 순서) · greedy 텍스트 {len(sol)}개")

    tok, model = load(MODEL, a.quant)
    L = model.config.num_hidden_layers
    log(f"모델 {MODEL} · {a.quant} · 층 {L} · hidden_states 인덱스 {LAYER_HS} 사용")
    assert LAYER_HS == L // 2, f"층 가정이 깨졌다: L//2={L // 2} != {LAYER_HS}"

    H, meta = [], []
    t0 = time.time()
    for n, (q, p) in enumerate(zip(pidx, probs), 1):
        if int(q) not in sol:
            raise SystemExit(f"pidx {q} 의 greedy 텍스트가 없다")
        pid = tok(tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{p}"}],
                                          tokenize=False, add_generation_prompt=True),
                  add_special_tokens=False)["input_ids"]
        sid = tok(sol[int(q)], add_special_tokens=False)["input_ids"][:WIN]
        ids = torch.tensor([pid + sid], device=model.device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        s = len(pid)
        h = out.hidden_states[LAYER_HS][0, s:s + len(sid), :].float().cpu().numpy().astype(np.float16)
        H.append(h)
        meta.append({"pidx": int(q), "n_tok": int(len(sid)), "tok_ids": [int(x) for x in sid]})
        del out
        torch.cuda.empty_cache()
        if n % 5 == 0 or n == len(pidx):
            log(f"  {n}/{len(pidx)} · {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{ALIAS}_hidden_l{LAYER_HS}_{a.quant}.npz",
             **{f"h{i}": x for i, x in enumerate(H)},
             pidx=pidx, n_tok=np.array([m["n_tok"] for m in meta]))
    (OUT / f"{ALIAS}_hidden_meta_{a.quant}.json").write_text(
        json.dumps({"layer_hs": LAYER_HS, "win": WIN, "quant": a.quant, "rows": meta}), encoding="utf-8")
    mb = (OUT / f"{ALIAS}_hidden_l{LAYER_HS}_{a.quant}.npz").stat().st_size / 1e6
    log(f"hidden 저장 {mb:.0f} MB · 토큰 수 최소 {min(m['n_tok'] for m in meta)} "
        f"중앙 {sorted(m['n_tok'] for m in meta)[len(meta) // 2]} 최대 {max(m['n_tok'] for m in meta)}")
    log("모델을 해제한다 — 다음 단계는 SAE 를 올린다(동시에는 8 GB 에 안 들어간다)")


# ---------------------------------------------------------------- 단계 2: SAE
def load_sae(dev, dtype):
    import torch
    from safetensors.torch import load_file
    cfg = json.loads((SAE_DIR / "cfg.json").read_text(encoding="utf-8"))
    w = load_file(str(SAE_DIR / "sae_weights.safetensors"))
    log(f"SAE 텐서: {{k: tuple(v.shape) for k, v in w.items()}}".replace("{k: tuple(v.shape) for k, v in w.items()}",
        str({k: tuple(v.shape) for k, v in w.items()})))
    log(f"cfg: architecture {cfg.get('architecture')} · d_in {cfg.get('d_in')} · d_sae {cfg.get('d_sae')} · "
        f"normalize {cfg.get('normalize_activations')} · k {cfg.get('activation_fn_kwargs')}")
    W_enc = w["W_enc"].to(dev, dtype)            # (d_in, d_sae)
    W_dec = w["W_dec"].to(dev, dtype)            # (d_sae, d_in)
    b_enc = w["b_enc"].to(dev, dtype)
    b_dec = w["b_dec"].to(dev, dtype)
    return cfg, W_enc, W_dec, b_enc, b_dec


def stage_sae(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # **정밀도가 품질 게이트를 좌우한다** → float32 를 먼저 쓴다(SAE 가 float32 로 배포됐다).
    # 8 GB 에 가중치 5.37 GB + 활성 버퍼가 안 들어가면 float16 으로 내리고 **그 사실을 로그에 남긴다**
    # (TopK 선택이 바뀔 수 있으므로 게이트 숫자에 붙여 읽어야 한다).
    dtype = {"float32": torch.float32, "float16": torch.float16}[a.sae_dtype] if dev == "cuda" \
        else torch.float32
    cfg, W_enc, W_dec, b_enc, b_dec = load_sae(dev, dtype)
    k = int((cfg.get("activation_fn_kwargs") or {}).get("k", TOPK))
    apply_bdec_in = bool(cfg.get("apply_b_dec_to_input", False))
    log(f"TopK k={k} · apply_b_dec_to_input={apply_bdec_in} · 장치 {dev} {dtype}")

    z = np.load(OUT / f"{ALIAS}_hidden_l{LAYER_HS}_{a.quant}.npz", allow_pickle=True)
    pidx = z["pidx"].astype(int)
    num, den, l0s, rows = 0.0, 0.0, [], []
    sp_idx, sp_val = [], []
    for i, q in enumerate(pidx):
        x = torch.from_numpy(z[f"h{i}"].astype(np.float32)).to(dev, dtype)      # (T, 4096)
        xin = x - b_dec if apply_bdec_in else x
        pre = xin @ W_enc + b_enc                                               # (T, d_sae)
        val, idx = torch.topk(pre, k, dim=-1)
        val = torch.relu(val)
        rec = torch.zeros_like(x)
        rec.index_add_(0, torch.arange(x.shape[0], device=dev).repeat_interleave(k),
                       (val.reshape(-1, 1) * W_dec[idx.reshape(-1)]).to(dtype))
        rec = rec + b_dec
        xf, rf = x.float(), rec.float()
        num += float(((xf - rf) ** 2).sum())
        den += float(((xf - xf.mean(0, keepdim=True)) ** 2).sum())
        l0s.append(float((val > 0).sum(-1).float().mean()))
        sp_idx.append(idx.cpu().numpy().astype(np.int32))
        sp_val.append(val.float().cpu().numpy().astype(np.float32))
        rows.append(int(q))
        del x, pre, val, idx, rec
        torch.cuda.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == len(pidx):
            log(f"  {i + 1}/{len(pidx)} · 누적 분산설명률 {1 - num / max(1e-9, den):.4f}")
    fvu = num / max(1e-9, den)
    ev = 1.0 - fvu
    l0 = float(np.mean(l0s))
    log("")
    log(f"## 품질 게이트 — 분산설명률 **{ev:.4f}** · L0 **{l0:.2f}** (TopK k={k} 이므로 {k} 가 정상)")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{ALIAS}_sae_sparse_{a.quant}.npz",
                        pidx=np.array(rows), **{f"i{n}": x for n, x in enumerate(sp_idx)},
                        **{f"v{n}": x for n, x in enumerate(sp_val)})
    (OUT / f"{ALIAS}_sae_gate_{a.quant}.json").write_text(json.dumps(
        {"explained_variance": ev, "fvu": fvu, "l0": l0, "k": k, "d_sae": int(cfg["d_sae"]),
         "quant": a.quant, "layer_hs": LAYER_HS, "sae": "layer_15/width_163k/topk_64_raw",
         "n_rows": len(rows)}), encoding="utf-8")
    log(f"희소 활성 저장 {(OUT / f'{ALIAS}_sae_sparse_{a.quant}.npz').stat().st_size / 1e6:.0f} MB")
    if ev < 0.7:
        log(f"!! 분산설명률 {ev:.4f} < 0.7 → **사전등록대로 여기서 중단한다**. "
            f"정밀도 분기({a.quant})를 보고에 적고, bf16 재측정 여부는 보고 후 결정한다.")
        return 2
    log(f"게이트 통과 ({ev:.4f} >= 0.7) → 특징·평가 단계로 간다")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["hidden", "sae"])
    ap.add_argument("--quant", default="4bit", help="4bit = 주경로(A 와 같은 정밀도) · bf16 = 정밀도 분기")
    ap.add_argument("--sae-dtype", default="float32", choices=["float32", "float16"],
                    help="SAE 연산 정밀도. float32 가 기본(배포 정밀도) · OOM 이면 float16")
    a = ap.parse_args()
    return stage_hidden(a) if a.stage == "hidden" else stage_sae(a)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
