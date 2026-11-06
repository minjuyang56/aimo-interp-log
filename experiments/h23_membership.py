"""H23 기억 모드(멤버십 추론) 측정 — 문제 텍스트에 prefill 1회. 사전등록 2026-10-10 (지휘 #H23).

무엇을 재는가 (전부 '익숙할수록 spurious', 부호 고정)
  주지표 mink20  = 토큰 logprob 의 **하위 20% 평균** (Min-K% Prob, Shi et al. 2023). 높으면 spurious.
  보조  nll      = 토큰 평균 NLL. 낮으면 spurious.
  보조  zlib     = 총 NLL(nats) / zlib 압축 바이트 수. 암기면 압축률 대비 NLL 이 낮다.
  보조  d_mink   = mink20(원본) − mean(mink20(rename), mink20(typos)).
                   외운 문제는 **원본만 유독 익숙**해야 하므로 크면 spurious.

**`token_stats` 는 `_trash/experiments/h5_memorization.py` 의 것을 그대로 옮겼다.** 정의를 건드리면
H5(nll 0.66~0.70 · mink20) · H18(minkpp20 0.454) 와 숫자를 나란히 놓을 수 없게 된다 — 이번 측정의
용도 절반이 그 비교다(공식 33 판은 같은 조건의 재측정이다).

새로운 것은 **평가 대상**이다: H5·H18·H19 는 전부 공식 33문제의 **문제단위** 라벨로 쟀는데 그 라벨에는
모델×문제 성분이 설계상 없다(H2). 여기서는 **Skywork 자체 라벨 88개**(그 모델 것)로도 잰다.

사용:
  .venv/Scripts/python.exe experiments/h23_membership.py --model Skywork/Skywork-OR1-Math-7B --source own
  .venv/Scripts/python.exe experiments/h23_membership.py --model <m> --source official
"""

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from variants_official import edit_rename, edit_typos  # noqa: E402

OUT = ROOT / "experiments" / "results" / "h23"
OWN = ROOT / "experiments" / "results" / "genlab" / "skywork_labels_merged.json"
V2 = ROOT / "experiments" / "results" / "train-main-v2.parquet"


def log(m):
    print(m, flush=True)


def official_problems():
    """공식 train-main-v2 문제단위 라벨 (모델 간 다수결). H5·H18 과 같은 대상."""
    import pandas as pd
    df = pd.read_parquet(V2)
    rows = []
    for prob, g in df.groupby("problem", sort=True):
        vals = [bool(v) for v in g["is_robust"].tolist() if v is not None and v == v]
        if not vals:
            continue
        t, f = vals.count(True), vals.count(False)
        if t == f:
            continue
        rows.append({"problem": str(prob).strip(), "y": int(t > f), "pidx": len(rows)})
    return rows


def own_problems():
    """Skywork 자체 라벨 88 — **모델별 라벨**. 이게 H23 의 새로운 판이다."""
    d = json.loads(OWN.read_text(encoding="utf-8"))
    return [{"problem": r["problem"].strip(), "y": int(bool(r["label"])),
             "y_fin": (None if r.get("label_fin") is None else int(bool(r["label_fin"]))),
             "pidx": r["pidx"], "gen_cap": r.get("gen_cap")}
            for r in d if isinstance(r.get("label"), bool)]


def load_model(model_id):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              PreTrainedTokenizerFast)
    try:                                   # DeepSeek 은 AutoTokenizer 가 옛 LlamaTokenizer 로 열어 인코딩이 달라진다
        tok = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, device_map="auto",
                                             quantization_config=q, low_cpu_mem_usage=True).eval()
    return tok, m


def token_stats(tok, model, text):
    """**H5 의 것을 그대로 옮겼다** — 정의를 바꾸면 H5·H18 숫자와 비교가 깨진다.
    반환 (nll, mink20, minkpp20, nll_2nd, zlib), 토큰 수."""
    import torch
    ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(model.device)
    if ids.shape[1] < 4:
        return None
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()          # 다음 토큰 예측 (T-1, V)
    target = ids[0, 1:]
    logp_all = torch.log_softmax(logits, -1)
    lp = logp_all.gather(1, target[:, None])[:, 0]
    p_all = logp_all.exp()
    mu = (p_all * logp_all).sum(-1)
    sigma = ((p_all * (logp_all - mu[:, None]) ** 2).sum(-1)).clamp_min(1e-8).sqrt()
    z = (lp - mu) / sigma
    lp, z = lp.cpu().numpy(), z.cpu().numpy()
    n = len(lp)
    k = max(1, int(round(0.2 * n)))
    nll = float(-lp.mean())
    mink = float(np.sort(lp)[:k].mean())
    minkpp = float(np.sort(z)[:k].mean())
    nll2 = float(-lp[n // 2:].mean())
    zl = float(-lp.sum() / max(1, len(zlib.compress(text.encode("utf-8")))))
    return np.array([nll, mink, minkpp, nll2, zl], dtype=np.float32), n


def perturbed(text, pidx):
    """정답 불변 섭동 2종(rename · typos). 시드는 문제마다 고정 — 재현된다."""
    rng = np.random.default_rng(pidx)
    out = {}
    rn = edit_rename(text)
    if rn and rn != text:
        out["rename"] = rn
    tp = edit_typos(text, rng)
    if tp and tp != text:
        out["typos"] = tp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", choices=["own", "official"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = own_problems() if a.source == "own" else official_problems()
    alias = a.model.split("/")[-1].replace(".", "_")
    log(f"[H23] {a.model} · {a.source} · 문제 {len(rows)}")

    tok, model = load_model(a.model)
    recs = []
    for i, r in enumerate(rows):
        st = token_stats(tok, model, r["problem"])
        if st is None:
            log(f"  문제 {i} 너무 짧음 → 제외")
            continue
        f0, n0 = st
        pm = {}
        for kind, txt in perturbed(r["problem"], r["pidx"]).items():
            s2 = token_stats(tok, model, txt)
            if s2 is not None:
                pm[kind] = float(s2[0][1])          # 섭동본의 mink20
        rec = {"pidx": r["pidx"], "y": r["y"], "y_fin": r.get("y_fin"),
               "problem": r["problem"], "ntok": int(n0),
               "nll": float(f0[0]), "mink20": float(f0[1]), "minkpp20": float(f0[2]),
               "nll_2nd": float(f0[3]), "zlib": float(f0[4]),
               "pert_mink": pm,
               # 외운 문제는 원본만 유독 익숙해야 한다 → 크면 spurious (부호 고정)
               "d_mink": (float(f0[1]) - float(np.mean(list(pm.values())))) if pm else None}
        recs.append(rec)
        if i % 10 == 0:
            log(f"  {i + 1}/{len(rows)} nll {rec['nll']:.3f} mink20 {rec['mink20']:.3f} "
                f"d_mink {rec['d_mink'] if rec['d_mink'] is None else round(rec['d_mink'], 3)}")
    p = OUT / f"{alias}_{a.source}_h23.json"
    p.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    nd = sum(1 for r in recs if r["d_mink"] is not None)
    log(f"[H23] 저장 {p.relative_to(ROOT)} · {len(recs)}건 (d_mink 계산된 것 {nd})")


if __name__ == "__main__":
    main()
