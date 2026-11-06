"""H18 · H18b · H19 — 암기 신호 세 가지. 로컬 4bit, 생성은 64토큰뿐. 사전등록 2026-10-02 (지휘 #53).

문제당 모델당 비용: 순전파 2회(원본·rename) + greedy 64토큰 1회.
  H18   Min-K%++ (K=20%) on the **problem text**
        score_t = (log p(x_t) − mu_t) / sigma_t,  mu_t = E_{z~p}[log p(z)], sigma_t = sd_{z~p}[log p(z)]
        값 = 하위 20% score 의 평균.  **부호: 암기 신호 클수록 spurious** → robust 점수 = −minkpp
  H18b  문제 텍스트 **앞 50%** 만 주고 greedy **64토큰** 생성 → 뒤 50% 원문과 **ROUGE-L**(단어 LCS F1).
        **부호: 복원 잘 될수록 spurious** → robust 점수 = −rougeL
  H19   NLL(rename 문제 텍스트) − NLL(원본 문제 텍스트).  **부호: 갭 클수록 spurious** → robust 점수 = −gap
4모델 점수는 모델별 z 표준화 후 평균. 검문소: 공식 33문제 · 문제 단위 null 1000 · log(단어수) 잔차 AUC 병기.

사용:  .venv/Scripts/python.exe experiments/h18_h19.py --extract [--model M]
       .venv/Scripts/python.exe experiments/h18_h19.py [--perms 1000]
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
from variants_official import edit_rename  # noqa: E402

RES = HERE / "results"
K_FRAC = 0.20
GEN_TOK = 64


def official_problems():
    import pandas as pd
    df = pd.read_parquet(RES / "train-main-v2.parquet")
    rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna()
        if len(lab):
            rows.append({"pidx": i, "problem": p.rstrip(), "y": int(bool(lab.astype(bool).mean() >= 0.5))})
    return rows


def nll_and_minkpp(tok, model, text):
    """반환: (평균 NLL, Min-K%++ 20%). 텍스트 자체의 토큰 확률만 본다(채팅 템플릿 없음)."""
    import torch
    ids = tok(text, return_tensors="pt")["input_ids"].to(model.device)
    if ids.shape[1] < 4:
        return float("nan"), float("nan")
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()
    tgt = ids[0, 1:]
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    mu = (p * logp).sum(-1)                                   # E_z[log p(z)]
    var = (p * logp.pow(2)).sum(-1) - mu.pow(2)
    sigma = var.clamp(min=1e-8).sqrt()
    lp_tok = logp.gather(-1, tgt[:, None]).squeeze(-1)
    nll = float(-lp_tok.mean())
    z = (lp_tok - mu) / sigma
    k = max(1, int(len(z) * K_FRAC))
    minkpp = float(torch.topk(z, k, largest=False).values.mean())
    return nll, minkpp


def rouge_l(pred, ref):
    """단어 LCS 기반 F1."""
    a, b = pred.split(), ref.split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[len(a)][len(b)]
    if l == 0:
        return 0.0
    pr, rc = l / len(a), l / len(b)
    return 2 * pr * rc / (pr + rc)


def extract(model_id, rows):
    import torch
    tok = load_tokenizer(model_id)
    model = load_model(model_id, "4bit")
    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        text = r["problem"]
        nll_o, mkpp = nll_and_minkpp(tok, model, text)
        rn = edit_rename(text)
        nll_r = nll_and_minkpp(tok, model, rn)[0] if rn else float("nan")
        w = text.split()
        half = max(1, len(w) // 2)
        prefix, rest = " ".join(w[:half]), " ".join(w[half:])
        ids = tok(prefix, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            g = model.generate(input_ids=ids, max_new_tokens=GEN_TOK, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        cont = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        out.append({"pidx": r["pidx"], "y": r["y"], "minkpp": mkpp, "nll_gap": nll_r - nll_o,
                    "rougeL": rouge_l(cont, rest), "nll": nll_o, "words": len(w)})
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    f = RES / f"h18_{alias(model_id)}.npz"
    np.savez_compressed(f, store=np.array(out, dtype=object), model=model_id)
    print(f"저장 {f}", flush=True)
    del model
    torch.cuda.empty_cache()


def evaluate(perms=1000):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    data = {}
    for m in LOCAL_MODELS:
        f = RES / f"h18_{alias(m)}.npz"
        if f.is_file():
            data[m] = list(np.load(f, allow_pickle=True)["store"])
    if not data:
        out("결과 없음"); return
    names = list(data)
    common = sorted(set.intersection(*[{int(r["pidx"]) for r in data[m]} for m in names]))
    y = np.array([next(int(r["y"]) for r in data[names[0]] if int(r["pidx"]) == p) for p in common])
    lw = np.log(np.array([next(float(r["words"]) for r in data[names[0]] if int(r["pidx"]) == p) for p in common]))
    perm_ys = [rng.permutation(y) for _ in range(perms)]

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    def z(v):
        v = np.asarray(v, float)
        return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)

    out(f"H18 · H18b · H19 (로컬 4bit, 생성 {GEN_TOK}토큰) · 모델 {len(names)} · 문제 {len(y)} "
        f"(robust {int(y.sum())} / spurious {int(len(y) - y.sum())}) · null {perms}회 · 부호: 암기 신호 클수록 spurious")
    out(f"{'가설':6s} {'지표':10s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'잔차null':>13s} {'robust평균':>11s} {'spurious평균':>12s}")
    res = {}
    for tag, key in (("H18", "minkpp"), ("H18b", "rougeL"), ("H19", "nll_gap")):
        per = []
        for m in names:
            d = {int(r["pidx"]): float(r[key]) for r in data[m]}
            per.append(z([d[p] for p in common]))
        x = np.nanmean(per, 0)
        res[tag] = x
        ok = np.isfinite(x)                     # rename 불가 문제 등 결측 제외
        yy = y[ok]
        pys = [p[ok] for p in perm_ys] if ok.all() else [rng.permutation(yy) for _ in range(perms)]
        sc = -x[ok]
        auc = roc_auc_score(yy, sc)
        nl = [roc_auc_score(p, sc) for p in pys]
        lo, hi = np.percentile(nl, [2.5, 97.5])
        X = np.column_stack([np.ones(ok.sum()), lw[ok]])
        b, *_ = np.linalg.lstsq(X, x[ok], rcond=None)
        rs = -(x[ok] - X @ b)
        ra = roc_auc_score(yy, rs)
        nl2 = [roc_auc_score(p, rs) for p in pys]
        lo2, hi2 = np.percentile(nl2, [2.5, 97.5])
        f1 = "▲" if auc > hi else ("▼" if auc < lo else " ")
        f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
        out(f"{tag:6s} {key:10s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2} "
            f"{x[ok][yy == 1].mean():11.3f} {x[ok][yy == 0].mean():12.3f}  n={int(ok.sum())}")
    np.savez_compressed(RES / "h18_scores.npz", pidx=np.array(common), y=y, **{k: v for k, v in res.items()})
    (RES / "h18_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    if a.extract:
        rows = official_problems()
        print(f"대상 {len(rows)}문제", flush=True)
        for m in ([a.model] if a.model else LOCAL_MODELS):
            print(f"=== {m} ===", flush=True)
            extract(m, rows)
    else:
        evaluate(a.perms)
