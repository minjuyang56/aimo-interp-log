"""H5 — 암기(memorization) 신호. 라벨은 문제가 정한다 → spurious = 모델이 그 문제를 학습 때 봐서 외운 경우?

측정 (forward 1회, 생성 없음, **문제 텍스트 자체**의 토큰 로그확률. 채팅 템플릿 없이 raw 텍스트 + BOS):
  nll        평균 토큰 NLL
  mink20     Min-K% Prob (k=20): 로그확률 하위 20% 토큰의 평균 (멤버십 추론 표준)
  minkpp20   Min-K%++: 토큰별 z = (logp − μ_vocab)/σ_vocab 의 하위 20% 평균 (분포 기준 정규화)
  nll_2nd    뒷절반 조건부 NLL (앞절반은 조건으로만)
  zlib       총 NLL(nats) / zlib 압축 바이트 수 (길이·엔트로피 효과 제거용)

사전등록 부호: '쉽게 예측됨(낮은 NLL·높은 mink) = spurious'. 즉 robust 점수 = NLL 계열은 +, mink 계열은 −.
반대로 나와도 뒤집지 않고 그대로 보고한다.

모델: 판정 대상 4종(4bit) 각각 — 자기 행(train-main-v2에서 model_id가 그 모델인 행)과 v2 전체 + MATH137.
      DeepSeek은 공용 계측기 역할도 겸한다.

  --extract --model M   → results/h5_<M>.npz   (F: (n,5) · src: v2|math · y · pidx · problems)
  (기본) 평가            → (a) v2 모델별/풀링 AUC·bacc + 섞기 null  (b) 길이 통제  (c) 137→v2 외부검증

사용:  .venv/Scripts/python.exe experiments/h5_memorization.py --extract --model Qwen/Qwen3.5-4B
       .venv/Scripts/python.exe experiments/h5_memorization.py
"""

import argparse
import csv
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "analysis"))
from _paths import AGG_CSV  # noqa: E402
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import LOCAL_MODELS, load_model, problem_table  # noqa: E402

FEATS = ["nll", "mink20", "minkpp20", "nll_2nd", "zlib"]
SIGN = {"nll": +1, "mink20": -1, "minkpp20": -1, "nll_2nd": +1, "zlib": +1}   # robust 점수 방향 (사전등록)
RES = HERE / "results"


def texts_all():
    """v2 36문제(문제 단위 라벨) + MATH 137. 반환: list of (src, pidx, text, y, model_rows)"""
    import pandas as pd
    out = []
    tbl = problem_table()
    df = pd.read_parquet(RES / "train-main-v2.parquet")
    for r in tbl.itertuples():
        models = sorted(df[df.problem == r.problem].model_id.unique().tolist())
        out.append(("v2", int(r.pidx), r.problem, int(r.y), models))
    seen = set()
    for i, r in enumerate(csv.DictReader(open(AGG_CSV, newline="", encoding="utf-8"))):
        if r["problem_id"] in seen:
            continue
        seen.add(r["problem_id"])
        out.append(("math", i, r["original_problem"], int(r["model_is_robust"].strip().lower() == "true"), []))
    return out


def token_stats(tok, model, text):
    import torch
    ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(model.device)
    if ids.shape[1] < 4:
        return None
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()          # 다음 토큰 예측 (T-1, V)
    target = ids[0, 1:]
    logp_all = torch.log_softmax(logits, -1)
    lp = logp_all.gather(1, target[:, None])[:, 0]                      # (T-1,)
    p_all = logp_all.exp()
    mu = (p_all * logp_all).sum(-1)                                     # E[log p]
    sigma = ((p_all * (logp_all - mu[:, None]) ** 2).sum(-1)).clamp_min(1e-8).sqrt()
    z = (lp - mu) / sigma
    lp, z = lp.cpu().numpy(), z.cpu().numpy()
    n = len(lp); k = max(1, int(round(0.2 * n)))
    nll = float(-lp.mean())
    mink = float(np.sort(lp)[:k].mean())
    minkpp = float(np.sort(z)[:k].mean())
    nll2 = float(-lp[n // 2:].mean())
    zl = float(-lp.sum() / max(1, len(zlib.compress(text.encode("utf-8")))))
    return np.array([nll, mink, minkpp, nll2, zl], dtype=np.float32), n


def extract(model_id):
    import torch
    rows = texts_all()
    tok = load_tokenizer(model_id); model = load_model(model_id, "4bit")
    F, src, pidx, y, ntok, probs, mrows = [], [], [], [], [], [], []
    t0 = time.time()
    for i, (s, p, text, yy, models) in enumerate(rows, 1):
        r = token_stats(tok, model, text)
        if r is None:
            continue
        F.append(r[0]); ntok.append(r[1]); src.append(s); pidx.append(p); y.append(yy); probs.append(text)
        mrows.append(",".join(models))
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    out = RES / f"h5_{alias(model_id)}.npz"
    np.savez_compressed(out, F=np.stack(F), src=np.array(src), pidx=np.array(pidx), y=np.array(y, dtype=np.int8),
                        ntok=np.array(ntok), problems=np.array(probs), model_rows=np.array(mrows), model=model_id, feats=np.array(FEATS))
    print(f"저장 {out}  F{np.stack(F).shape}  {time.time() - t0:.0f}s")
    del model; torch.cuda.empty_cache()


def evaluate(perms=30):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    def lognw(texts):
        return np.log1p([len(re.findall(r"[A-Za-z]+", t)) for t in texts])

    def auc_row(name, score, y, lw, null_scores=None):
        auc = roc_auc_score(y, score)
        bacc = balanced_accuracy_score(y, (score > np.median(score)).astype(int))
        null = [roc_auc_score(rng.permutation(y), score) for _ in range(perms)]
        lo, hi = np.percentile(null, [2.5, 97.5])
        resid = score - np.polyval(np.polyfit(lw, score, 1), lw)
        r_auc = roc_auc_score(y, resid)
        rho = spearmanr(score, lw).correlation
        flag = "▲" if auc > hi else ("▼" if auc < lo else " ")
        out(f"  {name:34s} AUC {auc:.3f} bacc {bacc:.3f}  null[{lo:.2f},{hi:.2f}]{flag}  ρ(len){rho:+.2f}  길이잔차AUC {r_auc:.3f}")

    data = {m: np.load(RES / f"h5_{alias(m)}.npz", allow_pickle=True) for m in LOCAL_MODELS if (RES / f"h5_{alias(m)}.npz").is_file()}
    if not data:
        out("추출 결과 없음"); return
    out(f"H5 암기 신호 · 사전등록 부호: robust 점수 = +nll, −mink20, −minkpp20, +nll_2nd, +zlib  (perms={perms})")

    # (a) v2 — 모델별 (v2 33 라벨 문제 전체, 계측기 = 그 모델)  +  자기 행만
    out("\n(a) v2 33문제, 계측기별 (라벨은 문제 단위)")
    pooled = {}
    for m, d in data.items():
        v2 = (d["src"] == "v2") & (d["y"] >= 0)
        y, F, lw = d["y"][v2].astype(int), d["F"][v2].astype(np.float32), lognw(d["problems"][v2])
        out(f"== {alias(m)}  (n={len(y)}, robust {y.sum()})")
        for j, f in enumerate(FEATS):
            sc = SIGN[f] * F[:, j]
            auc_row(f, sc, y, lw)
            pooled.setdefault(f, []).append((sc - sc.mean()) / (sc.std() + 1e-6))
        own = v2 & np.array([alias(m).split("-")[0].lower() in mr.lower() for mr in d["model_rows"]])
        if own.sum() >= 6 and d["y"][own].min() != d["y"][own].max():
            yo, Fo = d["y"][own].astype(int), d["F"][own].astype(np.float32)
            out(f"   자기 행만 (n={own.sum()}, robust {yo.sum()}): " + "  ".join(f"{f} {roc_auc_score(yo, SIGN[f] * Fo[:, j]):.2f}" for j, f in enumerate(FEATS)))
    if len(data) >= 2:
        first = next(iter(data.values())); v2 = (first["src"] == "v2") & (first["y"] >= 0)
        y, lw = first["y"][v2].astype(int), lognw(first["problems"][v2])
        out(f"== 4모델 z-평균 풀링 (n={len(y)})")
        for f in FEATS:
            auc_row(f, np.mean(pooled[f], axis=0), y, lw)

    # (c) 외부검증 137 → v2 (모델별, 5특징 로지스틱; 부호는 학습이 정하되 null로 판정)
    out("\n(c) 외부검증: MATH137로 학습 → v2 33 시험 (5특징 로지스틱 + 길이 대조군)")
    for m, d in data.items():
        tr = d["src"] == "math"; te = (d["src"] == "v2") & (d["y"] >= 0)
        Xtr, ytr, Xte, yte = d["F"][tr].astype(np.float32), d["y"][tr].astype(int), d["F"][te].astype(np.float32), d["y"][te].astype(int)
        def fit(Xa, ya, Xb):
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
            return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit((Xa - mu) / sd, ya).decision_function((Xb - mu) / sd)
        mg = fit(Xtr, ytr, Xte)
        null = [roc_auc_score(yte, fit(Xtr, rng.permutation(ytr), Xte)) for _ in range(perms)]
        lo, hi = np.percentile(null, [2.5, 97.5]); auc = roc_auc_score(yte, mg)
        flag = "▲" if auc > hi else ("▼" if auc < lo else " ")
        lwtr, lwte = lognw(d["problems"][tr])[:, None], lognw(d["problems"][te])[:, None]
        auc_len = roc_auc_score(yte, fit(lwtr, ytr, lwte))
        out(f"  {alias(m):30s} 5특징 AUC {auc:.3f} bacc {balanced_accuracy_score(yte, (mg > 0).astype(int)):.3f}  null[{lo:.2f},{hi:.2f}]{flag}   길이만 AUC {auc_len:.3f}")
    (RES / "h5_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장 {RES / 'h5_report.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--model", default=None); ap.add_argument("--perms", type=int, default=30)
    a = ap.parse_args()
    if a.extract:
        for m in ([a.model] if a.model else LOCAL_MODELS):
            extract(m)
    else:
        evaluate(a.perms)
