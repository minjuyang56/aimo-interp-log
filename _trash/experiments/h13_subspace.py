"""H13 — 표면 부분공간 PCA. 사전등록 2026-09-29.

모델별 L//2층. MATH137 문제마다 (변형 − 원문제) 마지막 토큰 차이벡터(rename·typos·distract, 3개/문제)를 모아 PCA 상위 k=8 = 표면 부분공간 S.
S는 **MATH137만으로** 산출하고 v2에는 적용만 한다. 중심화 평균도 MATH137 원문제 벡터의 평균.
h(v2 원문제 L//2층 마지막 토큰, 중심화) = 표면성분 P_S h + 구조성분 (I − P_S) h.
지표(부호 고정)
  energy   = ‖P_S h‖² / ‖h‖²                        큼 = spurious (점수 = −energy)
  compass13 = cos(구조성분, 북쪽) − cos(표면성분, 북쪽)   큼 = robust   (북쪽 = 최종층 마지막 토큰, MATH137 최종층 평균으로 중심화)
검문소: null 30 · 길이·표면개수 잔차화 · 모델 간 순위상관 · 통과 시만 MATH137 DeepSeek 1회(그때는 137 자체 평가).

  --extract --model M   MATH137 (원문 + 변형 3) 와 v2 (원문) 의 L//2·최종층 마지막 토큰 저장
  (기본) 평가

사용:  .venv/Scripts/python.exe experiments/h13_subspace.py --extract --model M
       .venv/Scripts/python.exe experiments/h13_subspace.py
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import LOCAL_MODELS, load_model  # noqa: E402
from gen_labels import EDITS  # noqa: E402
from h9_essence import prompt_of, texts, units  # noqa: E402

RES = HERE / "results"; K = 8; VARIANTS = ("rename", "typos", "distract")


def fwd(model, **kw):
    """logits_to_keep 미지원 모델(TypeError)이면 빼고 다시 호출."""
    try:
        return model(**kw)
    except TypeError:
        kw.pop("logits_to_keep", None); return model(**kw)



def last_vecs(tok, model, text):
    import torch
    ids = tok(prompt_of(tok, text), return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        hs = fwd(model, input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
    mid = len(hs) // 2
    return hs[mid][0, -1].float().cpu().numpy(), hs[-1][0, -1].float().cpu().numpy(), ids.shape[1]


def extract(model_id):
    import torch
    tok = load_tokenizer(model_id); model = load_model(model_id, "4bit"); rng = np.random.default_rng(0); t0 = time.time()
    store = {}
    for which in ("math", "v2"):
        M, N, D, Y, P, T, NS = [], [], [], [], [], [], []
        for k, (pidx, text, y) in enumerate(texts(which), 1):
            m0, n0, tlen = last_vecs(tok, model, text)
            diffs = []
            if which == "math":
                for v in VARIANTS:
                    t = EDITS[v](text, rng)
                    if t and t != text:
                        m1, _, _ = last_vecs(tok, model, t); diffs.append(m1 - m0)
            M.append(m0); N.append(n0); D.append(np.array(diffs) if diffs else np.zeros((0, len(m0)))); Y.append(y); P.append(pidx); T.append(tlen)
            NS.append(sum(1 for _, _, b in units(text) if b == "N_topic"))
            if k % 25 == 0: print(f"  [{which}] {k}  {time.time() - t0:.0f}s", flush=True)
        store[which] = dict(mid=np.array(M, np.float32), fin=np.array(N, np.float32), diffs=np.array(D, dtype=object), y=np.array(Y, np.int8), pidx=np.array(P), T=np.array(T), n_surf=np.array(NS))
    out = RES / f"h13_{alias(model_id)}.npz"
    np.savez_compressed(out, **{f"{w}_{k}": v for w, d in store.items() for k, v in d.items()}, model=model_id)
    print(f"저장 {out}  {time.time() - t0:.0f}s"); del model; torch.cuda.empty_cache()


def evaluate(perms=30):
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0); lines = []
    def out(s=""): print(s); lines.append(s)
    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls); b, *_ = np.linalg.lstsq(X, x, rcond=None); return x - X @ b
    def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    data = {m: np.load(RES / f"h13_{alias(m)}.npz", allow_pickle=True) for m in LOCAL_MODELS if (RES / f"h13_{alias(m)}.npz").is_file()}
    if not data: out("결과 없음"); return
    out(f"H13 표면 부분공간 (S = MATH137 차이벡터 PCA k={K}, v2 적용) · 사전등록: energy 큼 = spurious · compass13 큼 = robust · null {perms}회")
    out(f"{'모델':12s} {'지표':9s} {'n':>3s} {'AUC':>6s} {'bacc':>6s} {'null95%':>13s} {'ρ(n_surf)':>9s} {'ρ(logT)':>8s} {'잔차AUC':>8s} {'잔차null':>13s}  S설명분산")
    pooled = {"energy": [], "compass13": []}; vals = {"energy": {}, "compass13": {}}; y_ref = None
    for m, d in data.items():
        diffs = np.vstack([x for x in d["math_diffs"] if len(x)]); mu_mid = d["math_mid"].mean(0); mu_fin = d["math_fin"].mean(0)
        Dc = diffs - diffs.mean(0); U, S, Vt = np.linalg.svd(Dc, full_matrices=False); B = Vt[:K]          # (K, D)
        ev = float((S[:K] ** 2).sum() / (S ** 2).sum())
        lab = d["v2_y"] >= 0; y = d["v2_y"][lab].astype(int); ns = d["v2_n_surf"][lab].astype(float); T = np.log(d["v2_T"][lab].astype(float))
        H = d["v2_mid"][lab] - mu_mid; NN = d["v2_fin"][lab] - mu_fin
        energy, comp = [], []
        for h, n in zip(H, NN):
            hs = B.T @ (B @ h); hc = h - hs
            energy.append(float(hs @ hs / (h @ h + 1e-9))); comp.append(cos(hc, n) - cos(hs, n))
        for key, x, sign in (("energy", np.array(energy), -1), ("compass13", np.array(comp), +1)):
            sc = sign * x; auc = roc_auc_score(y, sc); bacc = balanced_accuracy_score(y, (sc > np.median(sc)).astype(int))
            null = [roc_auc_score(rng.permutation(y), sc) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
            rs = sign * resid(x, [T, ns]); ra = roc_auc_score(y, rs)
            null2 = [roc_auc_score(rng.permutation(y), rs) for _ in range(perms)]; lo2, hi2 = np.percentile(null2, [2.5, 97.5])
            f1 = "▲" if auc > hi else ("▼" if auc < lo else " "); f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
            out(f"{alias(m)[:12]:12s} {key:9s} {len(y):3d} {auc:6.3f} {bacc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {spearmanr(x, ns).correlation:+9.2f} {spearmanr(x, T).correlation:+8.2f} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2}  {ev:.2f}")
            pooled[key].append(rs / (rs.std() + 1e-9)); vals[key][m] = x; y_ref = y
    if y_ref is not None:
        for key in ("energy", "compass13"):
            if len(pooled[key]) >= 2:
                P = np.mean(pooled[key], 0); null = [roc_auc_score(rng.permutation(y_ref), P) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5]); au = roc_auc_score(y_ref, P)
                names = list(vals[key]); corr = "  ".join(f"{alias(names[i])[:6]}-{alias(names[j])[:6]} {spearmanr(vals[key][names[i]], vals[key][names[j]]).correlation:+.2f}" for i in range(len(names)) for j in range(i + 1, len(names)))
                out(f"{'풀링(잔차)':12s} {key:9s} {len(y_ref):3d} {'':>6s} {'':>6s} {'':>13s} {'':>9s} {'':>8s} {au:8.3f} [{lo:.2f},{hi:.2f}]{'▲' if au > hi else ('▼' if au < lo else ' ')}   ρ: {corr}")
    (RES / "h13_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--model"); ap.add_argument("--perms", type=int, default=30)
    a = ap.parse_args()
    if a.extract:
        for m in ([a.model] if a.model else LOCAL_MODELS): extract(m)
    else:
        evaluate(a.perms)
