"""H10 — 나침반 ([author] 가설), A 버전: 층 궤적, forward만. 사전등록 2026-09-29.

재료: H9 추출이 함께 저장한 results/h9_layers_<모델>_<set>.npz
  store[i]      pidx · y · imp(단위별 지우기 중요도) · buckets · unit_layers (U, L+1, D) · last_layers (L+1, D)
  layer_mean    (L+1, D)  = 전체 문제·문제토큰의 층별 평균 (중심화용)

정의
  중심화        모든 벡터에서 층별 layer_mean 을 뺀 뒤 코사인
  구조액기스_l   V_rel+N_math+Q 단위의 층 l 벡터를 지우기 중요도로 가중합
  표면액기스_l   N_topic 동일. N_topic 없는 문제는 표면항 0 (개수 보고)
  북쪽          마지막 층 마지막 토큰 상태 (중심화)
지표(부호 고정)
  compass  = 평균_l [ cos(구조_l, 북쪽) − cos(표면_l, 북쪽) ]      큼 = robust
  steady   = std_l  cos(구조_l, 북쪽)                              작음 = robust  (점수 = −steady)
검문소: 섞기 null 30회 · 길이(log T)·단어비율(cnt) 잔차화 · 모델 간 순위상관 · 통과 시에만 MATH137 DeepSeek 1회

사용:  .venv/Scripts/python.exe experiments/h10_compass.py [--set v2]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias  # noqa: E402
from extract_train_v2 import LOCAL_MODELS  # noqa: E402

RES = HERE / "results"
STRUCT = {"V_rel", "N_math", "Q"}


def cos(a, b):
    na, nb = np.linalg.norm(a, axis=-1) + 1e-8, np.linalg.norm(b, axis=-1) + 1e-8
    return (a * b).sum(-1) / (na * nb)


def compute(model_id, which, src="h10"):
    d = np.load(RES / f"{src}_layers_{alias(model_id)}_{which}.npz", allow_pickle=True)
    mu = d["layer_mean"].astype(np.float32)                    # (L+1, D)
    rows = []
    for rec in d["store"]:
        U = rec["unit_layers"].astype(np.float32) - mu[None]   # (U, L+1, D) 중심화
        north = rec["last_layers"][-1].astype(np.float32) - mu[-1]
        imp, b = rec["imp"], rec["buckets"]
        st = np.array([x in STRUCT for x in b]); tp = ~st
        def essence(mask):
            if not mask.any() or imp[mask].sum() <= 0:
                return None
            w = imp[mask] / imp[mask].sum()
            return (U[mask] * w[:, None, None]).sum(0)          # (L+1, D)
        S, P = essence(st), essence(tp)
        if S is None:
            continue
        cs = cos(S[1:], north[None])                            # 층 1..L
        cp = cos(P[1:], north[None]) if P is not None else np.zeros_like(cs)
        rows.append((rec["pidx"], rec["y"], float((cs - cp).mean()), float(cs.std()), float(tp.mean()), P is None,
                     float(rec.get("T", rec["unit_layers"].shape[0]))))
    return rows


def main():
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="v2"); ap.add_argument("--perms", type=int, default=30); ap.add_argument("--src", default="h10", choices=["h10", "h9"])
    a = ap.parse_args(); rng = np.random.default_rng(0); lines = []
    def out(s=""): print(s); lines.append(s)
    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls); bb, *_ = np.linalg.lstsq(X, x, rcond=None); return x - X @ bb
    out(f"H10 나침반 A ({a.set}, 가중치={'균등' if a.src == 'h10' else 'H9 지우기 중요도'}) · 사전등록: compass 큼 = robust · steady 작음 = robust · 중심화 코사인 · null {a.perms}회")
    out(f"{'모델':12s} {'지표':8s} {'n':>3s} {'AUC':>6s} {'bacc':>6s} {'null95%':>13s} {'ρ(cnt)':>7s} {'ρ(logT)':>8s} {'잔차AUC':>8s} {'잔차null':>13s}")
    pooled = {"compass": [], "steady": []}; vals = {"compass": {}, "steady": {}}; y_ref = None
    for m in LOCAL_MODELS:
        f = RES / f"{a.src}_layers_{alias(m)}_{a.set}.npz"
        if not f.is_file():
            continue
        R = compute(m, a.set, a.src)
        R = [r for r in R if r[1] >= 0]
        y = np.array([r[1] for r in R]).astype(int); cnt = np.array([r[4] for r in R]); T = np.log(np.array([r[6] for r in R]))
        no_topic = sum(r[5] for r in R)
        for key, col, sign in (("compass", 2, +1), ("steady", 3, -1)):
            x = np.array([r[col] for r in R]); sc = sign * x
            auc = roc_auc_score(y, sc); bacc = balanced_accuracy_score(y, (sc > np.median(sc)).astype(int))
            null = [roc_auc_score(rng.permutation(y), sc) for _ in range(a.perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
            rs = sign * resid(x, [T, cnt]); ra = roc_auc_score(y, rs)
            null2 = [roc_auc_score(rng.permutation(y), rs) for _ in range(a.perms)]; lo2, hi2 = np.percentile(null2, [2.5, 97.5])
            f1 = "▲" if auc > hi else ("▼" if auc < lo else " "); f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
            out(f"{alias(m)[:12]:12s} {key:8s} {len(y):3d} {auc:6.3f} {bacc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {spearmanr(x, cnt).correlation:+7.2f} {spearmanr(x, T).correlation:+8.2f} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2}"
                + (f"   (N_topic 없는 문제 {no_topic})" if key == "compass" and no_topic else ""))
            pooled[key].append(rs / (rs.std() + 1e-9)); vals[key][m] = x; y_ref = y
    if y_ref is not None and len(vals["compass"]) >= 2:
        for key in ("compass", "steady"):
            P = np.mean(pooled[key], 0); null = [roc_auc_score(rng.permutation(y_ref), P) for _ in range(a.perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
            au = roc_auc_score(y_ref, P); names = list(vals[key])
            corr = "  ".join(f"{alias(names[i])[:6]}-{alias(names[j])[:6]} {spearmanr(vals[key][names[i]], vals[key][names[j]]).correlation:+.2f}" for i in range(len(names)) for j in range(i + 1, len(names)))
            out(f"{'풀링(잔차)':12s} {key:8s} {len(y_ref):3d} {'':>6s} {'':>6s} {'':>13s} {'':>7s} {'':>8s} {au:8.3f} [{lo:.2f},{hi:.2f}]{'▲' if au > hi else ('▼' if au < lo else ' ')}   모델간 ρ: {corr}")
    (RES / f"h10_report_{a.src}_{a.set}.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
