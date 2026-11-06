"""H10-B 확장 분석 (지휘 #23, 사전등록 2026-09-29 17:45 — 나머지 3모델 결과 보기 전 작성).

(A) 지표 결합 (가중치 학습 없음): combo = z(compass_mean L//2) + z(traced_ratio L//2) − z(curvature L//2), z는 모델별 표준화. 큼 = robust.
    보조: 위 3특징 로지스틱 leave-one-problem-out (C=0.1, 폴드 안에서 표준화). 문제 단위 라벨(다수결) 33.
(B) 모델 간 비교(상호작용): rel_x = 이 모델 x − 나머지 3모델 평균 x (같은 문제), x ∈ {compass_mean, combo}. 큼 = robust.
    학습셋은 모델 간 라벨 차이 0건(H2)이라 문제 단위 라벨로는 0.5 근처가 정상 → (B)는 **모델 자기 라벨 행**(train-main-v2의 model_id별 is_robust)으로만 평가하고,
    절대값(compass) vs 상대값(rel_compass) 비교표를 보고. 4모델 전부 있어야 계산.
검문소 동일: null 30 · 문제길이(log T) 잔차화 · 모델 간 순위상관.

사용:  .venv/Scripts/python.exe experiments/h10b_ext.py [--perms 30]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from h10b_eval import ORDER, load_all, metrics  # noqa: E402

RES = HERE / "results"
ALIAS2ID = {"Qwen3_5-4B": "Qwen/Qwen3.5-4B", "Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
            "Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think", "DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}


def own_labels():
    """model_id → {pidx: y_own}. pidx는 h10b_trajectory.problems()와 같은 순서(groupby problem, sort=False, rstrip)."""
    import pandas as pd
    df = pd.read_parquet(RES / "train-main-v2.parquet")
    pid = {p.rstrip(): i for i, (p, _) in enumerate(df.groupby("problem", sort=False))}
    out = {}
    for r in df.itertuples():
        if r.is_robust is None or (isinstance(r.is_robust, float) and np.isnan(r.is_robust)):
            continue
        out.setdefault(r.model_id, {})[pid[r.problem.rstrip()]] = int(bool(r.is_robust))
    return out


def features(store):
    """모델 하나: {pidx: (y, logT, compass_mean, traced_ratio, curvature)} (L//2, 모델 전체 청크 평균으로 중심화)."""
    allc = np.concatenate([np.asarray(r["chunks"], np.float32)[:, 0] for r in store]); mu = allc.mean(0)
    out = {}
    for r in store:
        m = metrics(np.asarray(r["chunks"], np.float32)[:, 0] - mu)
        if m is None:
            continue
        out[int(r["pidx"])] = (int(r["y"]), float(np.log(float(r["T"]))), m["compass_mean"], m["traced_ratio"], m["curvature"])
    return out


def z(x):
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-9)


def evaluate(perms=30):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0); lines = []
    def out(s=""): print(s); lines.append(s)
    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls); b, *_ = np.linalg.lstsq(X, x, rcond=None); return x - X @ b
    def band(y, sc, n=None):
        null = [roc_auc_score(rng.permutation(y), sc) for _ in range(n or perms)]; lo, hi = np.percentile(null, [2.5, 97.5]); a = roc_auc_score(y, sc)
        return a, lo, hi, ("▲" if a > hi else ("▼" if a < lo else " "))
    data = load_all()
    if not data:
        out("결과 없음"); return
    names = sorted(data, key=lambda n: ORDER.index(n) if n in ORDER else 99)
    F = {n: features(data[n]) for n in names}
    out(f"H10-B 확장 (A) 결합 지표 · 문제 단위 라벨 · null {perms}회 · 부호: 큼 = robust")
    out(f"{'모델':12s} {'지표':10s} {'n':>3s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'잔차null':>13s}")
    combo = {}
    for n in names:
        f = F[n]; pids = sorted(p for p in f if f[p][0] >= 0)
        if len(pids) < 6: out(f"{n[:12]:12s} (라벨 {len(pids)}, 부족)"); continue
        y = np.array([f[p][0] for p in pids]); T = np.array([f[p][1] for p in pids])
        cm, tr, cv = (np.array([f[p][k] for p in pids]) for k in (2, 3, 4))
        c = z(cm) + z(tr) - z(cv); combo[n] = dict(zip(pids, c))
        a, lo, hi, fl = band(y, c); rs = resid(c, [T]); ra, lo2, hi2, fl2 = band(y, rs)
        out(f"{n[:12]:12s} {'combo':10s} {len(y):3d} {a:6.3f} [{lo:.2f},{hi:.2f}]{fl} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{fl2}")
        # 보조: LOPO 로지스틱 C=0.1
        X = np.column_stack([cm, tr, cv]); pred = np.zeros(len(y))
        for i in range(len(y)):
            tr_idx = np.array([j for j in range(len(y)) if j != i])
            mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-9
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000).fit((X[tr_idx] - mu) / sd, y[tr_idx])
            pred[i] = clf.decision_function(((X[i] - mu) / sd)[None])[0]
        a, lo, hi, fl = band(y, pred); rs = resid(pred, [T]); ra, lo2, hi2, fl2 = band(y, rs)
        out(f"{n[:12]:12s} {'lopo_C0.1':10s} {len(y):3d} {a:6.3f} [{lo:.2f},{hi:.2f}]{fl} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{fl2}")
    if len(combo) >= 2:
        common = sorted(set.intersection(*[set(v) for v in combo.values()]))
        y = np.array([F[names[0]][p][0] for p in common]); T = np.array([F[names[0]][p][1] for p in common])
        P = np.mean([resid(np.array([combo[n][p] for p in common]), [T]) for n in combo], 0); P = P / (P.std() + 1e-9)
        a, lo, hi, fl = band(y, P, 1000)   # 풀링 null 1000회, 문제 단위 섞기 (지휘 #25)
        corr = "  ".join(f"{a_[:6]}-{b_[:6]} {spearmanr([combo[a_][p] for p in common], [combo[b_][p] for p in common]).correlation:+.2f}" for i, a_ in enumerate(list(combo)) for b_ in list(combo)[i + 1:])
        out(f"{'풀링(잔차)':12s} {'combo':10s} {len(common):3d} {'':>6s} {'':>13s} {a:8.3f} [{lo:.2f},{hi:.2f}]{fl}   ρ: {corr}")

    out(); out(f"H10-B 확장 (B) 모델 간 상대값 · 모델 자기 라벨 행 · null {perms}회 · 부호: 큼 = robust")
    if len(names) < 4:
        out(f"  4모델 필요, 현재 {len(names)} → (B) 보류"); (RES / "h10b_ext_report.txt").write_text("\n".join(lines), encoding="utf-8"); return
    own = own_labels()
    common = sorted(set.intersection(*[set(F[n]) for n in names]))
    abs_cm = {n: {p: F[n][p][2] for p in common} for n in names}
    abs_co = {n: {p: combo[n].get(p, np.nan) for p in common} for n in names}
    rel_cm = {n: {p: abs_cm[n][p] - np.mean([abs_cm[m][p] for m in names if m != n]) for p in common} for n in names}
    rel_co = {n: {p: abs_co[n][p] - np.nanmean([abs_co[m][p] for m in names if m != n]) for p in common} for n in names}
    out(f"{'모델':12s} {'n_own':>5s} {'compass':>8s} {'rel_compass':>12s} {'combo':>8s} {'rel_combo':>10s}   (자기 라벨 AUC; n<6 또는 한 클래스면 —)")
    pool = {k: ([], []) for k in ("compass", "rel_compass", "combo", "rel_combo")}; pool_T = []; pool_y = []
    for n in names:
        lab = own.get(ALIAS2ID[n], {}); pids = [p for p in common if p in lab]
        y = np.array([lab[p] for p in pids]); cells = []
        for key, src in (("compass", abs_cm), ("rel_compass", rel_cm), ("combo", abs_co), ("rel_combo", rel_co)):
            x = np.array([src[n][p] for p in pids], float)
            if len(y) >= 6 and 0 < y.sum() < len(y) and np.isfinite(x).all():
                cells.append(f"{roc_auc_score(y, x):8.3f}")
            else:
                cells.append(f"{'—':>8s}")
            pool[key][0].extend(list(x)); pool[key][1].extend(list(y))
        pool_T.extend([F[n][p][1] for p in pids]); pool_y.extend(list(y))
        out(f"{n[:12]:12s} {len(y):5d} {cells[0]} {cells[1]:>12s} {cells[2]} {cells[3]:>10s}")
    yp = np.array(pool_y); Tp = np.array(pool_T)
    out(f"{'풀링(모델,문제) 행':20s} n={len(yp)} robust={int(yp.sum())}")
    for key in pool:
        x = np.array(pool[key][0], float); ok = np.isfinite(x)
        if ok.sum() < 6 or yp[ok].min() == yp[ok].max(): out(f"  {key:12s} (부족)"); continue
        a, lo, hi, fl = band(yp[ok], x[ok]); rs = resid(x[ok], [Tp[ok]]); ra, lo2, hi2, fl2 = band(yp[ok], rs)
        out(f"  {key:12s} AUC {a:.3f} [{lo:.2f},{hi:.2f}]{fl}  잔차 {ra:.3f} [{lo2:.2f},{hi2:.2f}]{fl2}")
    corr = "  ".join(f"{a_[:6]}-{b_[:6]} {spearmanr([rel_cm[a_][p] for p in common], [rel_cm[b_][p] for p in common]).correlation:+.2f}" for i, a_ in enumerate(names) for b_ in names[i + 1:])
    out(f"  rel_compass 모델 간 ρ: {corr}")
    (RES / "h10b_ext_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=30)
    evaluate(ap.parse_args().perms)
