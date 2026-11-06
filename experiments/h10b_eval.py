"""H10-B 평가 — 풀이 생성 궤적 나침반 (로컬 CPU). 사전등록 2026-09-29 16:25 (지휘 #19), 결과 보기 전 작성.

입력: results/h10b/**/<모델별칭>_v2.npz (Kaggle h10b_trajectory.py 출력. store: pidx·y·chunks(C,2,D: [L//2, 마지막층])·T·gen_tokens·eos)
층: 주지표 L//2 (인덱스 0), 보조 마지막층 (인덱스 1). 중심화 = 그 모델 전체 문제·전체 청크의 평균 청크벡터를 뺌.
지표 (부호 고정)
  (a) compass_mean  = mean_c cos(중심화 청크c, 중심화 마지막 청크)     큼 = robust
      compass_slope = 청크 지수에 대한 cos의 최소제곱 기울기            큼 = robust  (점점 마지막 방향으로 돈다)
  (b) traced_ratio  = ‖c_last − c_first‖ / Σ‖c_{i+1} − c_i‖ (전진거리/경로길이) 큼 = robust
      curvature     = mean_i (1 − cos(d_i, d_{i+1})), d_i = c_{i+1} − c_i     큼 = spurious (점수 = −curvature)
  (c) 길이 보정 (Gjølbye 2026): 생성길이는 1024 고정이지만 EOS 조기 종료로 청크 수가 달라질 수 있음 → log T(문제 길이)와 n_chunks 둘 다 잔차화.
검문소: null 30(모델별) · 풀링 null 1000(문제 단위 라벨 한 번 섞어 전 모델 동일 적용, 지휘 #25) · 잔차 AUC · 모델 간 순위상관 · 통과 시만 MATH137 DeepSeek 1회.
다중비교: 지표 5×층 2 = 10검정 → 본 지표는 **주지표 compass_mean L//2** 와 (A) combo 만. 나머지는 탐색적.

사용:  .venv/Scripts/python.exe experiments/h10b_eval.py [--perms 30]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RES = HERE / "results" / "h10b"
POOL_PERMS = 1000   # 풀링 null 횟수 (지휘 #25)
ORDER = ["Qwen3_5-4B", "Skywork-OR1-Math-7B", "Olmo-3-7B-Think", "DeepSeek-R1-0528-Qwen3-8B"]


def metrics(C):
    """C: (n_chunks, D) 중심화 청크벡터. 반환 dict."""
    n = len(C)
    if n < 3:
        return None
    nrm = np.linalg.norm(C, axis=1) + 1e-8
    cos = (C @ C[-1]) / (nrm * nrm[-1])
    x = np.arange(n); slope = float(np.polyfit(x, cos, 1)[0])
    d = np.diff(C, axis=0); dn = np.linalg.norm(d, axis=1)
    ratio = float(np.linalg.norm(C[-1] - C[0]) / (dn.sum() + 1e-8))
    cc = (d[:-1] * d[1:]).sum(1) / (dn[:-1] * dn[1:] + 1e-8)
    return dict(compass_mean=float(cos.mean()), compass_slope=slope, traced_ratio=ratio, curvature=float((1 - cc).mean()), n_chunks=n)


def load_all(only=None):
    out = {}
    for f in sorted(RES.rglob("*_v2.npz")):
        name = f.stem[:-3]
        if only and not any(o.lower() in name.lower() for o in only): continue
        d = np.load(f, allow_pickle=True); store = list(d["store"])
        if name in out and len(out[name]) >= len(store): continue     # 같은 모델이 여러 커널에 있으면(r1 부분·r2 전체) 문제 수 많은 쪽
        out[name] = store
    return out


def evaluate(perms=30, only=None):
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0); lines = []
    def out(s=""): print(s); lines.append(s)
    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls); b, *_ = np.linalg.lstsq(X, x, rcond=None); return x - X @ b
    data = load_all(only)
    if not data:
        out("결과 없음 (results/h10b/ 비어 있음)"); return
    KEYS = [("compass_mean", +1), ("compass_slope", +1), ("traced_ratio", +1), ("curvature", -1)]
    out(f"H10-B 풀이 궤적 나침반 (v2, Kaggle 생성 1024토큰·32토큰 청크) · 사전등록 부호: compass·ratio 큼=robust, curvature 큼=spurious · null {perms}회")
    for li, lname in ((0, "L//2"), (1, "last")):
        out(f"\n## 층 {lname}")
        out(f"{'모델':12s} {'지표':13s} {'n':>3s} {'AUC':>6s} {'bacc':>6s} {'null95%':>13s} {'ρ(nchunk)':>9s} {'ρ(logT)':>8s} {'잔차AUC':>8s} {'잔차null':>13s}")
        pooled = {k: [] for k, _ in KEYS}; vals = {k: {} for k, _ in KEYS}; raw = {k: {} for k, _ in KEYS}; y_ref = None; pid_ref = None
        for name in sorted(data, key=lambda n: ORDER.index(n) if n in ORDER else 99):
            store = data[name]
            allc = np.concatenate([np.asarray(r["chunks"], np.float32)[:, li] for r in store]); mu = allc.mean(0)
            rows = []
            for r in store:
                if int(r["y"]) < 0: continue
                m = metrics(np.asarray(r["chunks"], np.float32)[:, li] - mu)
                if m is None: continue
                rows.append((int(r["pidx"]), int(r["y"]), float(r["T"]), m))
            if len(rows) < 6: out(f"{name[:12]:12s} (라벨 문제 {len(rows)}개, 부족)"); continue
            pid = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); T = np.log(np.array([r[2] for r in rows]))
            nch = np.array([r[3]["n_chunks"] for r in rows], float)
            out(f"-- {name}: 라벨 {len(y)} (robust {y.sum()}) · 청크 수 중앙 {np.median(nch):.0f} [{nch.min():.0f},{nch.max():.0f}] · EOS 조기종료 {sum(1 for r in store if r.get('eos', False))}/{len(store)}")
            for key, sign in KEYS:
                x = np.array([r[3][key] for r in rows]); sc = sign * x
                auc = roc_auc_score(y, sc); bacc = balanced_accuracy_score(y, (sc > np.median(sc)).astype(int))
                null = [roc_auc_score(rng.permutation(y), sc) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
                rs = sign * resid(x, [T, nch]); ra = roc_auc_score(y, rs)
                null2 = [roc_auc_score(rng.permutation(y), rs) for _ in range(perms)]; lo2, hi2 = np.percentile(null2, [2.5, 97.5])
                f1 = "▲" if auc > hi else ("▼" if auc < lo else " "); f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
                out(f"{name[:12]:12s} {key:13s} {len(y):3d} {auc:6.3f} {bacc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {spearmanr(x, nch).correlation:+9.2f} {spearmanr(x, T).correlation:+8.2f} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2}")
                vals[key][name] = dict(zip(pid, rs / (rs.std() + 1e-9))); raw[key][name] = dict(zip(pid, sign * (x - x.mean()) / (x.std() + 1e-9)))
        # 풀링: 모든 모델에 있는 문제만
        names = [n for n in vals["compass_mean"]]
        if len(names) >= 2:
            common = sorted(set.intersection(*[set(vals["compass_mean"][n]) for n in names]))
            ymap = {int(r["pidx"]): int(r["y"]) for n in names for r in data[n] if int(r["y"]) >= 0}
            yc = np.array([ymap[p] for p in common])
            # 풀링 null (지휘 #25): 문제 단위 라벨을 한 번 섞어 모든 모델에 동일 적용 — 풀링 점수가 문제별 모델평균이므로 yc 섞기가 곧 그것. POOL_PERMS회.
            perm_ys = [rng.permutation(yc) for _ in range(POOL_PERMS)]
            for key, _ in KEYS:
                P = np.mean([[vals[key][n][p] for p in common] for n in names], 0)
                null = [roc_auc_score(py, P) for py in perm_ys]; lo, hi = np.percentile(null, [2.5, 97.5]); au = roc_auc_score(yc, P)
                Pr = np.mean([[raw[key][n][p] for p in common] for n in names], 0)
                nullr = [roc_auc_score(py, Pr) for py in perm_ys]; lor, hir = np.percentile(nullr, [2.5, 97.5]); aur = roc_auc_score(yc, Pr)
                corr = "  ".join(f"{a[:6]}-{b[:6]} {spearmanr([vals[key][a][p] for p in common], [vals[key][b][p] for p in common]).correlation:+.2f}" for i, a in enumerate(names) for b in names[i + 1:])
                tag = " ★주지표" if (key == "compass_mean" and lname == "L//2") else ""
                out(f"{'풀링(잔차)':12s} {key:13s} {len(common):3d} {'':>6s} {'':>6s} {'':>13s} {'':>9s} {'':>8s} {au:8.3f} [{lo:.2f},{hi:.2f}]{'▲' if au > hi else ('▼' if au < lo else ' ')}   원점수 모델평균 {aur:.3f} [{lor:.2f},{hir:.2f}]{'▲' if aur > hir else ('▼' if aur < lor else ' ')}   ρ: {corr}{tag}")
    (HERE / "results" / "h10b_report_v2.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=30); ap.add_argument("--only", nargs="*", help="모델 별칭 부분 문자열 필터")
    a = ap.parse_args(); evaluate(a.perms, a.only)
