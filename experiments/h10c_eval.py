"""H10-C — 자체 생성 라벨에서의 궤적 나침반 평가. 사전등록 2026-09-30 (지휘 #31).

입력: experiments/results/genlab/**/<별칭><태그>.npz  (genlab_vllm.py 출력. store 에 chunks(C,D) fp16 · label_official · label_unofficial)
지표 (L//2층 청크, 모델 전체 청크 평균으로 중심화 — h10b_eval.metrics 와 같은 정의)
  **주지표 traced_ratio** = 전진거리/경로길이,  큼 = robust (부호 고정)
  compass_mean = mean_c cos(청크c, 마지막 청크)  — **부호 고정 없이 양방향 보고** (지휘 #31)
  탐색적: compass_slope · curvature
라벨: **공식(label_official) 이 주**, 비공식(label_unofficial) 은 같은 지표로 따로. 두 라벨 일치율도 보고.
통제: 모델별로 **log(원문제 토큰 수)** 와 **청크 수(생성 길이)** 를 잔차화. null 은 라벨 섞기 1000회.

사용:  .venv/Scripts/python.exe experiments/h10c_eval.py [--perms 1000] [--only Skywork]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from h10b_eval import metrics  # noqa: E402

RES = HERE / "results" / "genlab"
KEYS = [("traced_ratio", +1, "주지표"), ("compass_mean", 0, "양방향"), ("compass_slope", 0, "탐색"), ("curvature", -1, "탐색")]


def load_runs(only=None):
    out = {}
    for f in sorted(RES.rglob("*.npz")):
        if f.name.endswith("_gen.json"):
            continue
        name = f.stem
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        try:
            d = np.load(f, allow_pickle=True)
            store = list(d["store"])
        except Exception as e:
            print(f"  [건너뜀] {f.name}: {type(e).__name__}"); continue
        if not store or "label_official" not in store[0]:
            continue
        prev = out.get(name)
        if prev is None or len(store) > len(prev):
            out[name] = store
    return out


def features(store):
    """{pidx: dict(지표…, logT, n_chunks, y_off, y_un)} — 청크가 3개 이상인 문제만."""
    good = [s for s in store if np.asarray(s["chunks"]).ndim == 2 and np.asarray(s["chunks"]).shape[0] >= 3]
    if not good:
        return {}
    mu = np.concatenate([np.asarray(s["chunks"], np.float32) for s in good]).mean(0)
    out = {}
    for s in good:
        m = metrics(np.asarray(s["chunks"], np.float32) - mu)
        if m is None:
            continue
        det = s.get("worst_cause") or {}
        out[int(s["pidx"])] = dict(m, logT=float(np.log(max(1, int(s.get("prompt_tokens", 1))))),
                                   y_off=s.get("label_official", s.get("label")), y_un=s.get("label_unofficial"),
                                   gen=int(s.get("gen_tokens", 0)),
                                   unfin=float(det.get("n_unfin", 0)) / max(1, det.get("n_unfin", 0) + det.get("n_fin", 0)))
    return out


def evaluate(perms=1000, only=None):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls)
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    runs = load_runs(only)
    if not runs:
        out("결과 없음 (experiments/results/genlab 에 npz 없음)"); return lines
    out(f"H10-C 자체 라벨 궤적 나침반 · 주지표 traced_ratio(큼=robust) · compass_mean 은 양방향 · null {perms}회 · 통제: log(원문제 토큰) + 청크 수 + 미완주율(있을 때)")
    for name, store in runs.items():
        F = features(store)
        n_all = len(store)
        lab_off = [p for p in F if F[p]["y_off"] is not None]
        lab_un = [p for p in F if F[p]["y_un"] is not None]
        both = [p for p in F if F[p]["y_off"] is not None and F[p]["y_un"] is not None]
        agree = sum(F[p]["y_off"] == F[p]["y_un"] for p in both)
        n_fin = sum(bool(s.get("finished")) for s in store)
        n_cor = sum(bool(s.get("correct")) for s in store)
        spur = sum(F[p]["y_off"] is False for p in lab_off)
        out()
        out(f"== {name}: 문제 {n_all} · 완주 {n_fin} · 정답 {n_cor} · 궤적 {len(F)} · "
            f"공식 라벨 {len(lab_off)} (spurious {spur} = {spur / max(1, len(lab_off)):.0%}) · 비공식 {len(lab_un)} · "
            f"겹침 {len(both)}건 일치 {agree} ({agree / max(1, len(both)):.0%})")
        for tag, pids, ykey in (("공식", lab_off, "y_off"), ("비공식", lab_un, "y_un")):
            if len(pids) < 10:
                out(f"  [{tag}] 라벨 {len(pids)}개 — 10개 미만이라 평가 보류"); continue
            y = np.array([1 if F[p][ykey] else 0 for p in pids])
            if y.min() == y.max():
                out(f"  [{tag}] 라벨이 한 쪽뿐({y.sum()}/{len(y)}) — 평가 불가"); continue
            T = np.array([F[p]["logT"] for p in pids])
            NC = np.array([F[p]["n_chunks"] for p in pids], float)
            UF = np.array([F[p].get("unfin", 0.0) for p in pids], float)          # 지휘 #37-2: 미완주율 통제
            ctrls = [T, NC] + ([UF] if UF.std() > 1e-9 else [])
            perm_ys = [rng.permutation(y) for _ in range(perms)]
            out(f"  [{tag}] n={len(y)} robust {y.sum()} / spurious {len(y) - y.sum()}")
            out(f"    {'지표':14s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'잔차null':>13s} {'ρ(청크수)':>9s} {'ρ(logT)':>8s}  비고")
            for key, sign, note in KEYS:
                x = np.array([F[p][key] for p in pids], float)
                rs = resid(x, ctrls)
                for s_ in ((+1, -1) if sign == 0 else (sign,)):
                    a = roc_auc_score(y, s_ * x); ra = roc_auc_score(y, s_ * rs)
                    nl = [roc_auc_score(py, s_ * x) for py in perm_ys]; lo, hi = np.percentile(nl, [2.5, 97.5])
                    nl2 = [roc_auc_score(py, s_ * rs) for py in perm_ys]; lo2, hi2 = np.percentile(nl2, [2.5, 97.5])
                    f1 = "▲" if a > hi else ("▼" if a < lo else " "); f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
                    lbl = f"{key}{'(+)' if s_ > 0 else '(−)'}" if sign == 0 else key
                    out(f"    {lbl:14s} {a:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2} "
                        f"{spearmanr(x, NC).correlation:+9.2f} {spearmanr(x, T).correlation:+8.2f}  {note}")
    rep = HERE / "results" / "h10c_report.txt"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장 {rep}")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    evaluate(a.perms, a.only)
