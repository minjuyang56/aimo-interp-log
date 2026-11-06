"""H20 '원본만 풀기' 신호 + H15 외부검증 2 (baselines agg). GPU 0. 사전등록 2026-10-02 (지휘 #55).

H20 — 변형을 만들지 않고 **원본 풀이만** 본다(테스트타임 비용 1/5). H14 커널이 저장한 원본 샘플의
      토큰 수·답으로 계산한다. 원본이 max_new(16384)에 닿았는지는 토큰 수로 판정한다(상한에 닿으면 미완주).
  H20a 평균 생성 토큰 수   — 길수록 spurious (**주지표**, 점수 = −토큰수)
  H20b 완주율             — 낮을수록 spurious (점수 = +완주율)
  H20c k개 답 불일치율     — 높을수록 spurious (점수 = −불일치율)
  상한 시뮬: cap ∈ {4096, 8192, 16384} 에서 토큰 수는 min(t, cap), 완주는 t < cap 으로 다시 계산.

H15ext2 — baselines `data/math-robust-agg.csv` (주최측 생성, 137문제, qwen3-8b 1종).
  라벨 정의는 실행 시 확인해 출력한다. 재학습 없이 n_sym·d_sym·log(단어수) 의 AUC 만 본다.

사용:  .venv/Scripts/python.exe analysis/h20_original_only.py [--perms 1000]
"""

import argparse
import glob
import io
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))
from h15_rename_density import features  # noqa: E402

AGG_URL = "https://raw.githubusercontent.com/aimo-interp/baselines/main/data/math-robust-agg.csv"
MAX_NEW = 16384


def band(y, sc, perm_ys):
    auc = roc_auc_score(y, sc)
    nl = [roc_auc_score(p, sc) for p in perm_ys]
    lo, hi = np.percentile(nl, [2.5, 97.5])
    return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    # ---------------------------------------------------------------- H20
    hits = sorted(glob.glob(str(ROOT / "experiments/results/h14/**/*_h14.json"), recursive=True))
    rec = json.loads(Path(hits[-1]).read_text(encoding="utf-8")) if hits else []
    rows = []
    for r in rec:
        o = r["by_kind"]["orig"]
        toks, ans = o["tokens"], o["answers"]
        if not toks:
            continue
        rows.append({"y": r["y"], "toks": toks, "ans": ans, "words": len(r["problem"].split())})
    y = np.array([r["y"] for r in rows])
    lw = np.log(np.array([max(1, r["words"]) for r in rows], float))
    perm_ys = [rng.permutation(y) for _ in range(a.perms)]

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    def metrics_at(cap):
        mt = np.array([np.mean([min(t, cap) for t in r["toks"]]) for r in rows], float)
        fin = np.array([np.mean([t < cap for t in r["toks"]]) for r in rows], float)
        dis = []
        for r in rows:
            c = Counter("None" if v is None else v for v in r["ans"])
            dis.append(1 - c.most_common(1)[0][1] / len(r["ans"]))
        return mt, fin, np.array(dis, float)

    out(f"H20 '원본만 풀기' (H14 저장 데이터, GPU 0) · 문제 {len(y)} (robust {int(y.sum())} / spurious {int(len(y) - y.sum())}) · "
        f"null {a.perms}회 · 부호: 길수록·미완주일수록·불일치할수록 spurious")
    out(f"{'지표':10s} {'cap':>6s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'robust평균':>11s} {'spurious평균':>12s}")
    mt, fin, dis = metrics_at(MAX_NEW)
    for tag, x, sign in (("H20a 토큰", mt, -1), ("H20b 완주율", fin, +1), ("H20c 불일치", dis, -1)):
        auc, lo, hi, f1 = band(y, sign * x, perm_ys)
        ra = roc_auc_score(y, sign * resid(x))
        out(f"{tag:10s} {MAX_NEW:6d} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} {x[y == 1].mean():11.1f} {x[y == 0].mean():12.1f}")
    for cap in (4096, 8192):
        mt_c, fin_c, _ = metrics_at(cap)
        for tag, x, sign in (("H20a 토큰", mt_c, -1), ("H20b 완주율", fin_c, +1)):
            auc, lo, hi, f1 = band(y, sign * x, perm_ys)
            ra = roc_auc_score(y, sign * resid(x))
            out(f"{tag:10s} {cap:6d} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} {x[y == 1].mean():11.1f} {x[y == 0].mean():12.1f}")

    # ---------------------------------------------------------------- H15 외부검증 2
    import pandas as pd
    b = urllib.request.urlopen(urllib.request.Request(AGG_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=180).read()
    df = pd.read_csv(io.BytesIO(b))
    g = df.groupby("problem_id").agg(decay=("absolute_accuracy_decay", "max"),
                                     base=("base_accuracy", "max"),
                                     problem=("original_problem", "first")).reset_index()
    out()
    out(f"H15 외부검증 2 — baselines math-robust-agg.csv · 행 {len(df)} → 문제 {len(g)} · 모델 1종(qwen3-8b)")
    out(f"  라벨 정의(확인): **absolute_accuracy_decay == 0 이면 robust**, 아니면 spurious. "
        f"decay 는 0 아니면 **>= {df[df.absolute_accuracy_decay > 0].absolute_accuracy_decay.min():.2f}** 로 중간값이 없다. "
        f"robust {int((g.decay == 0).sum())} / spurious {int((g.decay > 0).sum())} · base_accuracy 중앙 {df.base_accuracy.median():.2f}")
    y2 = (g.decay == 0).astype(int).values
    F2 = [features(t) for t in g.problem]
    lw2 = np.log(np.array([max(1, f["words"]) for f in F2], float))
    perm2 = [rng.permutation(y2) for _ in range(a.perms)]

    def resid2(x):
        X = np.column_stack([np.ones(len(x)), lw2])
        bb, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ bb

    out(f"{'지표':10s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'robust평균':>11s} {'spurious평균':>12s}")
    for key in ("n_sym", "d_sym"):
        x = np.array([f[key] for f in F2], float)
        auc, lo, hi, f1 = band(y2, -x, perm2)
        ra = roc_auc_score(y2, -resid2(x))
        out(f"{key:10s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {ra:8.3f} {x[y2 == 1].mean():11.3f} {x[y2 == 0].mean():12.3f}")
    auc, lo, hi, f1 = band(y2, -lw2, perm2)
    out(f"{'log단어수':10s} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {'':>8s} {np.exp(lw2)[y2 == 1].mean():11.1f} {np.exp(lw2)[y2 == 0].mean():12.1f}")
    out(f"  한계: 문제셋이 MATH(단어 중앙 {int(np.median(np.exp(lw2)))})라 본선 올림피아드급과 다르고, 모델도 1종이며, "
        f"이 라벨은 'domain 변형이 짧은 문제를 무너뜨린' 길이 아티팩트가 이미 확인된 셋이다.")
    (ROOT / "experiments" / "results" / "h20_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
