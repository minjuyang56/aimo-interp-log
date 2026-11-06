"""H21 결합 지표 — **사후 탐색이다(사전등록 아님).** GPU 0.

왜 사후인지 명시: H21a·b·c 의 모델별 결과를 **이미 본 뒤**에 "drop 이 제일 세고 엔트로피는 4모델 방향이 같다" 를
보고 만든 조합이다. 그래서 여기 숫자는 가설 검정이 아니라 **설계 후보 비교**로만 읽어야 한다.
제출물에 쓰려면 이 조합을 고정한 채 새 데이터(리더보드)에서 확인하는 것이 유일한 정직한 경로다.

조합: combo = z(drop) + z(−mean_entropy)  (모델별 표준화, 가중치 학습 없음, 큼 = robust)
  drop         = logp(즉답|원본) − logp(즉답|상수변형)   [H21c 주지표, 큼 = robust]
  mean_entropy = 1024토큰 구간 평균 엔트로피            [H21a 주지표, 큼 = spurious]

사용:  .venv/Scripts/python.exe analysis/h21_combo.py [--perms 1000]
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H21 = ROOT / "experiments" / "results" / "h21"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    conf, gen = {}, {}
    for stage, store in (("conf", conf), ("gen", gen)):
        for f in sorted(glob.glob(str(H21 / f"*_{stage}.json"))):
            store[Path(f).name.replace(f"_{stage}.json", "")] = json.loads(Path(f).read_text(encoding="utf-8"))
    aliases = sorted(set(conf) & set(gen))
    if not aliases:
        print("conf·gen 양쪽이 다 있는 모델이 없다.")
        return
    probs = [r["problem"] for r in conf[aliases[0]]]
    y = np.array([r["y"] for r in conf[aliases[0]]])
    lw = np.log(np.array([max(1, len(re.findall(r"[A-Za-z]+", p))) for p in probs], float))
    rng = np.random.default_rng(0)
    PY = np.array([rng.permutation(y) for _ in range(a.perms)], float)
    n1, n0 = float(y.sum()), float((1 - y).sum())

    def ranks(x):
        o = np.asarray(x, float).argsort(kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(1, len(x) + 1, dtype=float)
        return r

    def band(score):
        r = ranks(score)
        auc = (float(r[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
        nl = (PY @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, np.asarray(x, float), rcond=None)
        return np.asarray(x, float) - X @ b

    def z(x):
        x = np.asarray(x, float)
        m = np.nanmean(x)
        x = np.where(np.isnan(x), m, x)
        return (x - m) / (np.nanstd(x) + 1e-9)

    lines = [f"H21 결합 (사후 탐색) · 모델 {len(aliases)}종 · 문제 {len(y)} "
             f"(robust {int(y.sum())}/spurious {int(len(y) - y.sum())}) · null {a.perms}회",
             f"{'모델':26s} {'drop':>18s} {'−entropy':>18s} {'combo':>18s} {'combo 잔차':>10s}"]
    combos = []
    for al in aliases:
        d = np.array([np.nan if r.get("drop") is None else float(r["drop"]) for r in conf[al]])
        e = np.array([float(r["mean_entropy"]) for r in gen[al]])
        c = z(d) + z(-e)
        combos.append(c)
        bd, be, bc = band(np.nan_to_num(d, nan=np.nanmean(d))), band(-e), band(c)
        rc = band(resid(c))
        lines.append(f"{al:26s} {bd[0]:6.3f} [{bd[1]:.2f},{bd[2]:.2f}]{bd[3]} "
                     f"{be[0]:6.3f} [{be[1]:.2f},{be[2]:.2f}]{be[3]} "
                     f"{bc[0]:6.3f} [{bc[1]:.2f},{bc[2]:.2f}]{bc[3]} {rc[0]:10.3f}")
    pool = np.mean(combos, axis=0)
    bp, rp = band(pool), band(resid(pool))
    lines += ["", f"모델 평균 combo: AUC {bp[0]:.3f} [{bp[1]:.2f},{bp[2]:.2f}]{bp[3]} · 길이잔차 {rp[0]:.3f}",
              "  (모델 평균은 문제 단위 점수다 — 모델별 균형 검증셋에서는 구조적으로 ~0.50. 참고용.)",
              "", "모델 간 combo 순위상관 (Spearman)"]
    for i, ai in enumerate(aliases):
        row = []
        for j, aj in enumerate(aliases):
            ri, rj = ranks(combos[i]), ranks(combos[j])
            row.append(f"{np.corrcoef(ri, rj)[0, 1]:14.3f}")
        lines.append(f"   {ai:26s}" + "".join(row))
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "experiments" / "results" / "h21_combo_report.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
