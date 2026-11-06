"""H23 기억 모드 평가 — 사전등록(지휘 #H23)대로 1회. GPU 0.

통과 기준 (측정 전 고정)
  Skywork 자체 라벨 88 에서 주지표 `mink20` 의 **log(단어수) 잔차 AUC** 가 셔플 null 95% 상단을
  넘을 때만 '기억 모드 신호가 있다' 고 적는다. **원점수만 넘고 잔차가 못 넘으면 '길이 교란의 재확인'** 이다.
  이 추가 기준을 둔 이유: 이 계열(H5·H18·H19)이 네 번 모두 길이 교란으로 죽었다(ρ(len) −0.44~−0.66).

부호 (고정, 뒤집지 않는다 — 전부 '익숙할수록 spurious')
  robust 점수 = −mink20(주지표) · +nll · +zlib · −d_mink
  보조로 H18 의 minkpp20(−), H5 의 nll_2nd(+)도 같은 파일에 있으니 병기한다.

사용:  .venv/Scripts/python.exe analysis/h23_eval.py
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "results" / "h23"
OUT = ROOT / "experiments" / "results" / "h23_report.txt"
PERMS, BOOT = 1000, 1000
#  지표 -> robust 점수의 부호 (사전등록)
SIGN = {"mink20": -1, "nll": +1, "zlib": +1, "d_mink": -1, "minkpp20": -1, "nll_2nd": +1}
PRIMARY = "mink20"
PRIOR = {  # 같은 지표의 기존 측정값 (나란히 적어 비교 가능하게)
    "nll": "H5 4모델 0.66~0.70 · 길이잔차 0.58~0.64",
    "mink20": "H5 Skywork null ▲ (풀링은 길이잔차에서 죽음)",
    "minkpp20": "H18 0.454 · 잔차 0.377",
    "zlib": "H5 0.50 (정확히 우연)",
    "d_mink": "H19 nll_gap 0.368 · 잔차 0.415 · 부호 반대 (통계량은 다름)",
}


def ranks(x):
    x = np.asarray(x, float)
    o = x.argsort(kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(1, len(x) + 1, dtype=float)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("H23 기억 모드(멤버십 추론) — 사전등록 평가 1회")
    out()
    out("부호 고정: robust 점수 = −mink20(주지표) · +nll · +zlib · −d_mink  (전부 '익숙할수록 spurious')")
    out("통과 기준: **주지표의 log(단어수) 잔차 AUC** 가 null 95% 상단을 넘어야 주장한다.")
    out("           원점수만 넘으면 '길이 교란의 재확인' 으로 적는다(이 계열이 네 번 그렇게 죽었다).")
    out()

    files = sorted(SRC.glob("*_h23.json"))
    if not files:
        out(f"측정 파일이 없다: {SRC}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for f in files:
        recs = json.loads(f.read_text(encoding="utf-8"))
        if len(recs) < 10:
            out(f"## {f.stem} — {len(recs)}건, 너무 적어 건너뜀")
            continue
        y = np.array([r["y"] for r in recs])
        if len(set(y.tolist())) < 2:
            out(f"## {f.stem} — 한 클래스뿐, 건너뜀")
            continue
        lw = np.log(np.array([max(1, len(re.findall(r"[A-Za-z]+", r["problem"]))) for r in recs], float))
        ntok = np.array([r["ntok"] for r in recs], float)
        rng = np.random.default_rng(0)
        PY = np.array([rng.permutation(y) for _ in range(PERMS)], float)
        boot = [rng.integers(0, len(y), len(y)) for _ in range(BOOT)]
        n1, n0 = float(y.sum()), float((1 - y).sum())

        def band(score):
            r = ranks(score)
            auc = (float(r[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
            nl = (PY @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
            lo, hi = np.percentile(nl, [2.5, 97.5])
            return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

        def auc_of(score, yy):
            r = ranks(score)
            a, b = float(yy.sum()), float((1 - yy).sum())
            if a == 0 or b == 0:
                return np.nan
            return (float(r[yy == 1].sum()) - a * (a + 1) / 2) / (a * b)

        def resid(x, basis):
            X = np.column_stack([np.ones(len(x)), basis])
            b, *_ = np.linalg.lstsq(X, np.asarray(x, float), rcond=None)
            return np.asarray(x, float) - X @ b

        out(f"## {f.stem} — {len(recs)}문제 (robust {int(n1)} / spurious {int(n0)})")
        out()
        out("| 지표 | 원점수 AUC | null 95% | 길이잔차 | 토큰수잔차 | 부트 CI | ρ(log단어수) | 기존 측정 |")
        out("|---|---:|---|---:|---:|---|---:|---|")
        verdict = {}
        for key in ("mink20", "nll", "zlib", "d_mink", "minkpp20", "nll_2nd"):
            vals = [r.get(key) for r in recs]
            if any(v is None for v in vals):
                keep = [i for i, v in enumerate(vals) if v is not None]
                if len(keep) < 10 or len(set(y[keep].tolist())) < 2:
                    out(f"| {key} | — | 값이 모자람({len(keep)}건) | | | | | {PRIOR.get(key,'')} |")
                    continue
                out(f"| {key} | (부분집합 {len(keep)}건) | | | | | | {PRIOR.get(key,'')} |")
                sub = np.array([vals[i] for i in keep], float) * SIGN[key]
                ysub, lwsub = y[keep], lw[keep]
                r = ranks(sub)
                a, b = float(ysub.sum()), float((1 - ysub).sum())
                auc = (float(r[ysub == 1].sum()) - a * (a + 1) / 2) / (a * b)
                PYs = np.array([rng.permutation(ysub) for _ in range(PERMS)], float)
                nl = (PYs @ r - a * (a + 1) / 2) / (a * b)
                lo, hi = np.percentile(nl, [2.5, 97.5])
                X = np.column_stack([np.ones(len(sub)), lwsub])
                bb, *_ = np.linalg.lstsq(X, sub, rcond=None)
                rs = ranks(sub - X @ bb)
                raucv = (float(rs[ysub == 1].sum()) - a * (a + 1) / 2) / (a * b)
                out(f"|   ↳ {key} | {auc:.3f} | [{lo:.2f}, {hi:.2f}] | {raucv:.3f} | | | | |")
                verdict[key] = (auc, lo, hi, raucv, hi)
                continue
            s = np.array(vals, float) * SIGN[key]
            auc, lo, hi, mk = band(s)
            rauc, rlo, rhi, rmk = band(resid(s, lw))
            tauc, tlo, thi, tmk = band(resid(s, np.log(ntok)))
            bl, bh = np.percentile([auc_of(s[i], y[i]) for i in boot], [2.5, 97.5])
            rho = float(np.corrcoef(ranks(s), ranks(lw))[0, 1])
            out(f"| {'**' + key + '**' if key == PRIMARY else key} | {auc:.3f} {mk} | "
                f"[{lo:.2f}, {hi:.2f}] | {rauc:.3f} {rmk} | {tauc:.3f} {tmk} | "
                f"[{bl:.2f}, {bh:.2f}] | {rho:+.2f} | {PRIOR.get(key, '')} |")
            verdict[key] = (auc, lo, hi, rauc, rhi)
        out()
        if PRIMARY in verdict:
            auc, lo, hi, rauc, rhi = verdict[PRIMARY]
            if rauc > rhi:
                out(f"→ **통과.** 주지표 {PRIMARY} 의 길이잔차 {rauc:.3f} > null 상단 {rhi:.2f} — "
                    "길이를 통제해도 남는다.")
            elif auc > hi:
                out(f"→ **길이 교란의 재확인.** 원점수 {auc:.3f} 는 null({hi:.2f})을 넘었지만 "
                    f"길이잔차 {rauc:.3f} 가 상단 {rhi:.2f} 을 못 넘었다 → 길이의 대리다. "
                    "H5 에서 본 것과 같은 양식이다.")
            else:
                out(f"→ **신호 없음.** 주지표 원점수 {auc:.3f} 가 null [{lo:.2f}, {hi:.2f}] 안이다.")
        # 대조 라벨
        yf = [r.get("y_fin") for r in recs]
        if all(v is not None for v in yf) and len(set(yf)) > 1:
            yfa = np.array(yf)
            s = np.array([r[PRIMARY] for r in recs], float) * SIGN[PRIMARY]
            out(f"   대조 라벨 label_fin 의 {PRIMARY} AUC {auc_of(s, yfa):.3f} "
                f"(robust {int(yfa.sum())} / spurious {int((1 - yfa).sum())}) — "
                "0.5 미만으로 뒤집히면 라벨 정의에 붙은 신호다")
        out()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
