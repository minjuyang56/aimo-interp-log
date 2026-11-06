"""H23 암기 신호가 **길이의 대리**임을 부호 반전으로 보인다. GPU 0. (지휘 #H23)

왜 이 스크립트가 따로 있나
  "길이 잔차를 빼면 AUC 가 떨어진다" 는 지금까지의 증거는 약하다 — 잔차화가 신호를 같이 깎았을
  수도 있다는 반론이 늘 가능하다. 더 센 증거는 **예측과 그 검증**이다:

    암기 신호(nll·mink20)가 길이의 대리에 불과하다면, **길이-라벨 방향이 반대인 데이터셋에서는
    그 신호의 부호도 반대가 되어야 한다.**

  두 판이 그 시험을 제공한다.
    * 공식 33 (train-main-v2, 올림피아드, 단어 중앙 52): 알려진 방향 = **길수록 spurious**
      (CLAUDE.md: "길이만으로 LOPO bacc 0.61, 길수록 spurious")
    * 자체 88 (AIME 4 + MATH Level 5 84, 단어 중앙 31): 방향이 반대다.
  nll 은 길이와 ρ ≈ −0.65 다(긴 텍스트일수록 토큰당 NLL 이 낮다 — 일반 현상이고 H5 에서 이미 쟀다).
  따라서 nll 의 AUC 는 그 셋의 길이-라벨 방향을 **따라가야** 한다.

사용:  .venv/Scripts/python.exe analysis/h23_length_proxy.py
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "results" / "h23"
OUT = ROOT / "experiments" / "results" / "h23_length_proxy.txt"


def ranks(x):
    x = np.asarray(x, float)
    o = x.argsort(kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(1, len(x) + 1, dtype=float)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def auc(score, y):
    r = ranks(score)
    n1, n0 = float(y.sum()), float((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (float(r[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("H23 암기 신호 = 길이의 대리 — 부호 반전으로 보인다")
    out()
    out("예측: nll·mink20 이 길이의 대리라면, **길이-라벨 방향이 반대인 셋에서 그 신호도 반대**가 된다.")
    out()
    out(f"| 판 | n | robust | 단어수 중앙 | 길이→robust AUC | nll AUC | mink20 AUC | ρ(nll,len) |")
    out("|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = {}
    for f in sorted(SRC.glob("*_h23.json")):
        recs = json.loads(f.read_text(encoding="utf-8"))
        if len(recs) < 10:
            continue
        y = np.array([r["y"] for r in recs])
        if len(set(y.tolist())) < 2:
            continue
        lw = np.array([len(re.findall(r"[A-Za-z]+", r["problem"])) for r in recs], float)
        nll = np.array([r["nll"] for r in recs], float)
        mk = np.array([r["mink20"] for r in recs], float)
        tag = "자체 88 (AIME+L5)" if "own" in f.stem else "공식 33 (train-v2)"
        # 부호는 사전등록대로: robust 점수 = +nll, −mink20. 길이는 '짧을수록 robust' 를 양으로.
        a_len, a_nll, a_mk = auc(-np.log(lw), y), auc(nll, y), auc(-mk, y)
        rho = float(np.corrcoef(ranks(nll), ranks(lw))[0, 1])
        out(f"| {tag} | {len(y)} | {int(y.sum())} | {np.median(lw):.0f} | "
            f"**{a_len:.3f}** | **{a_nll:.3f}** | {a_mk:.3f} | {rho:+.2f} |")
        rows[tag] = (a_len, a_nll, a_mk, rho)
    out()
    if len(rows) == 2:
        (l1, n1_, m1, r1), (l2, n2_, m2, r2) = rows.values()
        flip_len = (l1 - 0.5) * (l2 - 0.5) < 0
        flip_nll = (n1_ - 0.5) * (n2_ - 0.5) < 0
        out(f"길이-라벨 방향이 두 판에서 **{'뒤집힌다' if flip_len else '같다'}** "
            f"({l1:.3f} vs {l2:.3f})")
        out(f"nll 방향도 **{'뒤집힌다' if flip_nll else '같다'}** ({n1_:.3f} vs {n2_:.3f}) · "
            f"길이 AUC 와의 차이 {abs(n1_ - l1):.3f} / {abs(n2_ - l2):.3f}")
        out()
        if flip_len and flip_nll:
            out("→ **예측이 맞았다.** 암기 신호는 길이의 대리다.")
            out("   이건 '잔차화하면 AUC 가 떨어진다' 보다 센 증거다 — 잔차화는 신호를 같이 깎았을")
            out("   수도 있다는 반론이 가능하지만, **부호가 길이 방향을 따라 뒤집히는 것**은 대리가 아니면")
            out("   설명되지 않는다. ρ(nll, len) 이 두 판에서 똑같이 −0.65 인 것도 같은 그림이다.")
            out()
            out("   H5(9/29)가 공식 33 에서 본 nll 0.66~0.70 은 '암기' 가 아니라 '그 셋에서 짧은 문제가")
            out("   robust 였다' 를 읽은 것이다. 암기 신호 계열(H5·H18·H19·H23)은 여기서 닫는다.")
        else:
            out("→ 예측이 어긋났다. 길이 대리 설명만으로는 부족하다 — 다른 설명을 찾아야 한다.")
    else:
        out("두 판이 모두 있어야 비교가 된다 (현재 " + str(len(rows)) + "판).")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
