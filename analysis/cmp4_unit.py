"""#CMP4 분류기 단위검사 — **합성 데이터만.** 실제 라벨·특징을 쓰지 않는다.

왜 필요한가: #SAMP-FEAT 에서 `ast.parse` 는 통과하지만 런타임에 틀리는 버그로 파드가 1분에
죽었다($0.43). 분류기 넷은 전부 새 코드이므로 **의도대로 도는지 숫자로 먼저 보인다.**

**반드시 홀드아웃으로 재야 한다 (1차 구현이 여기서 틀렸다).** train·test 를 같은 행으로 주면
차원이 행보다 많아(d > n) 어떤 라벨링이든 완전 분리되고, 그래서 **라벨을 섞어도 AUC 1.000** 이
나왔다 — 검사가 아무것도 못 가르는 상태였다. 지금은 절반으로 적합해 나머지 절반을 채점한다.

검사 셋 (전부 통과해야 평가를 돌린다)
1. **분리 가능한 데이터에서 셋 다 분리한다** — 홀드아웃 AUC ≥ 0.9.
2. **라벨을 섞으면 0.5 근처** — 우연을 맞히는 코드가 아니다(0.35~0.65).
3. **D 만 순서를 본다** ← 가장 중요. 전구간 평균은 **같고** '신호가 든 구간' 만 다른 데이터를
   만들면 전구간 평균을 보는 B·C 는 우연이어야 하고, 위치 바인딩을 쓰는 D 는 분리해야 한다.
   이게 D 의 설계 주장(A·B·C 는 순서를 못 본다)을 직접 입증한다.

사용: .venv/Scripts/python.exe analysis/cmp4_unit.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from cmp4_eval import NSEG, fs_hdc, fs_massmean, prep  # noqa: E402

D_FEAT, NPER, REPS = 64, 20, 5
WAYS = (("B mass-mean", "raw", fs_massmean), ("C HDC", "hdc", fs_hdc),
        ("D HDC-seg", "hdcseg", fs_hdc))


def heldout(kind, feats, segs, y, layers, fit, tr, te):
    """절반으로 적합해 나머지 절반을 채점한다. 전처리는 `prep` 이 train 으로만 적합한다."""
    plan = {"outer": [(tr, te)], "inner": [[]], "proj": {}}
    pl = prep(kind, plan, feats, segs, layers)
    Ztr, Zte = pl["proj"][(0, layers[0])]
    return roc_auc_score(y[te], fit(Ztr, y[tr], Zte, None))


def split(n, rng):
    idx = rng.permutation(n)
    return np.sort(idx[: n // 2]), np.sort(idx[n // 2:])


def main():
    y = np.array([1] * NPER + [0] * NPER)
    n, layers, ok = len(y), [1], True

    # ---- 1·2: 평균이 벌어진 데이터 (모든 구간이 같다 = 순서 정보 없음)
    got = {nm: ([], []) for nm, *_ in WAYS}
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        X = rng.standard_normal((n, D_FEAT))
        X[y == 1, :8] += 2.0
        feats, segs = {1: X}, {1: np.repeat(X[:, None, :], NSEG, axis=1)}
        tr, te = split(n, rng)
        yp = rng.permutation(y)
        for nm, kind, fit in WAYS:
            got[nm][0].append(heldout(kind, feats, segs, y, layers, fit, tr, te))
            got[nm][1].append(heldout(kind, feats, segs, yp, layers, fit, tr, te))
    for nm, *_ in WAYS:
        a1, a2 = float(np.mean(got[nm][0])), float(np.mean(got[nm][1]))
        good = a1 >= 0.9 and 0.35 <= a2 <= 0.65
        ok &= good
        print(f"1·2 {nm:<12} 홀드아웃 AUC {a1:.3f} · 라벨 섞음 {a2:.3f}  "
              f"{'OK' if good else '**FAIL**'}   (기준: 분리 ≥0.9 · 섞음 0.35~0.65)")

    # ---- 3: 전구간 평균은 같고 '신호가 든 구간' 만 다르다
    #  클래스 1 = 앞 절반 구간에 +신호·뒤 절반에 −신호 / 클래스 0 = 그 반대.
    #  구간을 평균하면 상쇄되므로 전구간 평균은 두 클래스가 (거의) 같다.
    bump = np.zeros((NSEG, D_FEAT))
    bump[: NSEG // 2, :8] = +3.0
    bump[NSEG // 2:, :8] = -3.0
    r = {nm: [] for nm, *_ in WAYS}
    gaps = []
    for rep in range(REPS):
        rng = np.random.default_rng(200 + rep)
        S = rng.standard_normal((n, NSEG, D_FEAT)) * 0.3
        S[y == 1] += bump
        S[y == 0] -= bump
        W = S.mean(1)
        gaps.append(float(np.abs(W[y == 1].mean(0) - W[y == 0].mean(0)).max()))
        tr, te = split(n, rng)
        for nm, kind, fit in WAYS:
            r[nm].append(heldout(kind, {1: W}, {1: S}, y, layers, fit, tr, te))
    print(f"\n3 설계검사 · 전구간 평균의 클래스 간 최대 차 {np.mean(gaps):.3f} (0 에 가까워야 한다)")
    m = {nm: float(np.mean(v)) for nm, v in r.items()}
    for nm, *_ in WAYS:
        print(f"  {nm:<12} 홀드아웃 AUC {m[nm]:.3f}")
    cond = m["D HDC-seg"] >= 0.9 and m["C HDC"] <= 0.75 and m["B mass-mean"] <= 0.75
    ok &= cond
    print(f"  → D 만 순서를 본다: {'OK' if cond else '**FAIL**'} (D ≥0.9 · B·C ≤0.75)")

    print("\n" + ("**전부 통과**" if ok else "**실패 — 평가를 돌리지 말 것**"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
