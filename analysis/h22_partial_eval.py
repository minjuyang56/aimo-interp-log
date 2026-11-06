"""H22 부분 검증 — 저장된 H14 생성결과로 공식 33문제에서 점수 규칙을 돌려본다. GPU 0. (지휘 #H22-D)

**이것은 사전등록한 H22 평가가 아니다.** H22 의 설계는 원본 N=8 · 섭동 K=8(6유형) · 섭동당 S=4 인데,
여기서 쓸 수 있는 데이터는 H14 커널이 남긴 것뿐이라 다음으로 **쪼그라든다**:
    N = 2        (설계 8)
    K = 2        (설계 8) — 게다가 **rename 계열 둘뿐**이다. typos·distract·rephrase 가 없다.
    S = 2        (설계 4)
    cap = 16384  (설계와 같음)
따라서 여기서 나오는 AUC 는 **H22 의 성능 추정이 아니고**, (a) 점수 코드가 정의대로 도는지,
(b) 가장 약한 칸에서도 방향이 맞는지를 보는 것이다. 숫자를 H22 의 결과로 인용하면 안 된다.

왜 그래도 할 가치가 있나: H21cK 가 알려준 교훈이 '단일 변형 지표는 변형 선택에 따라 ±0.09 흔들린다' 였다.
rename 계열만 2개 쓴 이 측정이 null 을 못 넘는 것은 **예상된 결과**이고, 넘으면 오히려 운을 의심해야 한다.

의사정답(pseudo-gold)의 한계가 여기서 특히 크다: N=2 라 두 샘플이 갈리면 과반이 없다.
그 경우를 **따로 세서 보고**한다(설계의 N=8 에서는 드물어진다).

사용:  .venv/Scripts/python.exe analysis/h22_partial_eval.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "results" / "h14" / "h14" / "Skywork-OR1-Math-7B_h14.json"
OUT = ROOT / "experiments" / "results" / "h22_partial_report.txt"
PERMS, BOOT = 1000, 1000
PERT_KINDS = ("rename", "rename2")


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    recs = json.loads(SRC.read_text(encoding="utf-8"))
    out("H22 부분 검증 — 저장된 H14 데이터 (Skywork · cap 16384 · 공식 문제단위 라벨 33)")
    out()
    out(f"출처: {SRC.relative_to(ROOT)} · 레코드 {len(recs)}건")
    out("**축소된 설정**: N=2(설계 8) · K=2(설계 8, rename 계열만) · S=2(설계 4) · cap 16384(동일)")
    out("→ 이 숫자는 H22 의 성능이 아니다. 점수 규칙이 정의대로 도는지와 방향만 본다.")
    out()

    y, probs, scores, diags, no_major = [], [], [], [], 0
    for r in recs:
        bk = r.get("by_kind") or {}
        oa = (bk.get("orig") or {}).get("answers") or []
        pa = [(bk.get(k) or {}).get("answers") for k in PERT_KINDS]
        pa = [a for a in pa if a]
        if not oa or not pa:
            continue
        cnt = Counter(a for a in oa if a is not None)
        if not cnt:
            continue
        top = cnt.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            no_major += 1                       # N=2 에서 두 샘플이 갈린 경우 — 설계의 N=8 에서는 드물다
        gold, hits = top[0]
        stab = hits / len(oa)
        broken = sum(1 for a in pa
                     if (Counter(x for x in a if x is not None).most_common(1) or [(None, 0)])[0][0] != gold)
        bfrac = broken / len(pa)
        y.append(int(r["y"]))
        probs.append(r["problem"])
        scores.append(stab * bfrac)
        diags.append((gold, stab, broken, len(pa)))

    y = np.array(y)
    s = np.array(scores, float)
    out(f"평가 대상 {len(y)}문제 (robust {int(y.sum())} / spurious {int((1 - y).sum())})")
    out(f"원본 2샘플이 갈려 과반이 없던 문제: **{no_major}건** — 의사정답이 불안정한 몫이다")
    out()

    lw = np.log(np.array([max(1, len(re.findall(r"[A-Za-z]+", p))) for p in probs], float))
    rng = np.random.default_rng(0)
    PY = np.array([rng.permutation(y) for _ in range(PERMS)], float)
    boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(BOOT)]

    def ranks(x):
        x = np.asarray(x, float)
        o = x.argsort(kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(1, len(x) + 1, dtype=float)
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
        return r

    def auc_of(score, yy):
        r = ranks(score)
        n1, n0 = float(yy.sum()), float((1 - yy).sum())
        if n1 == 0 or n0 == 0:
            return np.nan
        return (float(r[yy == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)

    def band(score):
        r = ranks(score)
        n1, n0 = float(y.sum()), float((1 - y).sum())
        auc = (float(r[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
        nl = (PY @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, np.asarray(x, float), rcond=None)
        return np.asarray(x, float) - X @ b

    # 점수는 '클수록 spurious' 이므로 robust 쪽 AUC 를 보려면 부호를 뒤집는다 (부호 고정, 뒤집기 아님)
    rob = -s
    auc, lo, hi, mk = band(rob)
    bl, bh = np.percentile([auc_of(rob[i], y[i]) for i in boot_idx], [2.5, 97.5])
    rauc, rlo, rhi, rmk = band(resid(rob))
    out("## 주지표 score_spurious = P(원본 안정) × (섭동 과반 오답 비율)")
    out()
    out(f"  AUC(robust) **{auc:.3f}** {mk} · 셔플 null 95% [{lo:.2f}, {hi:.2f}] · 부트스트랩 CI [{bl:.2f}, {bh:.2f}]")
    out(f"  log(단어수) 잔차 AUC {rauc:.3f} {rmk} · null [{rlo:.2f}, {rhi:.2f}]")
    out()
    out(f"  점수 분포: robust 평균 {s[y == 1].mean():.3f} vs spurious 평균 {s[y == 0].mean():.3f}")
    out(f"  서로 다른 값 {len(set(s.tolist()))}개 (K=2·S=2 라 점수가 "
        f"{{0, 0.25, 0.5, 0.75, 1}} 격자에만 놓인다 = 해상도 한계)")
    out(f"  분포 {dict(sorted(Counter(np.round(s, 3).tolist()).items()))}")
    out()

    pred = s < 0.5
    acc = float((pred == (y == 1)).mean())
    out(f"## 절대 임계값 0.5 의 train 정확도 {acc:.3f} · True 예측 비율 {pred.mean():.2f}")
    out(f"  (상수 예측 기준선: 전부 True = {float((y == 1).mean()):.3f})")
    out()
    out("## 해석")
    out()
    if auc > hi:
        out("방향은 맞고 null 도 넘었다 — 다만 K=2·rename 계열만이라 **운을 의심해야 하는 쪽**이다")
        out("(H21cK: 단일 변형 지표는 변형 선택에 따라 ±0.09 흔들린다).")
    else:
        out("null 을 못 넘었다. **이건 예상된 결과다** — 설계의 1/4 칸(K 2/8 · S 2/4 · N 2/8)이고")
        out("섭동 유형이 rename 계열 둘뿐이라 '과반' 을 셀 분해능이 없다(점수가 5개 격자값만 가진다).")
    out("확인된 것은 점수 코드가 정의대로 돌고 미완주·추출실패가 오답으로 들어간다는 것이다.")
    out("판정에 쓸 숫자는 typos·distract·rephrase 를 포함한 K=8 측정에서만 나온다.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
