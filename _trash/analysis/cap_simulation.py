"""토큰 상한 시뮬레이션 — H14 저장 데이터만으로, GPU 0. (지휘 #53-2)

H14 커널은 공식 33문제 × (원본 + rename 2종) × k=2 를 **max_new 16384** 로 생성하고
샘플별 **생성 토큰 수**와 **추출된 답**을 저장했다. 그래서 더 낮은 상한 c 를 사후에 흉내낼 수 있다:
  "토큰 수 > c 였다면 그 샘플은 상한에 걸려 답을 못 냈다" → **오답으로 센다**(개정 #37 규칙과 동일).
  acc(c) = (토큰 수 <= c 이면서 의사정답과 일치한 샘플 수) / k
  rename drop(c) = acc_orig(c) − min(acc_rename(c), acc_rename2(c))
의사정답 = 원본 샘플들의 다수결 답(gold 미사용). 원본에서 답을 하나도 못 뽑은 문제는 제외한다.
부호 +: drop 이 클수록 spurious → robust 점수 = −drop. 공식 문제 단위 라벨 · null 1000.

이 표가 말해 주는 것: **상한을 어디에 두느냐가 라벨 신호를 얼마나 바꾸는가.**
사용:  .venv/Scripts/python.exe analysis/cap_simulation.py [--perms 1000]
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CAPS = (4096, 6144, 8192, 12288, 16384)
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    hits = sorted(glob.glob(str(ROOT / "experiments/results/h14/**/*_h14.json"), recursive=True))
    if not hits:
        print("H14 결과 없음"); return
    rec = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    rows = []
    for r in rec:
        bk = r["by_kind"]
        ans = [x for x in bk["orig"]["answers"] if x is not None]
        if not ans:
            continue
        gold = Counter(ans).most_common(1)[0][0]
        kinds = ["rename", "rename2"]
        if all(bk[k]["degenerate"] for k in kinds):
            continue
        rows.append({"y": r["y"], "gold": gold, "bk": bk,
                     "kinds": [k for k in kinds if not bk[k]["degenerate"]],
                     "words": len(r["problem"].split())})
    y = np.array([r["y"] for r in rows])
    lw = np.log(np.array([max(1, r["words"]) for r in rows], float))
    rng = np.random.default_rng(0)
    perm_ys = [rng.permutation(y) for _ in range(a.perms)]
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    def acc(kind_blk, gold, cap):
        toks, answers = kind_blk["tokens"], kind_blk["answers"]
        k = len(toks)
        return sum(1 for t, v in zip(toks, answers) if t <= cap and v == gold) / max(1, k)

    out(f"토큰 상한 시뮬레이션 (H14 저장 데이터, GPU 0) · 문제 {len(y)} (robust {int(y.sum())} / spurious {int(len(y) - y.sum())}) · "
        f"null {a.perms}회 · 부호 +(하락 클수록 spurious)")
    out(f"{'cap':>6s} {'원본 acc 평균':>12s} {'drop 평균':>10s} {'AUC':>6s} {'null95%':>13s} {'잔차AUC':>8s} {'상한초과 샘플비':>14s}")
    for cap in CAPS:
        drops, bases, over, tot = [], [], 0, 0
        for r in rows:
            b = acc(r["bk"]["orig"], r["gold"], cap)
            v = min(acc(r["bk"][k], r["gold"], cap) for k in r["kinds"])
            drops.append(b - v); bases.append(b)
            for k in ["orig"] + r["kinds"]:
                over += sum(1 for t in r["bk"][k]["tokens"] if t > cap)
                tot += len(r["bk"][k]["tokens"])
        d = np.array(drops)
        auc = roc_auc_score(y, -d)
        nl = [roc_auc_score(p, -d) for p in perm_ys]
        lo, hi = np.percentile(nl, [2.5, 97.5])
        ra = roc_auc_score(y, -resid(d))
        flag = "▲" if auc > hi else ("▼" if auc < lo else " ")
        out(f"{cap:6d} {np.mean(bases):12.3f} {d.mean():10.3f} {auc:6.3f} [{lo:.2f},{hi:.2f}]{flag} {ra:8.3f} {over / max(1, tot):14.2f}")
    out()
    out("읽는 법: 상한이 낮을수록 '상한 초과 = 오답'이 늘어 drop 이 커진다. AUC 가 상한에 따라 크게 움직이면")
    out("        그 신호는 추론 예산의 함수이지 문제의 성질이 아니다.")
    (ROOT / "experiments" / "results" / "cap_simulation_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
