"""#LEN-CAL 임계 생성 — 주최측 라벨의 사전확률에 맞춘다 (지휘 #FINAL-PLAN, 사전등록대로).

train = `train-main-v2` 의 **주최측 라벨 문제단위 33개**(robust 20 / spurious 13).
목표 = 예측 True 비율이 **그 라벨의 robust 비율 20/33 = 0.6061** 과 같아지는 임계.
규칙 = 점수(−log 단어수) **내림차순 k=round(0.6061×33)=20 번째와 21번째의 중점**.
동점이면 중점이 그 값이 되어 비율이 목표에서 벗어날 수 있다 → **벗어난 값을 그대로 적고
임계를 손대지 않는다**(사전등록).

라벨은 임계를 정하는 데만 쓰고 점수식에는 들어가지 않는다(점수는 텍스트만 본다).

사용: .venv/Scripts/python.exe analysis/build_lencal_artifacts.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "solutions/lencal"))
OUT = ROOT / "solutions/lencal/artifacts/lencal.json"
REPORT = ROOT / "experiments/results/lencal_artifacts.txt"


def main():
    import lencal_lib
    from h21_uncertainty import problems
    rows = problems()
    txt = [r["problem"] for r in rows]
    y = np.array([r["y"] for r in rows])
    n = len(y)
    target = float(y.mean())
    k = int(round(target * n))
    s = np.array(lencal_lib.scores(txt))
    order = np.argsort(-s, kind="stable")             # 점수 내림차순 (결정적)
    sv = s[order]
    thr = float((sv[k - 1] + sv[k]) / 2.0) if k < n else float(sv[-1] - 1e-9)
    pred = s >= thr
    tie = bool(sv[k - 1] == sv[k])

    art = dict(
        thr=thr, target_true_rate=target, k=k, n_train=n,
        n_robust=int(y.sum()),
        achieved_true_rate=float(pred.mean()),
        tie_at_boundary=tie,
        train_acc=float((pred == (y == 1)).mean()),
        boundary_scores=[float(sv[k - 1]), float(sv[k])] if k < n else None,
        rule=("thr = midpoint of the k-th and (k+1)-th scores in descending order, "
              "k = round(label robust rate x n). Score = -log(word count), identical to h15C."),
        note=("Pre-registered 2026-10-15 (#FINAL-PLAN). Calibration, not ranking: this score's AUC "
              "0.671 did not clear its shuffle-null upper bound 0.70 on these 33 problems "
              "(#ORG-TRAIN). Never tuned on leaderboard scores."),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, ensure_ascii=False, indent=2), encoding="utf-8")

    # h15C 와 점수식이 같은지 **수치로** 확인한다 (다르면 임계가 무의미하다)
    sys.path.insert(0, str(ROOT / "solutions/h15C"))
    import h15_lib
    s15 = np.array(h15_lib.scores(txt, "C"))
    mad = float(np.abs(s - s15).max())

    L = []

    def out(x=""):
        print(x, flush=True)
        L.append(x)

    out("#LEN-CAL 임계 생성 — 주최측 라벨 사전확률 교정")
    out(f"   train = train-main-v2 문제단위 라벨 {n}개 (robust {int(y.sum())} / "
        f"spurious {int((1 - y).sum())})")
    out(f"   목표 True 비율 = robust 비율 = {target:.4f} → k = round({target:.4f}×{n}) = {k}")
    out(f"   경계 점수 {art['boundary_scores']} · 동점 {tie}")
    out(f"   **임계 {thr:+.6f}** · 달성 True 비율 **{art['achieved_true_rate']:.4f}** "
        f"({int(pred.sum())}/{n}) · train 정확도 {art['train_acc']:.4f}")
    out(f"   h15C 점수식과 최대절대차 **{mad:.3e}** → {'같다' if mad == 0 else '**다르다 — 멈춰라**'}")
    out(f"   (참고) h15C 출하 임계 {h15_lib.load_art()['thr']['C']:+.6f} · "
        f"그 임계의 train True 비율 {float((s15 >= h15_lib.load_art()['thr']['C']).mean()):.4f}")
    flip = int((pred != (s15 >= h15_lib.load_art()["thr"]["C"])).sum())
    out(f"   두 임계가 train 에서 예측을 바꾸는 문제 수 **{flip}** / {n}")
    out(f"   단어 수 최소 {min(len(t.split()) for t in txt)} 중앙 "
        f"{sorted(len(t.split()) for t in txt)[n // 2]} 최대 {max(len(t.split()) for t in txt)}")
    out(f"   저장 {OUT.relative_to(ROOT).as_posix()}")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    if mad != 0:
        raise SystemExit("점수식이 h15C 와 다르다 — 아티팩트를 쓰지 말 것")


if __name__ == "__main__":
    main()
