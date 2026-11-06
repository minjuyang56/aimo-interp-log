"""H15 제출 아티팩트 생성 — 길이 잔차 계수와 임계값을 공식 train 33문제에서 **고정**. (지휘 #49-2)

임계값 규칙(사전 고정): **두 클래스 점수 중앙값의 중점**. train AUC 를 최대화하는 값이 아니다.
  score 는 모두 '높을수록 robust' 이고 thr = (median(score|robust) + median(score|spurious)) / 2,
  판정은 score >= thr -> True.
잔차 계수: n_sym ~ a·log(단어수) + b 를 train 33문제로 최소제곱 적합 (라벨을 쓰지 않는다).

세 제출물(h15A·h15B·h15C)에 같은 아티팩트를 복사한다.
사용:  .venv/Scripts/python.exe analysis/build_h15_artifacts.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))
from h15_rename_density import features, official_problems  # noqa: E402


def main():
    probs = official_problems()
    F = [features(p["problem"]) for p in probs]
    y = np.array([p["y"] for p in probs])
    ns = np.array([f["n_sym"] for f in F], float)
    lw = np.log(np.array([f["words"] for f in F], float))
    A = np.column_stack([lw, np.ones(len(lw))])
    (slope, intercept), *_ = np.linalg.lstsq(A, ns, rcond=None)
    resid = ns - (slope * lw + intercept)
    art = {"slope": float(slope), "intercept": float(intercept), "n_train": int(len(y)),
           "thr": {}, "note": "thr = 두 클래스 점수 중앙값의 중점 (train 고정, AUC 최적화 아님)"}
    for mode, x in (("A", ns), ("B", resid), ("C", lw)):
        s = -x                                            # 높을수록 robust
        thr = float((np.median(s[y == 1]) + np.median(s[y == 0])) / 2)
        art["thr"][mode] = thr
        pred = s >= thr
        art.setdefault("train_fit", {})[mode] = {
            "acc": float((pred == (y == 1)).mean()), "pred_true_rate": float(pred.mean()),
            "median_robust": float(np.median(s[y == 1])), "median_spurious": float(np.median(s[y == 0]))}
    print(f"잔차 계수: n_sym = {slope:.3f}·log(단어수) + {intercept:.3f}  (train {len(y)}문제, 라벨 미사용)")
    for mode in ("A", "B", "C"):
        t = art["train_fit"][mode]
        print(f"  {mode}: thr {art['thr'][mode]:+.4f} · train 정확도 {t['acc']:.3f} · True 비율 {t['pred_true_rate']:.2f} "
              f"· 중앙값 robust {t['median_robust']:+.3f} / spurious {t['median_spurious']:+.3f}")
    for name in ("h15A", "h15B", "h15C"):
        d = ROOT / "solutions" / name / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        (d / "h15.json").write_text(json.dumps(art, indent=2), encoding="utf-8")
    print("아티팩트 3곳에 저장 완료")


if __name__ == "__main__":
    main()
