"""compass 아티팩트 생성 — train v2 (Kaggle H10-B 출력) 에서 모델별 mu · z 통계 · 고정 임계값. (지휘 #26-2)

임계값 = 모델별 combo 분포의 **사전지정 분위수 Q** (기본 Q = 1 − robust 비율(train v2 라벨 20/33 ≈ 0.61) = 0.39 분위 →
combo 상위 61%를 True). 중앙값 분할 아님, 리더보드로 튜닝 안 함. 라벨은 임계값 계산에 쓰지 않는다(분위수만).
사용:  .venv/Scripts/python.exe solutions/compass/build_artifacts.py [--q 0.39]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from h10b_eval import load_all, metrics  # noqa: E402
from h10b_ext import ALIAS2ID  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--q", type=float, default=0.39); a = ap.parse_args()
    table = {}
    for alias, store in load_all().items():
        model_id = ALIAS2ID.get(alias)
        if model_id is None or len(store) < 30:
            print(f"skip {alias}: {len(store)}문제"); continue
        allc = np.concatenate([np.asarray(r["chunks"], np.float32)[:, 0] for r in store]); mu = allc.mean(0)
        feats = [metrics(np.asarray(r["chunks"], np.float32)[:, 0] - mu) for r in store]
        cm = np.array([f["compass_mean"] for f in feats]); tr = np.array([f["traced_ratio"] for f in feats]); cv = np.array([f["curvature"] for f in feats])
        z = {"cm": (float(cm.mean()), float(cm.std() + 1e-9)), "tr": (float(tr.mean()), float(tr.std() + 1e-9)), "cv": (float(cv.mean()), float(cv.std() + 1e-9))}
        combo = (cm - z["cm"][0]) / z["cm"][1] + (tr - z["tr"][0]) / z["tr"][1] - (cv - z["cv"][0]) / z["cv"][1]
        thr = float(np.quantile(combo, a.q))
        table[model_id] = {"mu": mu.astype(np.float32), "z": z, "threshold": thr, "n": len(store), "q": a.q}
        print(f"{model_id}: n={len(store)} combo 분위 {a.q} 임계값 {thr:+.3f} → train True 비율 {(combo >= thr).mean():.2f}")
    out = ROOT / "solutions" / "compass" / "artifacts" / "compass_thresholds.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, table=np.array(table, dtype=object))
    print("저장", out, "모델", list(table))


if __name__ == "__main__":
    main()
