"""H20 제출 아티팩트 — 상한 16384·8192 각각의 임계값을 공식 33문제에서 고정. (지휘 #56)

규칙(사전 고정): 임계값 = **두 클래스 중앙값의 중점**. train AUC 최적화가 아니다.
점수는 '원본 k 샘플의 평균 생성 토큰 수'이고 **클수록 spurious** 이므로 판정은
  mean_tokens <= thr  ->  True(robust)
상한 c 에서의 토큰 수는 min(t, c) 로 센다(그 예산이었다면 거기서 잘렸을 것이므로).
H15A 폴백 임계값도 같이 담는다(시간 부족 시 텍스트 규칙으로 전환).

사용:  .venv/Scripts/python.exe analysis/build_h20_artifacts.py
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))
from h15_rename_density import features, official_problems  # noqa: E402

CAPS = (16384, 8192)


def main():
    hits = sorted(glob.glob(str(ROOT / "experiments/results/h14/**/*_h14.json"), recursive=True))
    rec = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    rows = [{"y": r["y"], "toks": r["by_kind"]["orig"]["tokens"]} for r in rec if r["by_kind"]["orig"]["tokens"]]
    y = np.array([r["y"] for r in rows])
    art = {"n_train": int(len(y)), "instrument": "Skywork/Skywork-OR1-Math-7B", "k": len(rows[0]["toks"]),
           "decoding": {"temperature": 0.6, "top_p": 0.95, "seed": 0},
           "note": "thr = 두 클래스 중앙값의 중점 · mean_tokens <= thr 이면 True(robust)", "thr": {}, "train_fit": {}}
    for cap in CAPS:
        x = np.array([np.mean([min(t, cap) for t in r["toks"]]) for r in rows], float)
        thr = float((np.median(x[y == 1]) + np.median(x[y == 0])) / 2)
        pred = x <= thr
        art["thr"][str(cap)] = thr
        art["train_fit"][str(cap)] = {"acc": float((pred == (y == 1)).mean()), "pred_true_rate": float(pred.mean()),
                                      "median_robust": float(np.median(x[y == 1])), "median_spurious": float(np.median(x[y == 0]))}
        print(f"cap {cap:6d}: thr {thr:9.1f} · train 정확도 {art['train_fit'][str(cap)]['acc']:.3f} · "
              f"True 비율 {art['train_fit'][str(cap)]['pred_true_rate']:.2f} · "
              f"중앙값 robust {np.median(x[y == 1]):.0f} / spurious {np.median(x[y == 0]):.0f}")
    # H15A 폴백 (텍스트만): n_sym <= thr_nsym 이면 True
    probs = official_problems()
    ns = np.array([features(p["problem"])["n_sym"] for p in probs], float)
    yy = np.array([p["y"] for p in probs])
    thr_ns = float((np.median(ns[yy == 1]) + np.median(ns[yy == 0])) / 2)
    art["fallback_n_sym_thr"] = thr_ns
    print(f"폴백 H15A: n_sym <= {thr_ns:.1f} 이면 True · train 정확도 {float(((ns <= thr_ns) == (yy == 1)).mean()):.3f}")
    d = ROOT / "solutions" / "h20" / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    (d / "h20.json").write_text(json.dumps(art, indent=2), encoding="utf-8")
    print("저장", d / "h20.json")


if __name__ == "__main__":
    main()
