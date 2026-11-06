"""제출 A용 아티팩트 — DeepSeek-8B 층 18 프로브를 MATH137 + v2 33문제 전체(170문제)로 학습해 저장.

사전등록: 층 18 · C=0.05 · class_weight=balanced · 표준화는 학습셋 기준. 리더보드 보고 바꾸지 않는다.
계측기는 판정 대상 모델과 무관하게 DeepSeek-8B 하나(라벨이 문제 단위라는 H2 근거).

출력: solutions/inst_probe/artifacts/instrument_L18.npz  (layer, mu, sd, w, b, n_train)
사용:  .venv/Scripts/python.exe experiments/train_instrument.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LAYER, C = 18, 0.05
OUT = HERE.parent / "solutions" / "inst_probe" / "artifacts" / "instrument_L18.npz"


def main():
    d137 = np.load(HERE / "results" / f"math137_hidden_{alias(MODEL)}.npz", allow_pickle=True)
    dv2 = np.load(HERE / "results" / f"trainv2_hidden_{alias(MODEL)}.npz", allow_pickle=True)
    keep = dv2["y"] >= 0
    X = np.vstack([d137["H"][:, LAYER], dv2["H"][:, LAYER][keep]]).astype(np.float32)
    y = np.concatenate([d137["y"].astype(int), dv2["y"][keep].astype(int)])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit((X - mu) / sd, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, layer=LAYER, mu=mu, sd=sd, w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]),
             n_train=len(y), model_id=MODEL)
    margins = clf.decision_function((X - mu) / sd)
    print(f"학습 {len(y)}문제 (robust {y.sum()}) · 층 {LAYER} · C {C}")
    print(f"인샘플 부호 일치 {((margins > 0).astype(int) == y).mean():.3f} (참고용, 4096차원이라 의미 없음)")
    print(f"저장 {OUT}")


if __name__ == "__main__":
    main()
