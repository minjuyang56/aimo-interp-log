"""H1 — 절대 임계값 대신 배치 안 중앙값 분할. 검증셋이 모델별·문제별 균형이라는 공지를 이용한다.

are_robust가 받는 problems 배치 안에서 점수 순위를 매겨 상위 절반만 True로 한다.
그러면 임계값 보정이 필요 없고, 배치가 균형이면 상수 예측(0.5)보다 나은지는 순위가 맞느냐에만 달린다.

train-main-v2에서 모델별로 재계산:
  점수 후보   프로브 각 층(LOPO out-of-fold margin) · 자신감 특징 각각(raw) · 길이(대조군)
  평가        중앙값 분할 bacc  +  정답표 섞기 30회 null 분포(95% 구간)
  주의        여기선 배치 = 그 모델의 라벨 있는 33문제 전부. 실전 배치는 3~5문제라 분산이 더 크다.

사용:  .venv/Scripts/python.exe analysis/median_split.py [--n-tokens 512] [--perms 30]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
from extract_hidden import alias  # noqa: E402
from extract_train_v2 import LOCAL_MODELS  # noqa: E402
from train_train_v2 import lopo_margins  # noqa: E402

RES = HERE.parent / "experiments" / "results"


def median_split_bacc(score, y):
    pred = (score > np.median(score)).astype(int)
    return balanced_accuracy_score(y, pred)


def best_sign(score, y):
    """방향은 학습 데이터에서 정한다고 치고, 둘 중 나은 쪽. (실전에선 방향을 미리 고정해야 함)"""
    return max(median_split_bacc(score, y), median_split_bacc(-score, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=512)
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24])
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f"{'모델':14s} {'점수':22s} {'bacc':>6s} {'null 95%':>14s}  판정")
    for m in LOCAL_MODELS:
        hid = RES / f"trainv2_hidden_{alias(m)}.npz"; conf = RES / f"trainv2_conf_{alias(m)}_N{a.n_tokens}.npz"
        if not hid.is_file() or not conf.is_file():
            continue
        dh, dc = np.load(hid, allow_pickle=True), np.load(conf, allow_pickle=True)
        lab = dh["y"] >= 0; y = dh["y"][lab].astype(int)
        H = dh["H"].astype(np.float32)[lab]; F = dc["F"].astype(np.float32)[lab]; feats = list(dc["features"])
        lognw = np.log1p([len(re.findall(r"[A-Za-z]+", p)) for p in dh["problems"]])[lab]

        scores = {"길이(대조군)": lognw}
        for L in a.layers:
            if L < H.shape[1]:
                scores[f"프로브 층{L} (LOPO)"] = lopo_margins(H[:, L], y, 0.05)
        for i, f in enumerate(feats):
            if F[:, i].std() > 0:
                scores[f"자신감 {f}"] = F[:, i]
        scores["자신감 12특징 (LOPO)"] = lopo_margins(F, y, 1.0)

        # null (고정 점수용): 라벨을 섞어 best_sign 중앙값 분할 bacc 분포. 점수는 그대로.
        null = [best_sign(lognw, rng.permutation(y)) for _ in range(a.perms)]
        lo, hi = np.percentile(null, [2.5, 97.5])
        # null (LOPO 점수용): 섞은 라벨로 LOPO를 다시 돌려야 공정하다 — 학습이 들어가므로.
        null_lopo = []
        for _ in range(a.perms):
            yp = rng.permutation(y)
            null_lopo.append(best_sign(lopo_margins(F, yp, 1.0), yp))
        lo2, hi2 = np.percentile(null_lopo, [2.5, 97.5])
        for name, sc in scores.items():
            b_plus, b_minus = median_split_bacc(sc, y), median_split_bacc(-sc, y)
            is_lopo = "LOPO" in name
            L, Hh = (lo2, hi2) if is_lopo else (lo, hi)
            b = max(b_plus, b_minus)
            flag = "▲" if b > Hh else ("▼" if b < L else " ")
            sign = "+" if b_plus >= b_minus else "−"
            print(f"{alias(m)[:14]:14s} {name:22s} {b_plus:6.3f} {b_minus:6.3f} [{L:.2f}, {Hh:.2f}] {sign}{flag}")
        print()
    print("열: 점수 그대로(+) / 부호 뒤집음(−) 각각의 중앙값 분할 bacc. 실전에선 방향을 미리 고정해야 하므로")
    print("    '더 좋은 쪽'을 고르는 것 자체가 33문제에서 +0.03~0.06 낙관 편향이다.")
    print("null: 고정 점수는 라벨 섞기, LOPO 점수는 섞은 라벨로 LOPO 재학습. 95% 구간. ▲ = 구간 위.")
    print("LOPO 점수가 −방향에서만 높으면 anti-learning의 부호 반전이다 — 신호로 쓰지 않는다.")


if __name__ == "__main__":
    main()
