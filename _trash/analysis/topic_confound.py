"""프로브가 'robust'가 아니라 '문제 유형'을 배운 건 아닌지 확인한다.

137문제가 전부 같은 모델(DeepSeek 8B)이라 라벨이 갈리는 건 문제 쪽뿐이다.
robust 36문제가 특정 주제·길이에 몰려 있으면, 4096차원 프로브는 그 지름길을 배운다.

세 가지를 본다:
  [1] 라벨 × 주제(type) · 난이도(level) 분할표      — MATH 원본과 텍스트로 대조
  [2] 얕은 특징(길이·숫자 개수·수식 밀도·주제)만으로 라벨을 맞히면 몇 점인가
      → 프로브 cv 0.634와 비슷하면 프로브가 배운 게 얕은 특징일 가능성
  [3] 프로브를 '주제 하나를 통째로 빼고' 학습 → 뺀 주제에서 맞히나 (leave-one-type-out)
      → 무너지면 주제를 외운 것, 버티면 주제 너머의 신호

사용:  uv run python analysis/topic_confound.py [--layer 17] [--layer 31]
"""

import argparse
import io
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import require  # noqa: E402

HERE = Path(__file__).resolve().parent
HIDDEN = HERE.parent / "experiments" / "results" / "hidden_DeepSeek-R1-0528-Qwen3-8B.npz"
CACHE = HERE.parent / "experiments" / "results" / "hendrycks_math.parquet"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]


def norm(s):
    return re.sub(r"\s+", " ", s.replace("\r", "")).strip()


def load_math():
    if CACHE.is_file():
        return pd.read_parquet(CACHE)
    frames = []
    for subj in SUBJECTS:
        for split in ("train", "test"):
            url = f"https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main/{subj}/{split}-00000-of-00001.parquet"
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read()
            df = pd.read_parquet(io.BytesIO(raw))
            df["split"] = split
            frames.append(df[["problem", "level", "type", "split"]])
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df.problem.map(norm)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def shallow_features(text):
    return {
        "n_chars": len(text),
        "n_words": len(re.findall(r"[A-Za-z]+", text)),
        "n_numbers": len(re.findall(r"\d+", text)),
        "n_dollar": text.count("$"),
        "latex_frac": len(re.findall(r"\\[a-zA-Z]+", text)) / max(1, len(text.split())),
        "has_geom": int(bool(re.search(r"triangle|circle|angle|radius|\bABC\b|polygon|diameter", text, re.I))),
        "has_prob": int(bool(re.search(r"probability|random|dice|coin|chosen", text, re.I))),
    }


def cv_bacc(X, y, C=1.0, seeds=(0, 1, 2)):
    out = []
    for seed in seeds:
        pred = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
            pred[te] = clf.predict((X[te] - mu) / sd)
        out.append(balanced_accuracy_score(y, pred))
    return np.mean(out), np.std(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, action="append", default=None)
    args = parser.parse_args()
    layers = args.layer or [17, 31]
    require(HIDDEN)

    d = np.load(HIDDEN, allow_pickle=True)
    H, y, pid = d["H"].astype(np.float32), d["y"].astype(int), list(d["pid"])
    # 문제 텍스트는 agg CSV에서
    from _paths import AGG_CSV
    csv_rows = {r["problem_id"]: r["original_problem"] for r in
                __import__("csv").DictReader(open(AGG_CSV, newline="", encoding="utf-8"))}
    texts = [csv_rows[p] for p in pid]

    math = load_math()
    lookup = dict(zip(math.key, zip(math.type, math.level, math.split)))
    meta = [lookup.get(norm(t), ("?", "?", "?")) for t in texts]
    types = np.array([m[0] for m in meta]); levels = np.array([m[1] for m in meta])
    print(f"MATH 원본과 매칭: {sum(t != '?' for t in types)}/{len(pid)}\n")

    # [1] 분할표 ------------------------------------------------------------
    print("[1] robust 비율 × 주제")
    for t, n in Counter(types).most_common():
        m = types == t
        print(f"   {t:26s} {y[m].sum():2d}/{n:3d}  = {y[m].mean():.2f}")
    print("\n    robust 비율 × 난이도")
    for lv in sorted(set(levels)):
        m = levels == lv
        print(f"   {lv:10s} {y[m].sum():2d}/{m.sum():3d}  = {y[m].mean():.2f}")

    # [2] 얕은 특징 ----------------------------------------------------------
    feats = pd.DataFrame([shallow_features(t) for t in texts])
    print("\n[2] 얕은 특징 단독 AUC (0.5 = 무정보)")
    for c in feats.columns:
        print(f"   {c:12s} {roc_auc_score(y, feats[c]):.3f}")
    type_oh = pd.get_dummies(pd.Series(types), prefix="t").astype(float)
    lvl_num = pd.Series([int(re.sub(r"\D", "", l) or 0) for l in levels], name="level")
    Xs = np.hstack([feats.values, type_oh.values, lvl_num.values[:, None]]).astype(np.float32)
    m, s = cv_bacc(Xs, y)
    print(f"\n   얕은 특징 전부(길이·숫자·수식·주제·난이도) → 로지스틱 cv bacc {m:.3f} ± {s:.3f}")
    print(f"   비교: 프로브 cv bacc 0.634 (층 17)")

    # [3] leave-one-type-out ---------------------------------------------
    print("\n[3] 프로브 leave-one-type-out — 뺀 주제에서의 bacc")
    for layer in layers:
        X = H[:, layer]
        print(f"   층 {layer}:")
        preds = np.zeros_like(y); ok = True
        for t in sorted(set(types)):
            te = types == t; tr = ~te
            if y[te].min() == y[te].max():          # 한 클래스뿐이면 bacc 정의 불가
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
                preds[te] = clf.predict((X[te] - mu) / sd)
                print(f"     {t:26s} n={te.sum():3d}  (한 클래스뿐, 개별 bacc 생략)")
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
            preds[te] = clf.predict((X[te] - mu) / sd)
            print(f"     {t:26s} n={te.sum():3d}  bacc {balanced_accuracy_score(y[te], preds[te]):.3f}")
        print(f"     전체 (주제 밖 예측만 모아서)      bacc {balanced_accuracy_score(y, preds):.3f}")
    print("\n읽는 법: [3]이 [2]보다 뚜렷이 높고 0.5를 넘으면 프로브는 주제 너머를 본다.")
    print("        [3]이 0.5 근처로 무너지면 프로브가 배운 건 주제였다.")


if __name__ == "__main__":
    main()
