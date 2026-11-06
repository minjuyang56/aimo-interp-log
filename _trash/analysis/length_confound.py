"""프로브가 배운 게 '길이'인지, 그리고 길이가 왜 라벨과 얽혔는지 확인한다.

topic_confound.py가 문제 길이 단독 AUC 0.75를 찾았다. 여기서 두 가지를 잰다:

  (a) 프로브 out-of-fold 점수와 길이의 상관. 길이 위에 프로브가 뭘 더하는가.
      [길이]만으로 AUC vs [길이 + 프로브 점수] — 같으면 프로브는 길이의 대리다.
  (b) 유형별로 '무너진 문제'와 '버틴 문제'의 길이를 비교한다. 특정 유형에서만
      길이가 갈리면, 라벨의 길이 의존은 그 유형의 변형 생성 절차에서 온 것이다.

사용:  uv run python analysis/length_confound.py
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import AGG_CSV, FINAL_CSV, require  # noqa: E402

HIDDEN = Path(__file__).resolve().parent.parent / "experiments" / "results" / "hidden_DeepSeek-R1-0528-Qwen3-8B.npz"
TYPES = ["domain", "distract", "rephrase", "rename", "typos"]


def n_words(text):
    return len(re.findall(r"[A-Za-z]+", text))


def oof_margin(X, y, C=0.05, seed=0):
    m = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
        m[te] = clf.decision_function((X[te] - mu) / sd)
    return m


def cv_auc(X, y, seed=1):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(class_weight="balanced", max_iter=3000).fit(X[tr], y[tr])
        p[te] = clf.decision_function(X[te])
    return roc_auc_score(y, p)


def main():
    require(HIDDEN, AGG_CSV, FINAL_CSV)
    d = np.load(HIDDEN, allow_pickle=True)
    H, y, pid = d["H"].astype(np.float32), d["y"].astype(int), list(d["pid"])
    text = {r["problem_id"]: r["original_problem"] for r in csv.DictReader(open(AGG_CSV, newline="", encoding="utf-8"))}
    lognw = np.log1p([n_words(text[p]) for p in pid])

    print("(a) 프로브 점수 vs 길이")
    print(f"    길이 단독 AUC {roc_auc_score(y, lognw):.3f}")
    for layer in (17, 31):
        m = oof_margin(H[:, layer], y)
        rho = spearmanr(m, lognw).correlation
        resid = m - np.polyval(np.polyfit(lognw, m, 1), lognw)
        print(f"    층 {layer}: 프로브 oof AUC {roc_auc_score(y, m):.3f} · 상관(프로브,길이) ρ={rho:+.2f}"
              f" · [길이] {cv_auc(lognw[:, None], y):.3f} → [길이+프로브] {cv_auc(np.c_[lognw, m], y):.3f}"
              f" · 길이 제거 잔차 AUC {roc_auc_score(y, resid):.3f}")

    print("\n(b) 유형별: 무너진 문제 vs 버틴 문제의 길이")
    rows = [r for r in csv.DictReader(open(FINAL_CSV, newline="", encoding="utf-8")) if r["permutation_type"] != "OVERALL"]
    for t in TYPES:
        rs = [r for r in rows if r["permutation_type"] == t]
        L = np.array([n_words(r["original_problem"]) for r in rs], float)
        b = np.array([int(r["n_detrimental_permutations"]) > 0 for r in rs], int)
        if b.min() == b.max():
            print(f"    {t:9s} n={len(rs):3d}  (전부 {'무너짐' if b[0] else '버팀'})"); continue
        print(f"    {t:9s} n={len(rs):3d}  무너짐 {L[b == 1].mean():5.1f}단어 · 버팀 {L[b == 0].mean():5.1f}단어"
              f" · AUC(길이→버팀) {roc_auc_score(1 - b, L):.3f}")
    short = [r for r in rows if r["permutation_type"] == "domain" and n_words(r["original_problem"]) <= 12]
    print(f"\n    12단어 이하 문제의 domain 무너짐: {sum(int(r['n_detrimental_permutations']) > 0 for r in short)}/{len(short)}")
    for r in short[:3]:
        it = json.loads(r["permutations_causing_decay"])[:1]
        print(f"      {r['original_problem'][:55]!r}  →  {it[0]['question'][:60] + '…' if it else '-'!r}")
    print("\n읽는 법: (a)에서 [길이+프로브]가 [길이]와 같으면 프로브는 길이의 대리다.")
    print("        (b)에서 한 유형만 길이로 갈리면 라벨의 길이 의존은 그 유형의 변형 생성에서 왔다.")


if __name__ == "__main__":
    main()
