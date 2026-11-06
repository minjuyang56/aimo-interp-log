"""H4 외부 검증 — 옛 MATH 137문제(노이즈 라벨)로만 학습하고, train-main-v2 문제로만 시험한다.

학습과 시험이 다른 데이터·다른 시기·다른 출처라 이게 우리의 첫 '진짜' 시험이다.
계측기는 DeepSeek-8B 하나로 고정(공용). 특징은 forward 1회짜리만(프로브 층별 · 길이). 생성 없음.

주의: 옛 hidden_DeepSeek…npz는 잘못된 토크나이저(LlamaTokenizer)·CPU bf16으로 뽑은 것이라
v2 벡터(Fast 토크나이저·4bit)와 조건이 다르다. 여기서 137문제를 같은 조건으로 다시 뽑는다.

  --extract   137문제 forward → results/math137_hidden_DeepSeek…npz (GPU 4bit, 몇 분)
  (기본)      학습 137 → 시험 v2. 층별 bacc/AUC + 길이 대조군 + 라벨 섞기 null 30회

사용:  .venv/Scripts/python.exe experiments/h4_external.py --extract
       .venv/Scripts/python.exe experiments/h4_external.py
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "analysis"))
from _paths import AGG_CSV  # noqa: E402
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import SOLVE_PROMPT, load_model  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
OUT137 = HERE / "results" / f"math137_hidden_{alias(MODEL)}.npz"
V2 = HERE / "results" / f"trainv2_hidden_{alias(MODEL)}.npz"


def norm(s):
    return re.sub(r"\s+", " ", s.replace("\r", "")).strip().lower()


def nwords(s):
    return len(re.findall(r"[A-Za-z]+", s))


def load_math137():
    rows, seen = [], set()
    for r in csv.DictReader(open(AGG_CSV, newline="", encoding="utf-8")):
        if r["problem_id"] in seen:
            continue
        seen.add(r["problem_id"]); rows.append(r)
    return rows


def extract():
    import torch
    rows = load_math137()
    tok = load_tokenizer(MODEL); model = load_model(MODEL, "4bit")
    H, y = [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        text = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{r['original_problem']}"}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states
        H.append(np.stack([h[0, -1].float().cpu().numpy() for h in hs]).astype(np.float16))
        y.append(r["model_is_robust"].strip().lower() == "true")
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT137, H=np.stack(H), y=np.array(y), problems=np.array([r["original_problem"] for r in rows]), model=MODEL)
    print(f"저장 {OUT137}  H{np.stack(H).shape}  {time.time() - t0:.0f}s")


def fit_predict(Xtr, ytr, Xte, C):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit((Xtr - mu) / sd, ytr)
    return clf.decision_function((Xte - mu) / sd)


def evaluate(perms=30):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    d137, dv2 = np.load(OUT137, allow_pickle=True), np.load(V2, allow_pickle=True)
    p137, pv2 = list(d137["problems"]), list(dv2["problems"])
    overlap = {norm(p) for p in p137} & {norm(p) for p in pv2}
    keep = np.array([norm(p) not in overlap for p in pv2]) & (dv2["y"] >= 0)
    print(f"학습 MATH 137문제 (robust {int(d137['y'].sum())}) · 시험 v2 {keep.sum()}문제 (robust {int((dv2['y'][keep] == 1).sum())}) · 중복 제외 {len(overlap)}")
    Htr, ytr = d137["H"].astype(np.float32), d137["y"].astype(int)
    Hte, yte = dv2["H"].astype(np.float32)[keep], dv2["y"][keep].astype(int)
    ltr, lte = np.log1p([nwords(p) for p in p137]), np.log1p([nwords(p) for p in pv2])[keep]
    rng = np.random.default_rng(0)

    def row(name, Xtr, Xte, C):
        m = fit_predict(Xtr, ytr, Xte, C)
        bacc, auc = balanced_accuracy_score(yte, (m > 0).astype(int)), roc_auc_score(yte, m)
        null = []
        for _ in range(perms):
            mp = fit_predict(Xtr, rng.permutation(ytr), Xte, C)
            null.append(roc_auc_score(yte, mp))
        lo, hi = np.percentile(null, [2.5, 97.5])
        flag = "▲" if auc > hi else ("▼" if auc < lo else " ")
        print(f"  {name:16s} bacc {bacc:.3f}  AUC {auc:.3f}  null AUC [{lo:.2f}, {hi:.2f}] {flag}")
        return auc

    print(f"\n  {'특징':16s} {'시험 bacc':>10s} {'시험 AUC':>9s}  null(학습 라벨 섞기 {perms}회)")
    row("길이(대조군)", ltr[:, None], lte[:, None], 1.0)
    L = min(Htr.shape[1], Hte.shape[1])
    for layer in range(2, L, 4):
        row(f"프로브 층{layer}", Htr[:, layer], Hte[:, layer], 0.05)
    print("\n판정 기준: 시험 AUC가 null 구간 밖(▲)이면 MATH에서 배운 것이 올림피아드로 옮겨간 것. 길이 행이 같이 ▲면 길이의 대리.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--perms", type=int, default=30)
    a = ap.parse_args()
    extract() if a.extract else evaluate(a.perms)
