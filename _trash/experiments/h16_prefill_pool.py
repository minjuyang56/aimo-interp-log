"""H16 — prefill 평균풀링 probe, 4모델 점수 풀링. 사전등록 2026-10-01 (지휘 #48).

생성 없음. 문제 텍스트를 판정 대상 모델에 **한 번 넣고**(prefill) 층 L//4 · L//2 · 3L//4 에서
**문제 토큰 구간 평균 풀링** → 모델별 로지스틱(강한 L2) → **문제 단위로 4모델 점수 평균**.
라벨이 문제 단위(H2: 11/11 모델 간 동일)이므로 모델별 점수를 평균하는 것이 설계상 맞다.

평가: 문제 단위 **leave-one-out**. **층과 C 는 폴드 내부 5-fold CV 로만 선택**(nested). 층·C 는 4모델 공통.
      후보 C ∈ {0.003, 0.01, 0.03, 0.1}. null = 문제 단위 라벨 섞기 1000회.
H16b: 학습셋에 v3 Skywork 유효 라벨(AIME/level5 29건)을 더했을 때의 LOO AUC 변화 — **탐색**.

사용:  .venv/Scripts/python.exe experiments/h16_prefill_pool.py --extract [--model M]
       .venv/Scripts/python.exe experiments/h16_prefill_pool.py [--perms 1000] [--with-v3]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import LOCAL_MODELS, load_model  # noqa: E402
from h9_essence import prompt_of  # noqa: E402

RES = HERE / "results"
FRACS = (4, 2, 4 / 3)          # L//4, L//2, 3L//4 (hidden_states 인덱스 기준)


def official_problems():
    import pandas as pd
    df = pd.read_parquet(RES / "train-main-v2.parquet")
    rows = []
    for i, (p, g) in enumerate(df.groupby("problem", sort=False)):
        lab = g.is_robust.dropna()
        if len(lab):
            rows.append({"pidx": i, "problem": p.rstrip(), "y": int(bool(lab.astype(bool).mean() >= 0.5))})
    return rows


def v3_problems():
    """v3 Skywork 유효 라벨: robust 24 + 진짜 spurious 4 + 오답 우세 1. (H16b 탐색용)"""
    import json
    lp = sorted(RES.glob("genlab/**/Skywork-OR1-Math-7B_v3_labels.json"))
    pp = sorted(RES.glob("genlab/**/Skywork-OR1-Math-7B_probe13.json"))
    if not lp:
        return []
    lab = json.loads(lp[-1].read_text(encoding="utf-8"))
    verdict = {}
    if pp:
        verdict = {int(r["pidx"]): r["verdict"] for r in json.loads(pp[-1].read_text(encoding="utf-8"))}
    rows = []
    for s in lab:
        if s.get("label") is None:
            continue
        if s["label"] is False and verdict.get(int(s["pidx"])) == "truncation_artifact":
            continue                                   # 절단 아티팩트는 제외
        rows.append({"pidx": 10000 + int(s["pidx"]), "problem": s["problem"], "y": int(bool(s["label"]))})
    return rows


def extract(model_id, rows, tag):
    import torch
    tok = load_tokenizer(model_id)
    model = load_model(model_id, "4bit")
    L = model.config.num_hidden_layers + 1
    idx = sorted({max(0, min(L - 1, int(L / f))) for f in FRACS})
    print(f"[{alias(model_id)}] hidden_states {L}개 중 층 {idx} 사용", flush=True)
    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        p0 = prompt_of(tok, r["problem"])
        base = p0.index(r["problem"])
        enc = tok(p0, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        span = [j for j, (s_, e_) in enumerate(offs) if e_ > base and s_ < base + len(r["problem"])]
        with torch.no_grad():
            hs = model(input_ids=enc["input_ids"].to(model.device), output_hidden_states=True, logits_to_keep=1).hidden_states
        vec = np.stack([hs[li][0, span].float().mean(0).cpu().numpy() for li in idx])
        out.append({"pidx": r["pidx"], "y": r["y"], "vec": vec.astype(np.float32), "n_tok": len(span)})
        if i % 15 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    f = RES / f"h16_{alias(model_id)}_{tag}.npz"
    np.savez_compressed(f, store=np.array(out, dtype=object), layers=np.array(idx), model=model_id)
    print(f"저장 {f}", flush=True)
    del model
    torch.cuda.empty_cache()


def evaluate(perms=1000, with_v3=False, tag="off"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(0)
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    data = {}
    for m in LOCAL_MODELS:
        fs = [RES / f"h16_{alias(m)}_{t}.npz" for t in (["off", "v3"] if with_v3 else ["off"])]
        if not all(f.is_file() for f in fs):
            continue
        store = []
        for f in fs:
            store += list(np.load(f, allow_pickle=True)["store"])
        data[m] = store
    if len(data) < 2:
        out(f"모델 {len(data)}개뿐 — 최소 2개 필요"); return lines
    names = list(data)
    common = sorted(set.intersection(*[{int(r["pidx"]) for r in data[m]} for m in names]))
    y = np.array([next(int(r["y"]) for r in data[names[0]] if int(r["pidx"]) == p) for p in common])
    n_layer = data[names[0]][0]["vec"].shape[0]
    X = {m: np.stack([next(r["vec"] for r in data[m] if int(r["pidx"]) == p) for p in common]) for m in names}
    out(f"H16{'b' if with_v3 else ''} prefill 풀링 probe · 모델 {len(names)} · 문제 {len(y)} "
        f"(robust {int(y.sum())} / spurious {int(len(y) - y.sum())}) · 층 {n_layer}개 · nested LOO · null {perms}회")
    Cs = [0.003, 0.01, 0.03, 0.1]

    def fit_score(tr, te, li, C):
        """4모델 점수 평균. tr/te 는 인덱스 배열."""
        s_te = np.zeros(len(te))
        for m in names:
            Z = X[m][:, li]
            mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
            clf.fit((Z[tr] - mu) / sd, y[tr])
            s_te += clf.decision_function((Z[te] - mu) / sd)
        return s_te / len(names)

    pred = np.zeros(len(y))
    picks = []
    for i in range(len(y)):
        tr = np.array([j for j in range(len(y)) if j != i])
        best, best_auc = None, -1
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for li in range(n_layer):
            for C in Cs:
                sc, yy = [], []
                for a_, b_ in inner.split(tr, y[tr]):
                    sc.append(fit_score(tr[a_], tr[b_], li, C)); yy.append(y[tr[b_]])
                auc = roc_auc_score(np.concatenate(yy), np.concatenate(sc))
                if auc > best_auc:
                    best_auc, best = auc, (li, C)
        picks.append(best)
        pred[i] = fit_score(tr, np.array([i]), *best)[0]
    auc = roc_auc_score(y, pred)
    nl = [roc_auc_score(rng.permutation(y), pred) for _ in range(perms)]
    lo, hi = np.percentile(nl, [2.5, 97.5])
    from collections import Counter
    out(f"  LOO AUC **{auc:.3f}** [null {lo:.2f},{hi:.2f}]{'▲' if auc > hi else ('▼' if auc < lo else ' ')} · "
        f"내부 CV 선택 분포 {dict(Counter(picks))}")
    (RES / f"h16{'b' if with_v3 else ''}_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--set", default="off", choices=["off", "v3"])
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--with-v3", action="store_true")
    a = ap.parse_args()
    if a.extract:
        rows = official_problems() if a.set == "off" else v3_problems()
        print(f"대상 {len(rows)}문제 ({a.set})", flush=True)
        for m in ([a.model] if a.model else LOCAL_MODELS):
            extract(m, rows, a.set)
    else:
        evaluate(a.perms, a.with_v3)
