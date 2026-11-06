"""H6 — 내부 흔들기 (화이트박스, 라벨 불필요, 생성 없음).

판정 대상 모델 본인에게 원문제와 정답불변 변형 V개(rename·typos·distract)를 forward만 시켜,
  d_last   고정 층 마지막 토큰 벡터의 코사인 거리 (원본 vs 변형)
  d_mean   고정 층 문제 토큰 평균풀링 벡터의 코사인 거리
  d_last_all / d_mean_all   위 두 거리를 전 층에서 재서 평균 (층 선택 없음)
  kl_next  첫 답 토큰 분포(마지막 위치 다음토큰 로짓)의 KL(원본‖변형)
변형 3개에 대해 평균한 값을 문제 특징으로 쓴다.

사전등록 부호: 흔들림 작음 = robust  →  robust 점수 = −거리, −KL.
층은 미리 고정: 중간층(L//2) 하나 + 전 층 평균. 여러 층 중 최댓값 고르기 금지.
길이 통제: 변형으로 생긴 토큰 수 차이 |Δtok| 을 대조군으로 두고, 각 특징에서 |Δtok|을 회귀로 뺀 잔차도 본다.

모델별 자기 표현이라 모델×문제 상호작용 특징이다 (9/14 민감도 실험과의 차이: 4모델 각자 본인 · v2 라벨 · 길이변화 통제).

  --extract --model M   → results/h6_<M>.npz
  (기본) 평가            → v2 33문제: 모델별 · 4모델 z-평균 풀링, AUC/bacc + 섞기 null + 길이 대조군 + 잔차

사용:  .venv/Scripts/python.exe experiments/h6_internal_shake.py --extract --model Qwen/Qwen3.5-4B
       .venv/Scripts/python.exe experiments/h6_internal_shake.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import LOCAL_MODELS, SOLVE_PROMPT, load_model, problem_table  # noqa: E402
from gen_labels import EDITS  # noqa: E402  (rename · typos · distract; 규칙 기반)

VARIANTS = ("rename", "typos", "distract")
FEATS = ["d_last", "d_mean", "d_last_all", "d_mean_all", "kl_next"]
RES = HERE / "results"


def forward_states(tok, model, text):
    """반환: hidden (L+1, T, D) float32 CPU · 마지막 위치 로그확률 (V,) · 프롬프트 토큰 수 T · 문제 토큰 구간"""
    import torch
    prefix = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n"}], tokenize=False, add_generation_prompt=False)
    full = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}], tokenize=False, add_generation_prompt=True)
    ids = tok(full, return_tensors="pt").to(model.device)
    n_prefix = len(tok(prefix)["input_ids"])
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hs = torch.stack([h[0] for h in out.hidden_states]).float()            # (L+1, T, D)
    logp = torch.log_softmax(out.logits[0, -1].float(), -1)
    T = ids["input_ids"].shape[1]
    lo, hi = min(n_prefix, T - 1), T - 1                                    # 문제 토큰 구간(대략): 프롬프트 접두 뒤 ~ 생성 프롬프트 앞
    last = hs[:, -1]                                                        # (L+1, D)
    mean = hs[:, lo:hi].mean(1) if hi > lo else hs[:, -1]                  # (L+1, D)
    return last.cpu(), mean.cpu(), logp.cpu(), T


def cos_dist(a, b):
    import torch
    return (1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).numpy()   # (L+1,)


def extract(model_id):
    import torch
    tbl = problem_table()
    tok = load_tokenizer(model_id); model = load_model(model_id, "4bit")
    rng = np.random.default_rng(0)
    F, dtok, y, pidx, probs = [], [], [], [], []
    t0 = time.time()
    for r in tbl.itertuples():
        last0, mean0, lp0, T0 = forward_states(tok, model, r.problem)
        L = last0.shape[0]; mid = L // 2
        per_var, dts = [], []
        for v in VARIANTS:
            t = EDITS[v](r.problem, rng)
            if not t or t == r.problem:
                continue
            last1, mean1, lp1, T1 = forward_states(tok, model, t)
            dl, dm = cos_dist(last0, last1), cos_dist(mean0, mean1)
            kl = float((lp0.exp() * (lp0 - lp1)).sum())
            per_var.append([dl[mid], dm[mid], dl[1:].mean(), dm[1:].mean(), kl]); dts.append(abs(T1 - T0))
        if not per_var:
            continue
        F.append(np.mean(per_var, axis=0)); dtok.append(float(np.mean(dts)))
        y.append(int(r.y)); pidx.append(int(r.pidx)); probs.append(r.problem)
        print(f"  [{r.pidx + 1:2d}/{len(tbl)}] T={T0:4d} |Δtok|={dtok[-1]:.1f}  " + "  ".join(f"{f}={v:.3f}" for f, v in zip(FEATS, F[-1])) + f"  y={r.y:+d}  {time.time() - t0:.0f}s", flush=True)
    out = RES / f"h6_{alias(model_id)}.npz"
    np.savez_compressed(out, F=np.array(F, dtype=np.float32), dtok=np.array(dtok, dtype=np.float32), y=np.array(y, dtype=np.int8),
                        pidx=np.array(pidx), problems=np.array(probs), model=model_id, feats=np.array(FEATS), mid_layer=mid)
    print(f"저장 {out}  F{np.array(F).shape}  {time.time() - t0:.0f}s")
    del model; torch.cuda.empty_cache()


def evaluate(perms=30):
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0)
    lines = []
    def out(s=""): print(s); lines.append(s)

    def row(name, score, y, ctrl):
        auc = roc_auc_score(y, score); bacc = balanced_accuracy_score(y, (score > np.median(score)).astype(int))
        null = [roc_auc_score(rng.permutation(y), score) for _ in range(perms)]
        lo, hi = np.percentile(null, [2.5, 97.5]); flag = "▲" if auc > hi else ("▼" if auc < lo else " ")
        resid = score - np.polyval(np.polyfit(ctrl, score, 1), ctrl)
        out(f"  {name:12s} AUC {auc:.3f} bacc {bacc:.3f}  null[{lo:.2f},{hi:.2f}]{flag}  ρ(|Δtok|){spearmanr(score, ctrl).correlation:+.2f}  잔차AUC {roc_auc_score(y, resid):.3f}")

    data = {m: np.load(RES / f"h6_{alias(m)}.npz", allow_pickle=True) for m in LOCAL_MODELS if (RES / f"h6_{alias(m)}.npz").is_file()}
    if not data:
        out("결과 없음"); return
    out(f"H6 내부 흔들기 · 사전등록: 흔들림 작음 = robust (점수 = −특징) · 고정 층 = L//2 + 전 층 평균 · perms={perms}")
    pooled = {f: [] for f in FEATS}; y_ref = None
    for m, d in data.items():
        lab = d["y"] >= 0; y, F, ctrl = d["y"][lab].astype(int), d["F"][lab].astype(np.float32), d["dtok"][lab].astype(np.float32)
        out(f"\n== {alias(m)}  (n={len(y)}, robust {y.sum()}, 중간층 {int(d['mid_layer'])})")
        row("|Δtok| 대조군", -ctrl, y, ctrl + 1e-3)
        for j, f in enumerate(FEATS):
            sc = -F[:, j]; row(f, sc, y, ctrl); pooled[f].append((sc - sc.mean()) / (sc.std() + 1e-6))
        y_ref = (y, ctrl)
    if len(data) >= 2:
        y, ctrl = y_ref
        out(f"\n== {len(data)}모델 z-평균 풀링 (n={len(y)})")
        for f in FEATS:
            row(f, np.mean(pooled[f], axis=0), y, ctrl)
    out("\n판정: ▲ 없으면 내부 흔들림은 v2 라벨과 무관. ▲ 있어도 잔차AUC가 0.5로 돌아오면 변형이 바꾼 토큰 수의 대리.")
    (RES / "h6_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--model"); ap.add_argument("--perms", type=int, default=30)
    a = ap.parse_args()
    if a.extract:
        for m in ([a.model] if a.model else LOCAL_MODELS):
            extract(m)
    else:
        evaluate(a.perms)
