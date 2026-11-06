"""H8 — 표면 의존도 ([author] 아이디어): 잘못된 패턴 = 수학 구조가 아닌 표면 단서에 기대 맞힘.

사전등록 (2026-09-29):
  모델 본인 · forward+backward 1회 (생성 없음)
  대상 스칼라   마지막 프롬프트 토큰의 L//2층 표현 노름
  토큰 기여도   |∇_emb(스칼라) · 입력임베딩|  (문제 토큰 구간만)
  토큰 분류     규칙 기반, 사전 고정:
     구조 = 숫자 · 연산자/괄호 · LaTeX 명령 · 수학어 목록(STRUCT_WORDS)
     표면 = 문장 첫머리가 아닌 대문자 시작 단어(고유명사) + 수학어·불용어가 아닌 길이≥4 알파벳 단어
     그 외(불용어·구두점) 무시
  지표          surface_ratio = 표면 기여도 합 / (표면 + 구조 기여도 합)
  부호          비율 큼 = spurious  →  robust 점수 = −surface_ratio
검문소  (1) 섞기 null 30회  (2) 원문제 길이(log) + 표면 토큰 '개수' 비율 둘 다 잔차화 후 남는지
        (3) 모델 간 순위 상관  (4) 통과 시에만 MATH137 DeepSeek 1회
4bit에서 backward가 안 되면 bf16/CPU 오프로드로 DeepSeek·Qwen만.

사용:  .venv/Scripts/python.exe experiments/h8_surface_dependence.py --extract --model M [--set v2|math]
       .venv/Scripts/python.exe experiments/h8_surface_dependence.py
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
from extract_train_v2 import LOCAL_MODELS, SOLVE_PROMPT, load_model, problem_table  # noqa: E402

RES = HERE / "results"
STRUCT_WORDS = set("""sum product difference quotient divisible divides divisor divisors multiple multiples prime primes remainder
integer integers positive negative real rational irrational digit digits odd even square squares root cube factor factors
number numbers count total each every all any exactly least most more less greater smaller than equal equals equation
sequence sequences term terms series arithmetic geometric ratio proportion fraction decimal percent probability random
area perimeter length lengths width height radius diameter circle circles triangle triangles square rectangle polygon
angle angles degrees side sides vertex vertices point points line lines segment midpoint parallel perpendicular tangent
inscribed circumscribed circumradius inradius function functions polynomial coefficient coefficients solution solutions
value values maximum minimum largest smallest greatest find compute determine calculate evaluate how many what
distinct different possible satisfy satisfies condition consecutive infinite finite set subset element elements
modulo mod base bases written expressed relatively coprime gcd lcm floor ceiling absolute power powers exponent
""".split())
STOP = set("""the a an of to in on at by for with and or but if then is are was were be been being this that these those
it its as from into over under between among which who whom whose where when while there here such so not no nor
can could would should may might will shall do does did have has had he she they we you i his her their our your
let suppose assume given define denote consider note also only just both either neither own same other another""".split())


def classify(piece):
    """토큰 조각 → 'struct' | 'surface' | None"""
    p = piece.strip()
    if not p:
        return None
    if re.search(r"\d", p) or p.startswith("\\") or re.fullmatch(r"[+\-*/=<>^_(){}\[\]|!.,;:?$%&~]+", p):
        return "struct"
    w = re.sub(r"[^A-Za-z]", "", p)
    if not w:
        return None
    wl = w.lower()
    if wl in STRUCT_WORDS:
        return "struct"
    if wl in STOP:
        return None
    if w[0].isupper():
        return "surface"            # 문장 첫머리 여부는 아래에서 보정
    if len(w) >= 4:
        return "surface"
    return None


def attributions(tok, model, text):
    import torch
    prefix = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n"}], tokenize=False, add_generation_prompt=False)
    full = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}], tokenize=False, add_generation_prompt=True)
    ids = tok(full, return_tensors="pt")["input_ids"].to(model.device)
    n_prefix = len(tok(prefix)["input_ids"]); T = ids.shape[1]
    emb_layer = model.get_input_embeddings()
    emb = emb_layer(ids).detach().float().requires_grad_(True)
    out = model(inputs_embeds=emb.to(emb_layer.weight.dtype), output_hidden_states=True)
    L = len(out.hidden_states); mid = L // 2
    scalar = out.hidden_states[mid][0, -1].float().norm()
    scalar.backward()
    contrib = (emb.grad * emb).sum(-1)[0].abs().detach().cpu().numpy()     # (T,)
    pieces = [tok.decode([i]) for i in ids[0].tolist()]
    lo, hi = min(n_prefix, T - 1), T - 1
    surf = struct = 0.0; n_surf = n_struct = 0
    prev_end = True
    for i in range(lo, hi):
        c = classify(pieces[i])
        w = re.sub(r"[^A-Za-z]", "", pieces[i])
        if c == "surface" and w and w[0].isupper() and prev_end and len(w) < 4:
            c = None                                          # 문장 첫머리 대문자 짧은 단어는 고유명사로 안 봄
        if c == "surface":
            surf += contrib[i]; n_surf += 1
        elif c == "struct":
            struct += contrib[i]; n_struct += 1
        prev_end = pieces[i].strip().endswith((".", "?", "!"))
    model.zero_grad(set_to_none=True)
    ratio = surf / (surf + struct) if surf + struct > 0 else np.nan
    cnt = n_surf / (n_surf + n_struct) if n_surf + n_struct > 0 else np.nan
    return ratio, cnt, T, n_surf, n_struct


def texts(which):
    if which == "v2":
        return [(int(r.pidx), r.problem, int(r.y)) for r in problem_table().itertuples()]
    rows, seen = [], set()
    for i, r in enumerate(csv.DictReader(open(AGG_CSV, newline="", encoding="utf-8"))):
        if r["problem_id"] in seen: continue
        seen.add(r["problem_id"]); rows.append((i, r["original_problem"], int(r["model_is_robust"].strip().lower() == "true")))
    return rows


def extract(model_id, which, quant):
    import torch
    tok = load_tokenizer(model_id); model = load_model(model_id, quant)
    for p in model.parameters():
        p.requires_grad_(False)
    rows = texts(which); R = []; t0 = time.time()
    for i, (pidx, text, y) in enumerate(rows, 1):
        ratio, cnt, T, ns, nt = attributions(tok, model, text)
        R.append((pidx, ratio, cnt, T, ns, nt, y))
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ratio={ratio:.3f} cnt={cnt:.3f} T={T} surf={ns} struct={nt} y={y}  {time.time() - t0:.0f}s", flush=True)
    R = np.array(R, dtype=np.float32)
    out = RES / f"h8_{alias(model_id)}_{which}.npz"
    np.savez_compressed(out, pidx=R[:, 0].astype(int), ratio=R[:, 1], cnt=R[:, 2], T=R[:, 3], n_surf=R[:, 4], n_struct=R[:, 5],
                        y=R[:, 6].astype(np.int8), problems=np.array([t for _, t, _ in rows]), model=model_id, quant=quant)
    print(f"저장 {out}  {time.time() - t0:.0f}s")
    del model; torch.cuda.empty_cache()


def evaluate(perms=30, which="v2"):
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0); lines = []
    def out(s=""): print(s); lines.append(s)
    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls); b, *_ = np.linalg.lstsq(X, x, rcond=None); return x - X @ b
    data = {m: np.load(RES / f"h8_{alias(m)}_{which}.npz", allow_pickle=True) for m in LOCAL_MODELS if (RES / f"h8_{alias(m)}_{which}.npz").is_file()}
    if not data:
        out("결과 없음"); return
    out(f"H8 표면 의존도 ({which}) · 사전등록: 비율 큼 = spurious (점수 = −ratio) · null {perms}회")
    out(f"{'모델':12s} {'n':>3s} {'AUC':>6s} {'bacc':>6s} {'null95%':>13s} {'ρ(ratio,cnt)':>13s} {'ρ(ratio,logT)':>14s} {'cnt만 AUC':>9s} {'잔차AUC':>8s} {'잔차null':>13s}")
    pooled, pooled_r, ratios = [], [], {}
    for m, d in data.items():
        lab = (d["y"] >= 0) & np.isfinite(d["ratio"]); y = d["y"][lab].astype(int)
        r, c, T = d["ratio"][lab].astype(float), d["cnt"][lab].astype(float), np.log(d["T"][lab].astype(float))
        sc = -r; auc = roc_auc_score(y, sc); bacc = balanced_accuracy_score(y, (sc > np.median(sc)).astype(int))
        null = [roc_auc_score(rng.permutation(y), sc) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
        rs = -resid(r, [T, c]); ra = roc_auc_score(y, rs)
        null2 = [roc_auc_score(rng.permutation(y), rs) for _ in range(perms)]; lo2, hi2 = np.percentile(null2, [2.5, 97.5])
        f1 = "▲" if auc > hi else ("▼" if auc < lo else " "); f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
        out(f"{alias(m)[:12]:12s} {len(y):3d} {auc:6.3f} {bacc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {spearmanr(r, c).correlation:+13.2f} {spearmanr(r, T).correlation:+14.2f} {roc_auc_score(y, -c):9.3f} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2}")
        pooled.append((sc - sc.mean()) / sc.std()); pooled_r.append(rs / rs.std()); ratios[m] = r; y_ref = y
    if len(data) >= 2:
        y = y_ref; P, PR = np.mean(pooled, 0), np.mean(pooled_r, 0)
        null = [roc_auc_score(rng.permutation(y), PR) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
        out(f"{'풀링':12s} {len(y):3d} {roc_auc_score(y, P):6.3f} {'':>6s} {'':>13s} {'':>13s} {'':>14s} {'':>9s} {roc_auc_score(y, PR):8.3f} [{lo:.2f},{hi:.2f}]{'▲' if roc_auc_score(y, PR) > hi else ('▼' if roc_auc_score(y, PR) < lo else ' ')}")
        names = list(ratios)
        out("모델 간 ratio 순위 상관: " + "  ".join(f"{alias(names[i])[:8]}-{alias(names[j])[:8]} {spearmanr(ratios[names[i]], ratios[names[j]]).correlation:+.2f}" for i in range(len(names)) for j in range(i + 1, len(names))))
    (RES / f"h8_report_{which}.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--model"); ap.add_argument("--set", default="v2", choices=["v2", "math"])
    ap.add_argument("--quant", default="4bit", choices=["4bit", "none"]); ap.add_argument("--perms", type=int, default=30)
    a = ap.parse_args()
    if a.extract:
        for m in ([a.model] if a.model else LOCAL_MODELS):
            extract(m, a.set, a.quant)
    else:
        evaluate(a.perms, a.set)
