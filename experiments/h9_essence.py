"""H9 — 액기스 분리 ([author] 아이디어). 사전등록 2026-09-29.

(1) 단어 4통 규칙 분류 — 목록은 MATH137 어휘(results/math137_vocab.txt)만 보고 만들었고 v2로 조정하지 않는다.
      V_rel   관계·동작 동사 (divides, exceeds, satisfies, chosen, equals …)
      N_math  수학 명사 (remainder, prime, sum, integer, triangle, area …)
      Q       수량: 숫자 · $…$ 수식 한 덩어리 · 수량어 (twice, at least, each, largest …)
      N_topic 그 외 소재 명사·고유명사 (불용어 아닌 길이≥3 알파벳 단어 중 위 세 통에 없는 것)
(2) 중요도 = 지우기: 단위 하나를 중립어로 치환(V_rel→has, N_math·N_topic→thing, Q→some) → 마지막 토큰 L//2층 표현의
    코사인 변화량. 배치 forward, 생성 없음.
(3) 단어 자리 벡터 = 원문 forward의 L//2층 서브토큰 평균(오프셋 매핑). 구조액기스 = V_rel+N_math+Q 중요도 가중합,
    표면액기스 = N_topic 중요도 가중합.
(4) 지표(부호 고정, 큼 = spurious):  surface_share = Σimp(N_topic)/Σimp(전체)
                                   alignment = cos(last, 표면액기스) − cos(last, 구조액기스)
검문소: 섞기 null 30회 · 대조군(단어 개수 비율 · 원문제 길이) 잔차화 · 모델 간 순위상관 · 통과 시에만 MATH137 DeepSeek 1회.

사용:  .venv/Scripts/python.exe experiments/h9_essence.py --extract --model M [--set v2|math]
       .venv/Scripts/python.exe experiments/h9_essence.py [--set v2]
"""

import argparse
import csv
import json
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


def fwd(model, **kw):
    """logits_to_keep 미지원 모델(TypeError)이면 빼고 다시 호출."""
    try:
        return model(**kw)
    except TypeError:
        kw.pop("logits_to_keep", None); return model(**kw)

V_REL = set("""divides divided divisible exceeds exceed equals equal satisfy satisfies satisfying intersect intersects lies lie
varies vary chosen choose take taken put shake shakes add added weigh weighs obtain rearrange express expressed compute
evaluate simplify simplified find solve solved defined define represented represent participate enter measure averaging
reversed pointing turns cubed inscribed having combined contains contain formed form placed place arranged arrange
selected select removed remove leaves leave depart departs differ differs multiply multiplied subtract subtracted
holds hold gives give receives receive tangent touches meets meet passes pass""".split())
N_MATH = set("""number numbers value values sum sums integer integers digit digits remainder square squares prime primes
multiple multiples root roots area areas length lengths triangle triangles circle circles circular radius diameter polygon
octagon rectangle rectangles cone cones sphere angle angles side sides point points line lines vertical asymptotes graph
base bases sequence sequences term terms expression equation equations solution solutions divisor divisors factor factors
units expansion difference product quotient ratio decimal fraction fractions constant symmetry axes axis interior pair
pairs distance width height leg legs parentheses placement operation addition subtraction multiplication congruent
equivalent perimeter vertex vertices midpoint segment parallel perpendicular circumradius inradius polynomial coefficient
function probability set subset element power exponent modulo floor ceiling gcd lcm coprime arithmetic geometric series
mean median mode average matrix vector determinant complex real rational irrational""".split())
Q_WORDS = set("""one two three four five six seven eight nine ten eleven twelve twenty hundred thousand million twice thrice once
each every all both single per exactly least most more less than greater smaller maximum minimum largest smallest greatest
half third quarter double triple total first second last consecutive distinct different possible many much some several
few none any positive negative odd even whole nonzero""".split())
STOP = set("""the a an of to in on at by for with and or but if then is are was were be been being this that these those it
its as from into over under between among which who whom whose where when while there here such so not no nor can could
would should may might will shall do does did have has had he she they we you i his her their our your let suppose assume
given denote consider note also only just own same other another what how why whether about after before during until
your answer enter express simplest""".split())
NEUTRAL = {"V_rel": "has", "N_math": "thing", "Q": "some", "N_topic": "thing"}
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")


def units(text):
    """(start, end, bucket) 목록. 수식 덩어리·숫자·단어."""
    out = []
    covered = []
    for m in MATH_SPAN.finditer(text):
        out.append((m.start(), m.end(), "Q")); covered.append((m.start(), m.end()))
    def inmath(i): return any(a <= i < b for a, b in covered)
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        if not inmath(m.start()): out.append((m.start(), m.end(), "Q"))
    for m in re.finditer(r"[A-Za-z]+", text):
        if inmath(m.start()): continue
        w = m.group(0).lower()
        if w in STOP or len(w) < 3: continue
        b = "V_rel" if w in V_REL else "N_math" if w in N_MATH else "Q" if w in Q_WORDS else "N_topic"
        out.append((m.start(), m.end(), b))
    return sorted(out)


def texts(which):
    # 끝 공백·줄바꿈 제거: Qwen3.5 채팅 템플릿이 본문 끝 '\n'을 지워 p0.index(text)가 실패한다 (v2 36문제 중 2개)
    if which == "v2":
        return [(int(r.pidx), r.problem.rstrip(), int(r.y)) for r in problem_table().itertuples()]
    rows, seen = [], set()
    for i, r in enumerate(csv.DictReader(open(AGG_CSV, newline="", encoding="utf-8"))):
        if r["problem_id"] in seen: continue
        seen.add(r["problem_id"]); rows.append((i, r["original_problem"].rstrip(), int(r["model_is_robust"].strip().lower() == "true")))
    return rows


def prompt_of(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{text}"}], tokenize=False, add_generation_prompt=True)


def forward_last_and_mid(tok, model, prompts, batch):
    """배치 forward. 반환: 마지막 토큰 L//2 벡터 (n, D) 와, 첫 프롬프트에 대해서만 전체 (T, D) + 오프셋."""
    import torch
    tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    outs = []
    i = 0
    while i < len(prompts):
        tmax = max(len(tok(p)["input_ids"]) for p in prompts[i:i + batch]); bs = max(1, min(batch, 1600 // tmax))   # 배치×토큰 상한 (VRAM 포화 방지)
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device); i += bs
        with torch.no_grad():
            hs = fwd(model, **enc, output_hidden_states=True, logits_to_keep=1).hidden_states
        mid = len(hs) // 2
        outs.append(hs[mid][:, -1].float().cpu())
    return torch.cat(outs)


def extract(model_id, which, batch=8):
    import torch
    tok = load_tokenizer(model_id); model = load_model(model_id, "4bit")
    rows = texts(which); recs = []; t0 = time.time()
    layer_sum, layer_cnt, store = None, 0, []
    for k, (pidx, text, y) in enumerate(rows, 1):
        us = units(text)
        if len(us) < 4:
            continue
        # 원문 forward: 마지막 토큰 벡터 + 단어 자리 벡터 (오프셋 매핑)
        p0 = prompt_of(tok, text); base = p0.index(text)
        enc = tok(p0, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k2: v.to(model.device) for k2, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        mid = len(hs) // 2; H = hs[mid][0].float().cpu(); last = H[-1]
        ALL = torch.stack([h[0] for h in hs]).float().cpu()                 # (L+1, T, D)  H10용
        wvec, uidx = [], []
        for (s, e, b) in us:
            idx = [i for i, (a, c) in enumerate(offs) if c > base + s and a < base + e]
            uidx.append(idx)
            wvec.append(H[idx].mean(0) if idx else torch.zeros_like(last))
        span = [i for i, (a, c) in enumerate(offs) if c > base and a < base + len(text)]
        layer_sum = layer_sum + ALL[:, span].sum(1) if layer_sum is not None else ALL[:, span].sum(1)
        layer_cnt += len(span)
        unit_layers = torch.stack([ALL[:, idx].mean(1) if idx else torch.zeros(ALL.shape[0], ALL.shape[2]) for idx in uidx]).half()  # (U, L+1, D)
        lastcol = ALL[:, -1].half()                                                                                           # (L+1, D)
        # 지우기 forward
        abl = [text[:s] + NEUTRAL[b] + text[e:] for (s, e, b) in us]
        L = forward_last_and_mid(tok, model, [prompt_of(tok, t) for t in abl], batch)
        imp = (1 - torch.nn.functional.cosine_similarity(L, last[None], dim=-1)).numpy().clip(min=0)
        buckets = np.array([b for _, _, b in us])
        is_topic = buckets == "N_topic"; is_struct = ~is_topic
        tot = imp.sum() + 1e-9
        surface_share = float(imp[is_topic].sum() / tot)
        count_share = float(is_topic.mean())
        def essence(mask):
            w = imp[mask]
            return (torch.stack([wvec[i] for i in np.where(mask)[0]]) * torch.tensor(w)[:, None]).sum(0) / (w.sum() + 1e-9) if mask.any() else torch.zeros_like(last)
        cs = lambda a, b: float(torch.nn.functional.cosine_similarity(a[None], b[None])[0])
        align = cs(last, essence(is_topic)) - cs(last, essence(is_struct))
        recs.append((pidx, surface_share, align, count_share, len(us), int(is_topic.sum()), len(offs), y))
        store.append({"pidx": pidx, "y": y, "imp": imp.astype(np.float32), "buckets": buckets, "unit_layers": unit_layers.numpy(), "last_layers": lastcol.numpy()})
        if k % 6 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)} units={len(us)} topic={int(is_topic.sum())} share={surface_share:.3f} align={align:+.3f} y={y}  {time.time() - t0:.0f}s", flush=True)
    R = np.array(recs, dtype=np.float32)
    out = RES / f"h9_{alias(model_id)}_{which}.npz"
    np.savez_compressed(out, pidx=R[:, 0].astype(int), share=R[:, 1], align=R[:, 2], cnt=R[:, 3], n_units=R[:, 4], n_topic=R[:, 5],
                        T=R[:, 6], y=R[:, 7].astype(np.int8), model=model_id)
    np.savez_compressed(RES / f"h9_layers_{alias(model_id)}_{which}.npz", store=np.array(store, dtype=object),
                        layer_mean=(layer_sum / max(1, layer_cnt)).half().numpy(), model=model_id)
    print(f"저장 {out}  {time.time() - t0:.0f}s"); del model; torch.cuda.empty_cache()


def evaluate(perms=30, which="v2"):
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0); lines = []
    def out(s=""): print(s); lines.append(s)
    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls); b, *_ = np.linalg.lstsq(X, x, rcond=None); return x - X @ b
    data = {m: np.load(RES / f"h9_{alias(m)}_{which}.npz", allow_pickle=True) for m in LOCAL_MODELS if (RES / f"h9_{alias(m)}_{which}.npz").is_file()}
    if not data:
        out("결과 없음"); return
    out(f"H9 액기스 분리 ({which}) · 사전등록: share·align 큼 = spurious (점수 = −지표) · null {perms}회")
    out(f"{'모델':12s} {'지표':6s} {'n':>3s} {'AUC':>6s} {'null95%':>13s} {'ρ(지표,cnt)':>12s} {'ρ(지표,logT)':>13s} {'cnt만':>6s} {'잔차AUC':>8s} {'잔차null':>13s}")
    pooled = {"share": [], "align": []}; vals = {"share": {}, "align": {}}
    for m, d in data.items():
        lab = d["y"] >= 0; y = d["y"][lab].astype(int); c = d["cnt"][lab].astype(float); T = np.log(d["T"][lab].astype(float))
        for key in ("share", "align"):
            x = d[key][lab].astype(float); sc = -x; auc = roc_auc_score(y, sc)
            null = [roc_auc_score(rng.permutation(y), sc) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
            rs = -resid(x, [T, c]); ra = roc_auc_score(y, rs)
            null2 = [roc_auc_score(rng.permutation(y), rs) for _ in range(perms)]; lo2, hi2 = np.percentile(null2, [2.5, 97.5])
            f1 = "▲" if auc > hi else ("▼" if auc < lo else " "); f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
            out(f"{alias(m)[:12]:12s} {key:6s} {len(y):3d} {auc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {spearmanr(x, c).correlation:+12.2f} {spearmanr(x, T).correlation:+13.2f} {roc_auc_score(y, -c):6.3f} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2}")
            pooled[key].append(rs / (rs.std() + 1e-9)); vals[key][m] = x; y_ref = y
    if len(data) >= 2:
        for key in ("share", "align"):
            P = np.mean(pooled[key], 0); null = [roc_auc_score(rng.permutation(y_ref), P) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
            a = roc_auc_score(y_ref, P)
            names = list(vals[key])
            corr = "  ".join(f"{alias(names[i])[:6]}-{alias(names[j])[:6]} {spearmanr(vals[key][names[i]], vals[key][names[j]]).correlation:+.2f}" for i in range(len(names)) for j in range(i + 1, len(names)))
            out(f"{'풀링(잔차)':12s} {key:6s} {len(y_ref):3d} {'':>6s} {'':>13s} {'':>12s} {'':>13s} {'':>6s} {a:8.3f} [{lo:.2f},{hi:.2f}]{'▲' if a > hi else ('▼' if a < lo else ' ')}   모델간 ρ: {corr}")
    (RES / f"h9_report_{which}.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--model"); ap.add_argument("--set", default="v2", choices=["v2", "math"])
    ap.add_argument("--perms", type=int, default=30)
    a = ap.parse_args()
    if a.extract:
        for m in ([a.model] if a.model else LOCAL_MODELS):
            extract(m, a.set)
    else:
        evaluate(a.perms, a.set)
