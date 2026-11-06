"""H12 — 교환 개입 (activation patching). 사전등록 2026-09-29.

두 실행: 원문제 · 표면 변형(N_topic 단위: 대문자 단어→고정 이름, 소문자 단어→결정적 오타. surface_variant 참고).
  * 기존 edit_rename은 수식 안 기호만 바꿔 표면 토큰이 그대로라 이식 효과가 0에 가깝다 → 표면 단위만 건드리는 변형을 쓴다.
층 l마다: 변형 실행의 표면 토큰 자리 잔차(층 l 출력)를 원문제 실행의 같은 자리에 이식 → 최종 마지막 토큰 상태 코사인 변화 · 다음토큰 KL.
정렬: 단위(이름)별 서브토큰 평균으로 맞춤(양쪽 토큰 수가 달라도 됨). 표면 단위가 0개면 제외·개수 보고.
지표 = 층 평균 이식 효과 (표면 의존 인과량), 큼 = spurious (점수 = −지표). 짝수 층만, 층을 배치로.
검문소: null 30 · 길이·표면개수 잔차화 · 모델 간 순위상관 · 통과 시만 MATH137 DeepSeek 1회.

사용:  .venv/Scripts/python.exe experiments/h12_patch.py --extract --model M [--set v2]
       .venv/Scripts/python.exe experiments/h12_patch.py [--set v2]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import LOCAL_MODELS, load_model  # noqa: E402
from h9_essence import prompt_of, texts, units  # noqa: E402


def locate(p0, text):
    return p0.index(text)
from h11_knockout import load_eager  # noqa: E402

RES = HERE / "results"; BATCH = 8; TOK_BUDGET = 1600   # forward 1회당 배치×토큰 상한 (VRAM 포화 방지)


def fwd(model, **kw):
    """logits_to_keep 미지원 모델(TypeError)이면 빼고 다시 호출."""
    try:
        return model(**kw)
    except TypeError:
        kw.pop("logits_to_keep", None); return model(**kw)

NAMES = ["Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy"]


def surface_variant(text):
    """표면 변형 (사전등록, 결과 보기 전 확정 9/29 15:20):
    N_topic 단위 중 대문자 시작 단어 → 고정 이름 목록(등장 순서, 같은 단어는 같은 이름),
    소문자 단어(4자 이상) → 안쪽 두 글자 교환(결정적 오타). 그 외 단위는 그대로.
    이유: 대문자 이름만 바꾸면 v2 13/36·MATH 12/137만 남는다(커버리지 부족). 오타는 답을 보존하고 토큰만 바꾼다.
    반환: 새 텍스트, 바뀐 단위 수"""
    us = [(s, e) for s, e, b in units(text) if b == "N_topic"]
    mapping, out, pos, k, n = {}, [], 0, 0, 0
    for s, e in us:
        w = text[s:e]
        if w[0].isupper():
            if w not in mapping:
                mapping[w] = NAMES[k % len(NAMES)]; k += 1
            rep = mapping[w]
        elif len(w) >= 4 and w[1] != w[2]:
            rep = w[0] + w[2] + w[1] + w[3:]
        else:
            rep = w
        if rep != w: n += 1
        out.append(text[pos:s]); out.append(rep); pos = e
    out.append(text[pos:])
    return "".join(out), n


surface_rename = surface_variant  # 호환


class Patch:
    """층 l 출력에서 샘플 b의 지정 자리에 벡터를 덮어쓴다."""
    def __init__(self, layers):
        self.plan = None; self.handles = [lyr.register_forward_hook(self._hook(i)) for i, lyr in enumerate(layers)]

    def _hook(self, i):
        def f(module, args, output):
            if self.plan is None: return None
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            for b, (layer, pos, vec) in enumerate(self.plan):
                if layer == i:
                    hs[b, pos] = vec.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return f

    def remove(self):
        for h in self.handles: h.remove()


def unit_positions(tok, text, base, offs):
    return [[i for i, (a, c) in enumerate(offs) if c > base + s and a < base + e] for (s, e, b) in units(text) if b == "N_topic"]


def extract(model_id, which):
    import torch
    done = RES / f"h12_{alias(model_id)}_{which}.npz"
    if done.is_file():                      # 체인이 겹쳐 돌아 같은 모델을 두 번 뽑는 낭비를 막는다 (9/29)
        print(f"이미 있음, 건너뜀: {done}", flush=True); return
    tok = load_tokenizer(model_id); model, layers = load_eager(model_id); L = len(layers); patch = Patch(layers)
    rows = texts(which); recs = []; t0 = time.time(); n_skip = 0; Ls = list(range(0, L, 2))
    for k, (pidx, text, y) in enumerate(rows, 1):
        var, nmap = surface_variant(text)
        if nmap == 0:
            n_skip += 1; continue
        outs = []
        for t in (text, var):
            p0 = prompt_of(tok, t); base = locate(p0, t)
            enc = tok(p0, return_tensors="pt", return_offsets_mapping=True); offs = enc.pop("offset_mapping")[0].tolist()
            ids = enc["input_ids"].to(model.device)
            with torch.no_grad():
                o = fwd(model, input_ids=ids, output_hidden_states=True, logits_to_keep=1)
            outs.append((ids, o.hidden_states, unit_positions(tok, t, base, offs), o.hidden_states[-1][0, -1].float().cpu(), torch.log_softmax(o.logits[0, -1].float(), -1).cpu()))
        (ids0, hs0, pos0, h_last0, lp0), (ids1, hs1, pos1, _, _) = outs
        if len(pos0) != len(pos1) or any(not a or not b for a, b in zip(pos0, pos1)):
            n_skip += 1; continue
        effects_c, effects_k = [], []
        bs = max(1, min(BATCH, TOK_BUDGET // ids0.shape[1]))
        for i in range(0, len(Ls), bs):
            chunk = Ls[i:i + bs]; B = len(chunk)
            plan = []
            for l in chunk:
                # hidden_states[l+1] = 층 l 출력. 변형 실행의 단위 평균을 원문 실행의 단위 자리 전부에 이식
                pos_all, vec_all = [], []
                for a, b in zip(pos0, pos1):
                    v = hs1[l + 1][0, b].float().mean(0)
                    pos_all += a; vec_all += [v] * len(a)
                plan.append((l, torch.tensor(pos_all), torch.stack(vec_all)))
            patch.plan = plan
            with torch.no_grad():
                o = fwd(model, input_ids=ids0.repeat(B, 1), output_hidden_states=True, logits_to_keep=1)
            patch.plan = None
            h = o.hidden_states[-1][:, -1].float().cpu(); lp = torch.log_softmax(o.logits[:, -1].float(), -1).cpu()
            effects_c += (1 - torch.nn.functional.cosine_similarity(h, h_last0[None], dim=-1)).tolist()
            effects_k += (lp0.exp()[None] * (lp0[None] - lp)).sum(-1).tolist()
        n_surf = sum(len(a) for a in pos0)
        recs.append((pidx, y, float(np.mean(effects_c)), float(np.mean(effects_k)), float(np.max(effects_c)), n_surf, ids0.shape[1], nmap))
        if k % 6 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)} names={nmap} surf_tok={n_surf} eff_cos={np.mean(effects_c):.4f} eff_kl={np.mean(effects_k):.4f} y={y}  {time.time() - t0:.0f}s", flush=True)
    patch.remove()
    R = np.array(recs, dtype=np.float32)
    out = RES / f"h12_{alias(model_id)}_{which}.npz"
    np.savez_compressed(out, pidx=R[:, 0].astype(int), y=R[:, 1].astype(np.int8), eff_cos=R[:, 2], eff_kl=R[:, 3], eff_max=R[:, 4], n_surf=R[:, 5], T=R[:, 6], n_names=R[:, 7], n_skip=n_skip, L=L, model=model_id)
    print(f"저장 {out}  제외 {n_skip}  {time.time() - t0:.0f}s"); del model; torch.cuda.empty_cache()


def evaluate(perms=30, which="v2"):
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0); lines = []
    def out(s=""): print(s); lines.append(s)
    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls); b, *_ = np.linalg.lstsq(X, x, rcond=None); return x - X @ b
    data = {m: np.load(RES / f"h12_{alias(m)}_{which}.npz", allow_pickle=True) for m in LOCAL_MODELS if (RES / f"h12_{alias(m)}_{which}.npz").is_file()}
    if not data: out("결과 없음"); return
    out(f"H12 교환 개입 ({which}) · 사전등록: 이식 효과 큼 = spurious (점수 = −효과) · null {perms}회")
    out(f"{'모델':12s} {'지표':8s} {'n':>3s} {'AUC':>6s} {'bacc':>6s} {'null95%':>13s} {'ρ(n_surf)':>9s} {'ρ(logT)':>8s} {'잔차AUC':>8s} {'잔차null':>13s}")
    keys = ["eff_cos", "eff_kl", "eff_max"]; pooled = {k: [] for k in keys}; vals = {k: {} for k in keys}; y_ref = None
    for m, d in data.items():
        lab = d["y"] >= 0; y = d["y"][lab].astype(int); ns = d["n_surf"][lab].astype(float); T = np.log(d["T"][lab].astype(float))
        out(f"-- {alias(m)}: 제외 {int(d['n_skip'])}, L={int(d['L'])}")
        for key in keys:
            x = d[key][lab].astype(float); sc = -x
            auc = roc_auc_score(y, sc); bacc = balanced_accuracy_score(y, (sc > np.median(sc)).astype(int))
            null = [roc_auc_score(rng.permutation(y), sc) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
            rs = -resid(x, [T, ns]); ra = roc_auc_score(y, rs)
            null2 = [roc_auc_score(rng.permutation(y), rs) for _ in range(perms)]; lo2, hi2 = np.percentile(null2, [2.5, 97.5])
            f1 = "▲" if auc > hi else ("▼" if auc < lo else " "); f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
            out(f"{alias(m)[:12]:12s} {key:8s} {len(y):3d} {auc:6.3f} {bacc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {spearmanr(x, ns).correlation:+9.2f} {spearmanr(x, T).correlation:+8.2f} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2}")
            pooled[key].append(rs / (rs.std() + 1e-9)); vals[key][m] = x; y_ref = y
    if y_ref is not None:
        for key in keys:
            if len(pooled[key]) >= 2 and all(len(v) == len(y_ref) for v in pooled[key]):
                P = np.mean(pooled[key], 0); null = [roc_auc_score(rng.permutation(y_ref), P) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5]); au = roc_auc_score(y_ref, P)
                names = list(vals[key]); corr = "  ".join(f"{alias(names[i])[:6]}-{alias(names[j])[:6]} {spearmanr(vals[key][names[i]], vals[key][names[j]]).correlation:+.2f}" for i in range(len(names)) for j in range(i + 1, len(names)) if len(vals[key][names[i]]) == len(vals[key][names[j]]))
                out(f"{'풀링(잔차)':12s} {key:8s} {len(y_ref):3d} {'':>6s} {'':>6s} {'':>13s} {'':>9s} {'':>8s} {au:8.3f} [{lo:.2f},{hi:.2f}]{'▲' if au > hi else ('▼' if au < lo else ' ')}   ρ: {corr}")
    (RES / f"h12_report_{which}.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--model"); ap.add_argument("--set", default="v2", choices=["v2", "math"]); ap.add_argument("--perms", type=int, default=30)
    a = ap.parse_args()
    if a.extract:
        for m in ([a.model] if a.model else LOCAL_MODELS): extract(m, a.set)
    else:
        evaluate(a.perms, a.set)
