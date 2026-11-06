"""H11 — 표면 해방 층 (attention knockout, 인과적 분리). 사전등록 2026-09-29.

근거: Qu & Gomes 2026 (arXiv 2609.17804) 의 '스키마 추상화 단계' — 그 논문은 모델 수준 분석, 우리는 (모델, 문제) 단위 지표.

정의
  표면 토큰      h9 단어목록(MATH137만 보고 작성)의 N_topic 단위가 차지하는 서브토큰 자리
  차단           층 l..L−1 에서 마지막 프롬프트 토큰(query) → 표면 토큰(key) attention 을 모든 head에서 −inf
  변화(l)        최종층 마지막 토큰 상태의 코사인 거리 · 다음토큰 분포 KL(기준‖차단), 기준 = 차단 없음
  해방층         변화(l) 이 처음으로 전층차단(l=0) 변화의 10% 이하가 되는 가장 이른 l, 정규화 l/L  (없으면 1.0)
  면적           Σ_l 변화(l) / (L · 변화(0))  — 표면 의존 총량 (정규화)
  대조군         같은 개수의 무작위 비표면 토큰을 같은 방식으로 차단 → 보정치 = 표면 − 대조군
부호 고정: 해방층 작음 = robust (점수 = −해방층). 보정치도 같은 부호.
N_topic 없는 문제 제외, 개수 보고. 속도: 짝수 층만, 층들을 배치로 묶어 forward(4bit, eager attention).
Qwen3.5-4B 는 선형 attention 층이 섞여 있어 마스크가 전 층에 안 걸린다 → 결과에 표기.

검문소: 섞기 null 30회 · 길이·표면토큰 개수 잔차화 · 모델 간 순위상관 · 통과 시에만 MATH137 DeepSeek 1회.

사용:  .venv/Scripts/python.exe experiments/h11_knockout.py --extract --model M [--set v2]
       .venv/Scripts/python.exe experiments/h11_knockout.py [--set v2]

H11b (--mode all, 사전등록 9/29 15:55, H11 평가 전): 층 l부터 끝까지 **모든 query 토큰**→표면 key 차단(표면 토큰 자기 자신의 query 행은 제외)
  = 층 l 이후 표면 정보 신규 유입 금지. H11은 마지막 토큰의 직접 경로만 막아 효과가 ≈0(간접 경로 미차단) → 설계 결함 의심.
  지표·대조·부호·검문소는 H11과 동일, **주지표 lib_kl**. 파일 접두 h11b_.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_hidden import alias, load_tokenizer  # noqa: E402
from extract_train_v2 import LOCAL_MODELS  # noqa: E402
from h9_essence import prompt_of, texts, units  # noqa: E402

RES = HERE / "results"
BATCH = 8
STRIDE = 2       # 층 간격 (지휘 #20: DeepSeek·Qwen3.5는 4 — 해방층은 l/L 정규화라 해상도만 바뀜)
TOK_BUDGET = 1600  # forward 1회당 배치×토큰 상한 → VRAM 포화(공유메모리 스왑) 방지, 배치 자동 축소
MODE = "last"    # "last": query=마지막 토큰만 차단 (H11) · "all": 모든 query→표면 key 차단, 표면 토큰 자기 자신은 제외 (H11b, 사전등록 9/29 15:55)
PREFIX = "h11"


def load_eager(model_id):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, low_cpu_mem_usage=True, device_map="auto", attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    layers = [m for n, m in model.named_modules() if n.endswith(".layers") and isinstance(m, torch.nn.ModuleList)]
    layers = max(layers, key=len)                      # 디코더 층 리스트
    return model, layers


class Knock:
    """층별 pre-hook: 샘플 b는 start[b] 이상 층에서 query=마지막 → key∈block[b] 를 −inf."""
    def __init__(self, layers):
        self.layers = layers; self.start = None; self.block = None; self.handles = []
        for i, lyr in enumerate(layers):
            self.handles.append(lyr.register_forward_pre_hook(self._hook(i), with_kwargs=True))

    def _hook(self, i):
        def f(module, args, kwargs):
            if self.start is None:
                return None
            import torch
            am = kwargs.get("attention_mask")
            if am is None or am.dim() != 4:
                return None
            am = am.clone()
            neg = torch.finfo(am.dtype).min
            for b, s in enumerate(self.start):
                if i >= s and len(self.block[b]):
                    blk = self.block[b]
                    if MODE == "all":
                        keep = am[b][:, blk, :].clone()          # 표면 토큰이 query일 때의 행은 그대로
                        am[b, :, :, blk] = neg
                        am[b][:, blk, :] = keep
                    else:
                        am[b, :, -1, blk] = neg
            kwargs["attention_mask"] = am
            return args, kwargs
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def run_conditions(tok, model, knock, ids, starts, block_sets):
    """같은 입력을 B개 복제해 조건별 마스크로 forward. 반환: 최종층 마지막 토큰 (B, D), logp (B, V)"""
    import torch
    outs_h, outs_p = [], []
    T = ids.shape[1]; bs = max(1, min(BATCH, TOK_BUDGET // T))
    for i in range(0, len(starts), bs):
        st, bl = starts[i:i + bs], block_sets[i:i + bs]
        B = len(st)
        inp = ids.repeat(B, 1)
        causal = torch.full((T, T), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16, device=ids.device).triu(1)
        am = causal[None, None].repeat(B, 1, 1, 1)            # 4D additive mask → 각 층 hook이 받는다
        knock.start, knock.block = st, bl
        with torch.no_grad():
            try:
                o = model(input_ids=inp, attention_mask=am, output_hidden_states=True, logits_to_keep=1)   # 전체 시퀀스 logits(B×T×V) 안 만듦
            except TypeError:
                o = model(input_ids=inp, attention_mask=am, output_hidden_states=True)
        knock.start = None
        outs_h.append(o.hidden_states[-1][:, -1].float().cpu()); outs_p.append(torch.log_softmax(o.logits[:, -1].float(), -1).cpu())
    return torch.cat(outs_h), torch.cat(outs_p)


def extract(model_id, which):
    import torch
    tok = load_tokenizer(model_id); model, layers = load_eager(model_id); L = len(layers); knock = Knock(layers)
    rng = np.random.default_rng(0); rows = texts(which); recs = []; t0 = time.time(); n_skip = 0
    Ls = list(range(0, L, STRIDE))
    for k, (pidx, text, y) in enumerate(rows, 1):
        p0 = prompt_of(tok, text); base = p0.index(text)
        enc = tok(p0, return_tensors="pt", return_offsets_mapping=True); offs = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(model.device); T = ids.shape[1]
        surf = sorted({i for (s, e, b) in units(text) if b == "N_topic" for i, (a, c) in enumerate(offs) if c > base + s and a < base + e})
        if not surf:
            n_skip += 1; continue
        span = [i for i, (a, c) in enumerate(offs) if c > base and a < base + len(text)]
        pool = [i for i in span if i not in surf and i != T - 1]
        ctrl = sorted(rng.choice(pool, size=min(len(surf), len(pool)), replace=False).tolist()) if pool else []
        # 조건: 기준(차단 없음) + 표면 차단 각 l + 대조 차단 각 l
        starts = [L] + Ls + Ls
        blocks = [[]] + [surf] * len(Ls) + [ctrl] * len(Ls)
        H, P = run_conditions(tok, model, knock, ids, starts, blocks)
        h0, p0v = H[0], P[0]
        cosd = (1 - torch.nn.functional.cosine_similarity(H[1:], h0[None], dim=-1)).numpy()
        kl = (p0v.exp()[None] * (p0v[None] - P[1:])).sum(-1).numpy()
        n = len(Ls); cs, cc = cosd[:n], cosd[n:]; ks, kc = kl[:n], kl[n:]
        def lib(curve):
            full = curve[0] if curve[0] > 1e-9 else np.nan
            if np.isnan(full): return np.nan, np.nan
            below = np.where(curve <= 0.1 * full)[0]
            return (Ls[below[0]] / L if len(below) else 1.0), float(curve.sum() / (n * full))
        ls_c, area_c = lib(cs); lc_c, area_cc = lib(cc); ls_k, area_k = lib(ks); lc_k, _ = lib(kc)
        recs.append((pidx, y, ls_c, lc_c, ls_k, lc_k, area_c, area_cc, area_k, len(surf), T, cs[0], cc[0]))
        if k % 6 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)} surf={len(surf)} T={T} lib_cos={ls_c:.2f} ctrl={lc_c:.2f} lib_kl={ls_k:.2f} full_cos={cs[0]:.3f} y={y}  {time.time() - t0:.0f}s", flush=True)
    knock.remove()
    R = np.array(recs, dtype=np.float32)
    out = RES / f"{PREFIX}_{alias(model_id)}_{which}.npz"
    np.savez_compressed(out, pidx=R[:, 0].astype(int), y=R[:, 1].astype(np.int8), lib_cos=R[:, 2], lib_cos_ctrl=R[:, 3], lib_kl=R[:, 4], lib_kl_ctrl=R[:, 5],
                        area_cos=R[:, 6], area_cos_ctrl=R[:, 7], area_kl=R[:, 8], n_surf=R[:, 9], T=R[:, 10], full_cos=R[:, 11], full_cos_ctrl=R[:, 12],
                        n_skip=n_skip, L=L, model=model_id, stride=STRIDE)
    print(f"저장 {out}  제외(N_topic 없음) {n_skip}  {time.time() - t0:.0f}s"); del model; torch.cuda.empty_cache()


def evaluate(perms=30, which="v2"):
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0); lines = []
    def out(s=""): print(s); lines.append(s)
    def resid(x, ctrls):
        X = np.column_stack([np.ones(len(x))] + ctrls); b, *_ = np.linalg.lstsq(X, x, rcond=None); return x - X @ b
    data = {m: np.load(RES / f"{PREFIX}_{alias(m)}_{which}.npz", allow_pickle=True) for m in LOCAL_MODELS if (RES / f"{PREFIX}_{alias(m)}_{which}.npz").is_file()}
    if not data:
        out("결과 없음"); return
    out(f"{PREFIX.upper()} 표면 해방 층 ({which}, MODE={MODE}{', 주지표 lib_kl' if MODE == 'all' else ''}) · 사전등록: 해방층 작음 = robust (점수 = −지표) · null {perms}회")
    out(f"{'모델':12s} {'지표':14s} {'n':>3s} {'AUC':>6s} {'bacc':>6s} {'null95%':>13s} {'ρ(n_surf)':>9s} {'ρ(logT)':>8s} {'잔차AUC':>8s} {'잔차null':>13s}")
    keys = [("lib_cos", None), ("lib_cos−ctrl", ("lib_cos", "lib_cos_ctrl")), ("lib_kl", None), ("lib_kl−ctrl", ("lib_kl", "lib_kl_ctrl")), ("area_cos", None), ("area_cos−ctrl", ("area_cos", "area_cos_ctrl"))]
    pooled = {k: [] for k, _ in keys}; vals = {k: {} for k, _ in keys}; y_ref = None
    for m, d in data.items():
        lab = d["y"] >= 0; y = d["y"][lab].astype(int); ns = d["n_surf"][lab].astype(float); T = np.log(d["T"][lab].astype(float))
        out(f"-- {alias(m)}: 제외 {int(d['n_skip'])}, L={int(d['L'])}, stride={int(d['stride']) if 'stride' in d else 2}, 전층차단 cos 평균 {float(np.nanmean(d['full_cos'][lab])):.3f} (대조 {float(np.nanmean(d['full_cos_ctrl'][lab])):.3f})")
        for key, pair in keys:
            x = (d[pair[0]][lab] - d[pair[1]][lab]) if pair else d[key][lab]
            x = x.astype(float); ok = np.isfinite(x)
            if ok.sum() < 6 or y[ok].min() == y[ok].max():
                out(f"{alias(m)[:12]:12s} {key:14s} {ok.sum():3d}  (부족)"); continue
            yy, xx, nn, tt = y[ok], x[ok], ns[ok], T[ok]; sc = -xx
            auc = roc_auc_score(yy, sc); bacc = balanced_accuracy_score(yy, (sc > np.median(sc)).astype(int))
            null = [roc_auc_score(rng.permutation(yy), sc) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
            rs = -resid(xx, [tt, nn]); ra = roc_auc_score(yy, rs)
            null2 = [roc_auc_score(rng.permutation(yy), rs) for _ in range(perms)]; lo2, hi2 = np.percentile(null2, [2.5, 97.5])
            f1 = "▲" if auc > hi else ("▼" if auc < lo else " "); f2 = "▲" if ra > hi2 else ("▼" if ra < lo2 else " ")
            out(f"{alias(m)[:12]:12s} {key:14s} {len(yy):3d} {auc:6.3f} {bacc:6.3f} [{lo:.2f},{hi:.2f}]{f1} {spearmanr(xx, nn).correlation:+9.2f} {spearmanr(xx, tt).correlation:+8.2f} {ra:8.3f} [{lo2:.2f},{hi2:.2f}]{f2}")
            if ok.all():
                pooled[key].append(rs / (rs.std() + 1e-9)); vals[key][m] = xx; y_ref = yy
    if y_ref is not None:
        for key, _ in keys:
            if len(pooled[key]) >= 2:
                P = np.mean(pooled[key], 0); null = [roc_auc_score(rng.permutation(y_ref), P) for _ in range(perms)]; lo, hi = np.percentile(null, [2.5, 97.5])
                au = roc_auc_score(y_ref, P); names = list(vals[key])
                corr = "  ".join(f"{alias(names[i])[:6]}-{alias(names[j])[:6]} {spearmanr(vals[key][names[i]], vals[key][names[j]]).correlation:+.2f}" for i in range(len(names)) for j in range(i + 1, len(names)))
                out(f"{'풀링(잔차)':12s} {key:14s} {len(y_ref):3d} {'':>6s} {'':>6s} {'':>13s} {'':>9s} {'':>8s} {au:8.3f} [{lo:.2f},{hi:.2f}]{'▲' if au > hi else ('▼' if au < lo else ' ')}   ρ: {corr}")
    (RES / f"{PREFIX}_report_{which}.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--model"); ap.add_argument("--set", default="v2", choices=["v2", "math"]); ap.add_argument("--perms", type=int, default=30); ap.add_argument("--mode", default="last", choices=["last", "all"]); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    MODE = a.mode; STRIDE = a.stride; BATCH = a.batch; PREFIX = "h11" if MODE == "last" else "h11b"
    if a.extract:
        for m in ([a.model] if a.model else LOCAL_MODELS):
            extract(m, a.set)
    else:
        evaluate(a.perms, a.set)
