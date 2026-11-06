"""#ORG-LABEL 시험2 — sub2048 제출물 프로브를 **주최측 공개 라벨**(공식 33)에 전이 (지휘 #ORG-LABEL).

## 왜 값이 있나
`dist/sub2048-small.zip` 은 지금 업로드 후보이고(#SUB-GEN2048) 그 가중치는 **자체 라벨 93행**으로만
적합됐다. 주최측 라벨에서 한 번도 시험해 본 적이 없다. 그리고 우리 자체 라벨 문제와 공식 36문제는
**겹침 0**(8-그램 자카드 0.000, `genlab_v2.load_pool:74` 가 배제)이므로 **학습·테스트 중복이
구조적으로 0** 이다 — 지휘의 '우리 학습셋과 겹치는 문제 제외' 가 자동 충족된다.

## 특징 경로는 학습과 **글자 그대로** 같아야 한다
`experiments/probe_gen.py` 가 학습 npz 를 만든 그 경로를 그대로 쓴다:
  * 프롬프트 = `apply_chat_template([user: SOLVE_PROMPT + "\\n\\n" + 문제], add_generation_prompt=True)`
  * **따로 토크나이즈해 이어 붙인다** — `tok(prompt).ids + tok(풀이).ids[:2048]`.
    한 번에 세면 경계 토큰이 합쳐져 풀이 시작 위치가 밀린다.
  * 층 = `hidden_states[L//2]`(sub2048 고정) · 풀이 구간 평균 · **로컬 4bit nf4**(학습과 같은 정밀도)
**주장하지 않고 게이트로 증명한다**(`--check`): 우리 학습 greedy 텍스트 몇 행을 이 스크립트의
특징 함수에 넣어 학습 npz 와 코사인·최대절대차를 맞댄다. 어긋나면 평가하지 않고 멈춘다.

## 평가 (사전등록 #ORG-LABEL)
주판정 = 공식 **문제단위 다수결 라벨 33** 에서 모델별 AUC + 모델평균 · 셔플 null 1000.
보조 = 모델 자기 라벨 행(n<10 이면 판정 불가) · 고정 임계 `thr` 의 정확도·True 비율 ·
`log(단어수)` 잔차 · 기준선 h15C. 부호: 점수 큼 = robust (뒤집지 않는다).

사용:
  .venv/Scripts/python.exe analysis/org_sub2048_transfer.py --check      # 게이트만
  .venv/Scripts/python.exe analysis/org_sub2048_transfer.py --extract --model <mid>
  .venv/Scripts/python.exe analysis/org_sub2048_transfer.py --eval
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))

ART = ROOT / "solutions/sub2048/artifacts/sub2048.json"
GEN = ROOT / "experiments/results/runpod/genprobe/se44xjeoxn49hw/out"
TRAIN_GEN = ROOT / "experiments/results/runpod/genprobe/oi8q4v8ia3ojq0/out"
FEAT = ROOT / "experiments/results/probe"
OUT = ROOT / "experiments/results/org_sub2048_transfer.txt"
WIN = 2048
MODELS = ["Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


# ───────────────────────── 특징 (probe_gen.py 와 같은 경로) ─────────────────────────

def extract(mid, rows, layer_rule="L//2", quant="4bit"):
    """반환 (특징 행렬, 층 인덱스, 쓴 풀이 토큰 수 목록)."""
    import inspect

    import torch
    from h21_uncertainty import SOLVE_PROMPT, load
    tok, model = load(mid, quant)
    L = model.config.num_hidden_layers
    layer = L // 2 if layer_rule == "L//2" else int(layer_rule)
    # **로짓을 만들지 않는다.** forward 는 기본으로 B×T×V 로짓을 낸다 — DeepSeek 는 V=151k 라
    # T≈2,300 에서 **약 0.7~1.4 GB** 다. 8 GB VRAM 에 nf4 가중치 6.4 GB 를 올린 상태에서 그게
    # 공유 GPU 메모리(시스템 RAM)로 스필해 2026-10-15 에 36분을 먹고도 안 끝났다(H11 스왑 사고와
    # 같은 양식). 우리는 hidden state 만 쓰므로 로짓은 순수 낭비다.
    # **hidden state 는 이 인자에 영향받지 않는다** — 그래도 주장하지 않고 `--check` 게이트가
    # 학습 npz 와 코사인·최대절대차로 다시 증명한다.
    pnames = set(inspect.signature(type(model).forward).parameters)
    kw = {}
    for cand in ("logits_to_keep", "num_logits_to_keep"):
        if cand in pnames:
            kw[cand] = 1
            break
    print(f"[추출] {mid} · 층 {L} → hidden_states[{layer}] · 행 {len(rows)} · "
          f"로짓 절약 {kw or '지원 안 됨(그대로 간다)'}", flush=True)
    vecs, nsol, t0 = [], [], time.time()
    for n, r in enumerate(rows, 1):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{r['problem']}"}],
            tokenize=False, add_generation_prompt=True)
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        sid = tok(r["greedy_text"], add_special_tokens=False)["input_ids"][:WIN]
        if not sid:
            raise SystemExit(f"pidx {r.get('pidx')} 의 풀이가 비었다")
        nsol.append(len(sid))
        ids = torch.tensor([pid + sid], device=model.device)
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True)
        s = len(pid)
        vecs.append(o.hidden_states[layer][0, s:s + len(sid), :].float().mean(0).cpu().numpy())
        del o
        torch.cuda.empty_cache()
        if n % 5 == 0 or n == len(rows):
            print(f"  [추출] {n}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.stack(vecs), layer, nsol


def probe_score(art_m, X):
    """저장 가중치 적용: scaler → PCA → 로지스틱 결정함수. sub2048_lib 와 같은 식."""
    z = (X - np.asarray(art_m["mean"])) / np.asarray(art_m["scale"])
    p = (z - np.asarray(art_m["pca_mean"])) @ np.asarray(art_m["comps"]).T
    return p @ np.asarray(art_m["coef"]) + float(art_m["intercept"])


# ───────────────────────── 게이트: 학습 특징과 같은 경로인가 ─────────────────────────

def gate(nrow=3):
    """학습 greedy 텍스트를 이 스크립트의 특징 함수에 넣어 학습 npz 와 맞댄다."""
    mid = "Skywork/Skywork-OR1-Math-7B"
    npz = np.load(FEAT / "Skywork-OR1-Math-7B_gen2048.npz", allow_pickle=True)
    rows_all = json.loads((TRAIN_GEN / "Skywork-OR1-Math-7B_greedy.json").read_text(encoding="utf-8"))
    by = {r["pidx"]: r for r in rows_all}
    pid = [int(p) for p in npz["pidx"]][:nrow]
    rows = [by[p] for p in pid]
    L = 28                                            # Skywork hidden layers (아래에서 실측 확인)
    X, layer, _ = extract(mid, rows)
    ref = npz[f"h{L // 2}"][:nrow]
    cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(X, ref)]
    mad = float(np.abs(X - ref).max())
    out("## 게이트 — 특징 경로가 학습(sub2048 가중치를 적합한 그 npz)과 같은가")
    out(f"   Skywork {nrow}행 · pidx {pid} · 층 hidden_states[{layer}]")
    out(f"   코사인 {['%.6f' % c for c in cos]} · 최대절대차 {mad:.3e}")
    ok = min(cos) > 0.999999 and mad < 1e-3
    out(f"   → {'통과' if ok else '**실패 — 평가하지 않는다**'}")
    return ok


# ───────────────────────── 평가 ─────────────────────────

def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


def band(y, s, perms=1000, seed=0):
    """순열 null: 점수를 고정하고 라벨만 섞는다 → 밴드는 n·클래스비율로 정해진다."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    v = np.array([auc(rng.permutation(y), s) for _ in range(perms)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def resid(x, basis):
    """basis(상수항 포함)로 x 를 회귀하고 잔차를 돌려준다."""
    A = np.column_stack([np.ones(len(x)), basis])
    b, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()

    if a.check:
        raise SystemExit(0 if gate() else 1)

    if a.extract:
        # **모델당 별 프로세스로 돌린다.** 한 프로세스에서 셋을 연달아 올리면 8 GB VRAM 이
        # 단편화돼 세 번째(DeepSeek 8B nf4 ≈ 6.4 GB)가 `device_map="auto"` 에서
        # ValueError("Some modules are dispatched on the CPU or the disk") 로 죽는다 —
        # 2026-10-15 실측. `del model` + `empty_cache()` 로는 안 풀린다.
        if not a.model:
            import subprocess
            for mid in MODELS:
                alias = mid.split("/")[-1].replace(".", "_")
                if (FEAT / f"{alias}_org33.npz").is_file():
                    print(f"[추출] {alias} 이미 있음 → 건너뜀", flush=True)
                    continue
                print(f"[추출] 별 프로세스로 {mid}", flush=True)
                r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    "--extract", "--model", mid])
                if r.returncode != 0:
                    raise SystemExit(f"{mid} 추출 실패 (rc={r.returncode})")
            return
        for mid in [a.model]:
            alias = mid.split("/")[-1].replace(".", "_")
            rows = json.loads((GEN / f"{alias}_greedy.json").read_text(encoding="utf-8"))
            X, layer, nsol = extract(mid, rows)
            p = FEAT / f"{alias}_org33.npz"
            np.savez_compressed(
                p, X=X, layer=layer, pidx=np.array([r["pidx"] for r in rows]),
                n_sol_used=np.array(nsol), n_tok=np.array([r["n_tok"] for r in rows]),
                problems=np.array([r["problem"] for r in rows], dtype=object))
            print(f"[추출] 저장 {p} · {X.shape} · 풀이토큰 중앙 {sorted(nsol)[len(nsol) // 2]}",
                  flush=True)
        return

    if not a.eval:
        raise SystemExit("--check / --extract / --eval 중 하나를 줘라")

    # ---------- 라벨 ----------
    import pandas as pd
    from h21_uncertainty import problems
    probs = problems()                                 # 문제단위 다수결 라벨 33
    ptext = {r["problem"]: r for r in probs}
    df = pd.read_parquet(ROOT / "experiments/results/train-main-v2.parquet")
    art = json.loads(ART.read_text(encoding="utf-8"))

    out("#ORG-LABEL 시험2 — sub2048 프로브를 주최측 공개 라벨에 전이 (사전등록 1회)")
    out(f"   가중치 {ART.relative_to(ROOT).as_posix()} · 창 {art['win']} · PCA {art['npc']} · "
        f"층 {art['layer_rule']} · 고정 임계")
    out(f"   테스트 = 공식 train-main-v2 문제단위 라벨 **{len(probs)}문제** "
        f"(robust {sum(r['y'] for r in probs)} / spurious {len(probs) - sum(r['y'] for r in probs)})")
    out("   학습·테스트 중복: **0** (우리 자체 라벨 문제와 겹침 0 — 자카드 0.000)")
    out()

    rowsum = {}
    for mid in MODELS:
        alias = mid.split("/")[-1].replace(".", "_")
        p = FEAT / f"{alias}_org33.npz"
        if not p.is_file():
            out(f"## {alias} — 특징 파일 없음 ({p.name}) → 건너뜀")
            continue
        d = np.load(p, allow_pickle=True)
        am = art["per_model"][mid]
        if int(d["layer"]) != int(am["layer"]):
            raise SystemExit(f"층 불일치: 특징 {int(d['layer'])} vs 가중치 {am['layer']}")
        sc = probe_score(am, d["X"])
        pt = [str(x) for x in d["problems"]]
        y = np.array([ptext[t]["y"] for t in pt])
        w = np.array([math.log(max(1, len(t.split()))) for t in pt])
        A = auc(y, sc)
        lo, hi = band(y, sc, a.perms)
        Ar = auc(y, resid(sc, w))
        thr = float(am["thr"])
        pred = sc >= thr
        acc = float((pred == (y == 1)).mean())
        out(f"## {alias} · 층 {int(d['layer'])} · n {len(y)}")
        out(f"   **AUC {A:.3f}** · 셔플 null 95% [{lo:.2f}, {hi:.2f}] "
            f"{'▲' if A > hi else ''}{' ▼' if A < lo else ''}")
        out(f"   log(단어수) 잔차 AUC {Ar:.3f}  (게이트 아님 · 같은 밴드)")
        out(f"   고정 임계 {thr:+.4f} → 정확도 {acc:.3f} · True 비율 {pred.mean():.3f} "
            f"(train True 비율 {am['train_pred_true_rate']:.3f})")
        out(f"   참고 train: 자체라벨 n {am['n']} (robust {am['n_robust']}) · "
            f"내부CV AUC {am['cv_auc']:.3f} · train AUC {am['train_auc']:.3f}")
        # 모델 자기 라벨 행 (보조)
        g = df[df.model_id == mid]
        own = {str(r.problem).rstrip(): bool(r.is_robust) for r in g.itertuples()
               if r.is_robust is not None and str(r.is_robust) != "nan"}
        sel = [(s, own[t]) for s, t in zip(sc, pt) if t in own]
        if len(sel) < 10:
            out(f"   [보조] 모델 자기 라벨 행 {len(sel)}개 < 10 → **판정 불가**")
        else:
            yo = np.array([int(b) for _, b in sel])
            so = np.array([s for s, _ in sel])
            lo2, hi2 = band(yo, so, a.perms)
            out(f"   [보조] 모델 자기 라벨 {len(sel)}행 (robust {int(yo.sum())}) · "
                f"AUC {auc(yo, so):.3f} · null [{lo2:.2f}, {hi2:.2f}]")
        rowsum[alias] = dict(auc=A, lo=lo, hi=hi, res=Ar, acc=acc, tr=float(pred.mean()))
        out()

    # ---------- 기준선 h15C + Qwen 자리 ----------
    sys.path.insert(0, str(ROOT / "solutions/h15C"))
    import h15_lib
    pt33 = [r["problem"] for r in probs]
    y33 = np.array([r["y"] for r in probs])
    s15 = np.array(h15_lib.scores(pt33, "C"))
    l15, h15 = band(y33, s15, a.perms)
    thr15 = h15_lib.load_art()["thr"]["C"]
    out("## 기준선 h15C (−log 단어수, train 고정 임계) · Qwen3.5 자리")
    out(f"   AUC {auc(y33, s15):.3f} · null [{l15:.2f}, {h15:.2f}] · "
        f"임계 {thr15:+.4f} → 정확도 {float(((s15 >= thr15) == (y33 == 1)).mean()):.3f}")
    out("   **이 값은 in-sample 이다** — h15C 임계를 바로 이 33문제에서 고정했다(`n_train` 33).")
    out("   Qwen3.5-4B 는 sub2048 가중치가 없어(자체 라벨 0개) **프로브 판정 불가**이고,")
    out("   제출물에서 그 모델은 이 h15C 폴백으로 떨어진다.")
    out()

    # ---------- 주판정 ----------
    if rowsum:
        ms = float(np.mean([v["auc"] for v in rowsum.values()]))
        mh = float(np.mean([v["hi"] for v in rowsum.values()]))
        mr = float(np.mean([v["res"] for v in rowsum.values()]))
        npass = sum(1 for v in rowsum.values() if v["auc"] > v["hi"])
        out("## 주판정 (사전등록: 둘 다 만족해야 '우연 이상')")
        out(f"   1. 모델평균 AUC **{ms:.3f}** vs null 상단 평균 {mh:.2f} → "
            f"{'통과' if ms > mh else '**미달**'}")
        out(f"   2. 모델별 null 초과 **{npass}/{len(rowsum)}모델** (기준 2 이상) → "
            f"{'통과' if npass >= 2 else '**미달**'}")
        ok = ms > mh and npass >= 2
        out(f"   → **{'전이 입증' if ok else '전이 미입증'}**")
        out(f"   (병기) 모델평균 길이잔차 {mr:.3f} · 모델평균 고정임계 정확도 "
            f"{float(np.mean([v['acc'] for v in rowsum.values()])):.3f} · "
            f"True 비율 {float(np.mean([v['tr'] for v in rowsum.values()])):.3f}")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
