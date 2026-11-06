"""프로브 파일럿 — 문제 텍스트 **forward 1회**의 hidden state 추출. 로컬 4bit. (지휘 #75-2)

왜 forward 1회인가: 주최측이 원하는 건 내부 신호이고(README·8/3 Discord), 생성이 없으면 제출 비용이
문제당 0.05초 수준이라 3600초 예산에 거의 영향이 없다. H16 에서 같은 구조를 33문제로 돌렸을 때는
라벨이 모자라 AUC 0.531(null 안)이었다 — 이번에는 자체 생성 라벨 **88개**로 다시 본다.

추출 대상 두 묶음 (같은 모델·같은 풀링):
  1) 자체 라벨 88문제 (Skywork run0+run50 병합) → 학습·교차검증
  2) 공식 train-main-v2 문제 단위 라벨 33문제 → **외부 테스트**(학습에 절대 안 씀)
풀링: 문제 토큰 구간 평균(채팅 템플릿 포함 전체 프롬프트가 아니라 **문제 본문 토큰만**).
층: L//4 · L//2 · 3L//4 · 마지막. 층 선택은 분석 쪽 폴드 내부 CV 로만 한다.

사용:  .venv/Scripts/python.exe experiments/probe_pilot.py --model Skywork/Skywork-OR1-Math-7B
출력:  experiments/results/probe/<별칭>_{own,official}.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from h21_uncertainty import SOLVE_PROMPT, load, problems                 # noqa: E402

OUT = ROOT / "experiments" / "results" / "probe"
MERGED = ROOT / "experiments" / "results" / "genlab" / "skywork_labels_merged.json"


def own_rows(paths=None):
    """자체 라벨을 읽는다. 여러 파일을 주면 **pidx 로 중복 제거하며 합친다**(슬라이스가 여럿이다).
    `label_major`·`gen_k`·`gen_cap` 을 같이 실어 보낸다 — Amendment 2 가 주분석을 gen_k==8 로
    한정하고 과반 규칙을 민감도 열로 쓰기 때문에, 특징 파일에 그 메타가 있어야 평가에서 가를 수 있다."""
    files = [Path(x) for x in (paths or [MERGED])]
    seen, out = set(), []
    for f in files:
        for r in json.loads(f.read_text(encoding="utf-8")):
            if not isinstance(r.get("label"), bool) or r["pidx"] in seen:
                continue
            seen.add(r["pidx"])
            out.append({"problem": r["problem"], "label": r.get("label"),
                        "label_fin": r.get("label_fin"), "label_major": r.get("label_major"),
                        "pidx": r["pidx"], "max_drop": r.get("max_drop"),
                        "gen_k": r.get("gen_k"), "gen_cap": r.get("gen_cap")})
    return out


def official_rows():
    """공식 train-main-v2 문제단위 라벨 — **외부 테스트 전용**(학습에 절대 안 쓴다).
    주최측 라벨은 과반 규칙으로 만든 것이므로 label_major 자리에도 같은 값을 넣는다."""
    return [{"problem": r["problem"], "label": bool(r["y"]), "label_fin": bool(r["y"]),
             "label_major": bool(r["y"]), "pidx": r["pidx"], "max_drop": None,
             "gen_k": None, "gen_cap": None} for r in problems()]


def r5screen_rows():
    """#R5-TRANSFER (a) — `analysis/r5_rows.py` 가 고른 screen 전 문제. 라벨 대신 **base_acc**(정답률)를
    타깃으로 쓴다. 채택·탈락을 모두 포함하는 것이 요점이다(채택분만 쓰면 `base_acc >= 0.5` 필터가
    정답률 변이를 지워 버린다 — 실측 표준편차 0.07~0.17 vs 전 문제 0.21~0.28)."""
    import os
    src = ROOT / "experiments" / "results" / "r5_rows.json"
    store = json.loads(src.read_text(encoding="utf-8"))
    want = os.environ.get("R5_MODEL", "")
    key = next((k for k in store if k == want or k.endswith(want.split("/")[-1])), None)
    if key is None:
        raise SystemExit(f"r5_rows.json 에 {want} 가 없다 — 키: {list(store)}")
    return [{"problem": r["problem"], "label": r["base_acc"] >= 0.5, "label_fin": None,
             "label_major": None, "pidx": r["pidx"], "max_drop": None,
             "gen_k": r.get("k"), "gen_cap": None, "base_acc": r["base_acc"]}
            for r in store[key]]


def extract(tok, model, rows, layers, tag):
    """문제 본문 토큰 구간만 평균 풀링. 프롬프트 안에서 본문의 위치를 토큰 오프셋으로 찾는다."""
    import torch
    feats = {l: [] for l in layers}
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        full = tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{r['problem']}"}],
                                       tokenize=False, add_generation_prompt=True)
        # 본문 시작 위치: 프롬프트 문자열에서 문제 본문이 시작하는 문자 인덱스 → 토큰 인덱스로 환산
        cpos = full.rfind(r["problem"][:40])
        pre = tok(full[:cpos] if cpos > 0 else full, return_tensors="pt", add_special_tokens=False)["input_ids"]
        enc = tok(full, return_tensors="pt", add_special_tokens=False).to(model.device)
        s = int(pre.shape[1]) if cpos > 0 else 0
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for l in layers:
            # hidden_states[i] 는 **i-1 번 층의 출력**이다(9/30 확인). 여기서는 인덱스를 그대로 기록하고
            # 보고서에 hidden_states 인덱스로 명시한다 — 층 번호를 섞어 쓰면 재현이 안 된다.
            h = out.hidden_states[l][0, s:, :].float().mean(0).cpu().numpy()
            feats[l].append(h)
        del out
        torch.cuda.empty_cache()
        if n % 20 == 0 or n == len(rows):
            print(f"  [{tag}] {n}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    return {f"h{l}": np.stack(feats[l]) for l in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Skywork/Skywork-OR1-Math-7B")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="자체 라벨 json 경로들(슬라이스가 여럿이면 여러 개). 생략하면 Skywork 병합본.")
    ap.add_argument("--tags", nargs="*", default=["own", "official"],
                    help="추출할 묶음 (own / official)")
    ap.add_argument("--suffix", default="",
                    help="출력 파일명 접미사. **다른 라벨 묶음을 뽑을 때 꼭 주라** — 안 주면 "
                         "<별칭>_own.npz 를 덮어써 기존 특징을 잃는다(실제로 그럴 뻔했다).")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    alias = a.model.split("/")[-1].replace(".", "_")
    tok, model = load(a.model, a.quant)
    L = model.config.num_hidden_layers
    layers = sorted({max(1, L // 4), L // 2, (3 * L) // 4, L})
    print(f"[probe] {a.model} · 층 {L} · 추출 인덱스 {layers}", flush=True)
    avail = {"own": lambda: own_rows(a.labels), "official": official_rows,
             "r5screen": r5screen_rows}
    for tag in a.tags:
        rows = avail[tag]()
        print(f"[probe] {tag}: {len(rows)}문제", flush=True)
        feats = extract(tok, model, rows, layers, tag)
        np.savez_compressed(OUT / f"{alias}_{tag}{a.suffix}.npz", layers=np.array(layers),
                            y=np.array([int(r["label"]) for r in rows]),
                            # label_fin 은 **None 일 수 있다** (완주만 분모로 쓰면 중간대역에 들어가는 행).
                            # Olmo off40 에서 실제로 났다 — int(None) 으로 죽었다. y_major 와 같은 규약
                            # (-1 = 그 규칙으로는 라벨 없음)으로 싣고, 평가 쪽이 그 행을 뺀다.
                            y_fin=np.array([-1 if r.get("label_fin") is None
                                            else int(r["label_fin"]) for r in rows]),
                            y_major=np.array([-1 if r.get("label_major") is None
                                               else int(r["label_major"]) for r in rows]),
                            gen_k=np.array([0 if r.get("gen_k") is None else int(r["gen_k"])
                                            for r in rows]),
                            gen_cap=np.array([0 if r.get("gen_cap") is None else int(r["gen_cap"])
                                              for r in rows]),
                            pidx=np.array([r["pidx"] for r in rows]),
                            words=np.array([len(r["problem"].split()) for r in rows]),
                            base_acc=np.array([-1.0 if r.get("base_acc") is None
                                               else float(r["base_acc"]) for r in rows]),
                            problems=np.array([r["problem"] for r in rows], dtype=object), **feats)
        print(f"[probe] 저장 {OUT / f'{alias}_{tag}{a.suffix}.npz'}", flush=True)


if __name__ == "__main__":
    main()
