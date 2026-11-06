"""자체 라벨 생성 — **선별 단계만 로컬 GPU 에서** 돌린다 (지휘 #71-1).

왜 분업이 이렇게 되는가
  * Kaggle vLLM 은 **Qwen3.5 를 못 연다**(`'Qwen3_5Config' object has no attribute 'vocab_size'`, 지휘 #33 진단).
    그 모델은 어차피 HF 경로여야 하니 **로컬에서 돌리는 게 맞다** — Kaggle 시간을 아끼고 노트북이 놀지 않는다.
  * 로컬 4bit 배치 생성 실측은 단일 스트림 10 tok/s 가 아니라 **배치 8에서 합산 약 70 tok/s**
    (h21 gen 단계: 배치 8 × 1024토큰 = 112초). 선별(M=200·k=4·3,500토큰 ≈ 2.8M 토큰)이면 약 11시간 —
    Kaggle 로 치면 5.2 GPU-시간을 아낀다.
  * 생성 품질은 같다: 같은 프롬프트·같은 샘플링(T=0.6, top_p=0.95)·같은 정확도 규칙(미완주=오답, 개정 #37).
    4bit 양자화로 정확도가 조금 달라질 수 있다 — **그 모델의 라벨은 그 모델의 조건에서 난 것**이고,
    같은 모델을 Kaggle(fp16)과 섞지 않는다(섞으면 라벨 정의가 흔들린다).

출력은 Kaggle `genlab_v2.py --screen-glob` 이 그대로 읽는 스키마다(`n_hit` 명시 저장).
중간 저장하므로 멈췄다 이어서 돌릴 수 있다(`--resume`).

사용:
  .venv/Scripts/python.exe experiments/genlab_local_screen.py --model Qwen/Qwen3.5-4B --n 200 --k 4 --batch 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "kaggle"))
sys.path.insert(0, str(ROOT / "experiments"))
from genlab_v2 import BAND, acc_of, load_pool                      # noqa: E402
from solve_common import extract_answer                            # noqa: E402,F401  (acc_of 가 쓴다)
from variants_official import MATH_SPAN                            # noqa: E402,F401  (import 확인용)

OUT = ROOT / "experiments" / "results" / "genlab_local"
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8, help="동시 시퀀스 수(문제×k 가 이 값 이하가 되게 쪼갠다)")
    ap.add_argument("--max-new", type=int, default=6144)
    ap.add_argument("--source", default="aime,math5")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--resume", action="store_true", help="이미 저장된 문제는 건너뛴다")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    alias = a.model.split("/")[-1].replace(".", "_")
    f_out = OUT / f"{alias}_local_screen.json"

    probs = load_pool(a.n, a.source)
    done = {}
    if a.resume and f_out.is_file():
        done = {int(r["pidx"]): r for r in json.loads(f_out.read_text(encoding="utf-8"))}
        print(f"[이어서] 이미 {len(done)}문제 완료", flush=True)
    todo = [p for p in probs if p["pidx"] not in done]
    print(f"[로컬 선별] {a.model} · 남은 {len(todo)}/{len(probs)}문제 · k={a.k} · max_new={a.max_new}", flush=True)
    if not todo:
        return

    import torch
    from h21_uncertainty import load                               # 같은 로더(4bit·토크나이저 함정 처리 포함)
    tok, model = load(a.model, a.quant)

    per = max(1, a.batch // a.k)                                   # 한 번에 처리할 문제 수
    t0, store = time.time(), list(done.values())
    for st in range(0, len(todo), per):
        chunk = todo[st:st + per]
        prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{p['problem']}"}],
                                           tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(0)
        t1 = time.time()
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95,
                               num_return_sequences=a.k, max_new_tokens=a.max_new,
                               pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        ntok = int((new != tok.pad_token_id).sum())
        for i, p in enumerate(chunk):
            groups = []
            for r in range(a.k):
                row = new[i * a.k + r]
                live = row[row != tok.pad_token_id]
                txt = tok.decode(live, skip_special_tokens=True)
                fin = bool(len(live) < a.max_new)                  # EOS 로 끝났는가(상한에 닿지 않았는가)
                groups.append((txt, fin))
            acc, nfin, nhit = acc_of(groups, p["gold"], a.k)
            store.append({**p, "base_acc": acc, "n_fin": nfin, "n_hit": nhit,
                          "accepted": bool(BAND[0] <= acc <= BAND[1]),
                          "var_acc": {}, "var_detail": {}, "max_drop": None, "label": None,
                          "src": "local4bit"})
        f_out.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
        el = time.time() - t1
        acc_now = [s["base_acc"] for s in store if s["base_acc"] is not None]
        print(f"  [{len(store)}/{len(probs)}] {el:.0f}s · {ntok / max(1e-6, el):.1f} tok/s · "
              f"채택 {sum(1 for s in store if s['accepted'])} · 평균정확도 {sum(acc_now) / len(acc_now):.2f} · "
              f"누적 {(time.time() - t0) / 60:.0f}분", flush=True)
    print(f"[로컬 선별] 저장 {f_out} · 총 {(time.time() - t0) / 60:.0f}분", flush=True)


if __name__ == "__main__":
    main()
