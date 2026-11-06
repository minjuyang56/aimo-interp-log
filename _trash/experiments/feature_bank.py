"""전 레이어 hidden 요약 은행 — 라벨이 도착하면 층 스윕을 바로 돌리기 위한 사전 추출. (지휘 #35-3)

무엇: 이미 만들어 둔 생성문(genlab 커널의 <alias>_gen.json 안 gen_token_ids)을 teacher forcing 으로 한 번 통과시켜
      **모든(기본 2층 간격) 층**의 32토큰 청크 평균을 저장한다. 생성 시점 hidden 과 같은 벡터임은
      `experiments/tests_teacher_forcing.py` 로 검증됐다(코사인 1.000000).
왜: H10-B/C 는 L//2 한 층만 봤다. 라벨이 생기면 어느 층이 좋은지 스윕해야 하는데, 그때 다시 생성하지 않으려면 지금 뽑아 둬야 한다.
    (층 선택을 라벨 보고 하면 데이터 드레징이므로, 스윕 결과는 **탐색적**으로만 보고하고 주지표는 사전등록대로 L//2 를 쓴다.)

저장: results/fbank/<alias>_<태그>.npz  — store: [{pidx, prompt_tokens, gen_tokens, chunks(L', C, D) fp16, layers(L',)}]
      L' = 0,2,4,… 층 인덱스(hidden_states 기준). 용량 때문에 2층 간격이 기본.
체크포인트: 문제마다 npz 갱신. 이미 처리한 pidx 는 건너뛴다.

사용:  .venv/Scripts/python.exe experiments/feature_bank.py --model Skywork/Skywork-OR1-Math-7B \
           --json experiments/results/genlab/aimo-genlab-skywork-or1-math-7b-run/genlab/Skywork-OR1-Math-7B_gen.json --tag run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from variants_nonofficial import SOLVE_PROMPT  # noqa: E402

RES = HERE / "results" / "fbank"
CHUNK = 32


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M')}] {m}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    with (RES / "progress.txt").open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M')}] {m}\n")


def load_model(model_id, quant="4bit"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(device_map="auto", low_cpu_mem_usage=True)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    return tok, AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--stride", type=int, default=2, help="층 간격 (용량 절약)")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--window", type=int, default=1024, help="창 단위 forward 길이 (VRAM 보호)")
    a = ap.parse_args()
    window = a.window
    alias = a.model.split("/")[-1].replace(".", "_")
    out = RES / f"{alias}_{a.tag}.npz"
    RES.mkdir(parents=True, exist_ok=True)

    src = json.loads(Path(a.json).read_text(encoding="utf-8"))
    src = [s for s in src if s.get("gen_token_ids") or s.get("greedy_text")]
    done = {}
    if out.is_file():
        d = np.load(out, allow_pickle=True)
        done = {int(r["pidx"]): r for r in d["store"]}
        log(f"이어서: 이미 {len(done)}건")
    log(f"{a.model} 로드 ({a.quant}) · 대상 {len(src)}건 · 층 간격 {a.stride}")
    tok, model = load_model(a.model, a.quant)
    hs_n = model.config.num_hidden_layers + 1
    keep = list(range(0, hs_n, a.stride))
    log(f"hidden_states {hs_n}개 중 {len(keep)}개 저장")

    store = list(done.values())
    t0 = time.time()
    for i, s in enumerate(src, 1):
        if int(s["pidx"]) in done:
            continue
        try:
            pid = tok(tok.apply_chat_template([{"role": "user", "content": f"{SOLVE_PROMPT}\n\n{s['problem']}"}],
                                              tokenize=False, add_generation_prompt=True), return_tensors="pt")["input_ids"][0].tolist()
            gid = list(s["gen_token_ids"]) if s.get("gen_token_ids") else tok(s["greedy_text"], add_special_tokens=False)["input_ids"]
            full = pid + gid
            P = len(pid)
            # 창 단위 forward (past_key_values 이어받기) — 9천 토큰을 한 번에 넣으면 전층 hidden 이 VRAM 을 넘긴다.
            # 커널의 fp16 전량 forward 가 38문제 중 19건에서 OutOfMemoryError 로 죽은 원인이 이것이다.
            rows = {li: [] for li in keep}
            past = None
            for st_ in range(0, len(full), window):
                seg = full[st_:st_ + window]
                with torch.no_grad():
                    o = model(input_ids=torch.tensor([seg], device=model.device), past_key_values=past,
                              use_cache=True, output_hidden_states=True, logits_to_keep=1)
                past = o.past_key_values
                lo = max(0, P - st_)                      # 이 창에서 생성 구간이 시작되는 위치
                if lo < len(seg):
                    for li in keep:
                        rows[li].append(o.hidden_states[li][0, lo:].float().cpu())
                del o
            mats = []
            for li in keep:
                H = torch.cat(rows[li]) if rows[li] else torch.zeros(0, 1)
                C = [H[j:j + CHUNK].mean(0) for j in range(0, len(H), CHUNK)]
                mats.append(torch.stack(C) if C else torch.zeros(0, H.shape[-1]))
            arr = torch.stack(mats).numpy().astype(np.float16) if mats and len(mats[0]) else np.zeros((len(keep), 0, 1), np.float16)
            store.append({"pidx": int(s["pidx"]), "prompt_tokens": P, "gen_tokens": len(gid),
                          "chunks": arr, "layers": np.array(keep), "label_official": s.get("label_official", s.get("label")),
                          "label_unofficial": s.get("label_unofficial"), "finished": s.get("finished"), "correct": s.get("correct")})
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"  실패 pidx={s['pidx']}: {type(e).__name__} {str(e)[:80]}")
            torch.cuda.empty_cache()
            continue
        np.savez_compressed(out, store=np.array(store, dtype=object), model=a.model, stride=a.stride, chunk=CHUNK)
        if len(store) % 10 == 0 or i == len(src):
            log(f"  {len(store)}/{len(src)} · {(time.time() - t0) / 60:.1f}분 · "
                f"문제당 {(time.time() - t0) / max(1, len(store) - len(done)):.1f}s · {out.stat().st_size / 1e6:.0f} MB")
    log(f"완료 {len(store)}건 · 저장 {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
