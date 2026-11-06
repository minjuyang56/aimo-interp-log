"""genlab v2 — 주최측식 max_drop 라벨 (지휘 #33, 사전등록 2026-09-30).

왜 다시 짜나: v1 라벨은 spurious 가 2% 밖에 안 나와(85건 중 2건) 판정이 불가능했다. 원인은
  (a) 문제가 쉬워 원본 정확도가 1.0 이고 (b) 라벨 기준이 "변형에서 한 번이라도 틀리면 spurious" 인 이진 판정이라
  모델이 안정적으로 맞히는 문제만 남았다. 주최측 정의(여러 번 풀려 정확도 하락폭)를 그대로 쓴다.

절차 (모델당)
  0. 문제풀: AIME(AI-MO/aimo-validation-aime) + 선택적으로 MATH level 5. 정수 정답, v2 36문제와 중복 제거.
  1. **선별**: 원본을 k회 샘플(T=0.6, top_p=0.95) → base_acc. **0.25 ≤ base_acc ≤ 0.75 만 채택**
     (너무 쉬우면 변형이 안 무너지고, 너무 어려우면 원본부터 못 푼다 → 둘 다 라벨이 안 갈린다)
  2. **변형**: 공식 3종(rename·typos·distract) 각 1개, 변형마다 k회 샘플 → var_acc
  3. **라벨**: max_drop = base_acc − min(var_acc) → **≤0.10 robust · ≥0.25 spurious · 그 사이 제외** (train-main-v2 규칙과 동일)
  4. **궤적**: 채택된 문제의 원본을 greedy 1회 생성 → HF teacher forcing forward 1회로 L//2 32토큰 청크 평균
     (H10-B·H10-C 와 같은 정의. layers[mid-1] 훅 — hidden_states[i] 는 i-1층 출력이라 여기서 어긋나면 안 된다)

v1 실패 원인 수정 (지휘 #33-B)
  * **max_model_len 고정**: vLLM 이 모델 기본 최대길이(DeepSeek 131072 · Olmo 65536)로 KV 캐시를 잡으려다 죽었다 → prompt+max_new 에 맞춰 지정.
  * **HF 폴백**: vLLM 이 아키텍처를 못 읽으면(Qwen3.5: Qwen3_5Config has no attribute vocab_size) transformers generate 로 돌린다.

분석은 사전등록 그대로: traced_ratio L//2(큼=robust, 청크수·문제길이 잔차화) · compass 양방향 · 문제 단위 permutation null 1000.
"""

import argparse
import io
import os as _os
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

AIME_URL = "https://huggingface.co/datasets/AI-MO/aimo-validation-aime/resolve/main/data/train-00000-of-00001.parquet"
MATH_URL = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main/{s}/test-00000-of-00001.parquet"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "intermediate_algebra", "number_theory", "prealgebra", "precalculus", "geometry"]
V2_URL = "https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/main/data/train-00000-of-00001.parquet"
BOXED_INT = re.compile(chr(92) * 2 + r"boxed\s*\{\s*(-?[\d,]+)\s*\}")
CHUNK = 32
BAND = (0.5, 1.0)          # 개정 #37-1: base_acc >= 0.5 (1.0 도 max_drop 측정 가능)
DROP_ROBUST, DROP_SPUR = 0.10, 0.25


def _get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read()


def load_pool(n_problems, source="aime", seed=0):
    import pandas as pd
    rows = []
    if "aime" in source:
        df = pd.read_parquet(io.BytesIO(_get(AIME_URL)))
        for r in df.itertuples():
            try:
                rows.append({"problem": r.problem.strip(), "gold": int(str(r.answer).strip()), "level": "AIME", "subject": "aime"})
            except Exception:
                pass
    if "math5" in source:
        for s in MATH_SUBJECTS:
            try:
                df = pd.read_parquet(io.BytesIO(_get(MATH_URL.format(s=s))))
            except Exception:
                continue
            for r in df.itertuples():
                if r.level != "Level 5":
                    continue
                m = BOXED_INT.search(r.solution)
                if m:
                    rows.append({"problem": r.problem.strip(), "gold": int(m.group(1).replace(",", "")), "level": "Level 5", "subject": s})
    cache = Path("train-main-v2.parquet")
    if not cache.is_file():
        cache.write_bytes(_get(V2_URL))
    v2 = set(p.strip() for p, _ in pd.read_parquet(cache).groupby("problem", sort=False))
    rows = [r for r in rows if r["problem"] not in v2]
    idx = np.random.default_rng(seed).permutation(len(rows))[:n_problems]
    out = [rows[i] for i in sorted(idx)]
    for i, r in enumerate(out):
        r["pidx"] = i
    print(f"문제풀 {len(out)}개 (후보 {len(rows)} · source={source})", flush=True)
    return out


# ---------------------------------------------------------------- 생성 백엔드
class Gen:
    """vLLM 우선, 실패하면 HF. samples(prompts, k, temp) -> [[텍스트…]] · finished 플래그."""

    def __init__(self, model_id, max_new, max_seqs=48, max_len=None):
        self.model_id, self.max_new = model_id, max_new
        self.kind = None
        self.max_len = max_len or (max_new + 2048)
        try:
            import torch
            from vllm import LLM
            self.llm = LLM(model=model_id, dtype="half", tensor_parallel_size=max(1, torch.cuda.device_count()),
                           max_num_seqs=max_seqs, max_model_len=self.max_len, gpu_memory_utilization=0.90,
                           trust_remote_code=True, seed=0)
            self.tok = self.llm.get_tokenizer(); self.kind = "vllm"
            # **디토큰화를 vLLM 에 맡기지 않는다** (#TOK-AUDIT). 같은 token_ids 를 올바른
            # 토크나이저로 다시 디코드한다. 계측용 누적 카운터도 여기서 0 으로 둔다.
            self.dtok = safe_detok(model_id)
            self.m_raw = self.m_safe = self.n_diff = self.n_out = 0
            print(f"[Gen] vLLM · max_model_len={self.max_len} · vLLM토크나이저 "
                  f"{type(self.tok).__name__} · 디코드 "
                  f"{type(self.dtok).__name__ if self.dtok is not None else 'o.text(폴백)'}", flush=True)
        except Exception as e:
            print(f"[Gen] vLLM 실패 ({type(e).__name__}: {str(e)[:120]}) → HF 폴백", flush=True)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
            try:
                self.tok = PreTrainedTokenizerFast.from_pretrained(model_id)
            except Exception:
                self.tok = AutoTokenizer.from_pretrained(model_id)
            self.tok.padding_side = "left"
            if self.tok.pad_token_id is None:
                self.tok.pad_token = self.tok.eos_token
            self.hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto",
                                                           low_cpu_mem_usage=True).eval()
            self.kind = "hf"

    def samples(self, prompts, k, temp, batch=8):
        """반환: [(텍스트, 완주여부)] * k 를 프롬프트마다."""
        if self.kind == "vllm":
            from vllm import SamplingParams
            # **디코딩 민감도 오버라이드 (지휘 #TEMP-SENS, 2026-10-12).** 환경변수가 없으면
            # 아래 두 줄은 아무것도 하지 않고 기존 호출과 **글자 그대로 같다** — Kaggle 라벨과
            # RunPod 라벨이 같은 로직으로 만들어진다는 성질은 그대로다.
            # 바깥(엔트리)에서 `vllm.SamplingParams` 를 감싸려다 실패했다: vllm 0.31.0 이
            # lazy import 중 그 이름을 `SamplingParams | PoolingParams` 로 평가하므로
            # 클래스 자리에 함수를 끼우면 TypeError 로 죽는다. 그래서 **생성 지점**에서 바꾼다.
            # greedy(temp == 0) 경로는 건드리지 않는다.
            _kw = {}
            if temp > 0 and float(_os.environ.get("TS_TEMP", 0) or 0) > 0:
                temp = float(_os.environ["TS_TEMP"])
                _tk = int(_os.environ.get("TS_TOPK", 0) or 0)
                if _tk > 0:
                    _kw["top_k"] = _tk
                print(f"[#TEMP-SENS] SamplingParams 오버라이드: temperature={temp} "
                      f"top_k={_kw.get('top_k', '없음')} (top_p·max_tokens·n·seed 불변)", flush=True)
            sp = SamplingParams(n=k, temperature=temp, top_p=0.95 if temp > 0 else 1.0, max_tokens=self.max_new, seed=0, **_kw)
            outs = self.llm.generate(prompts, sp)
            res = []
            for out in outs:
                grp = []
                for o in out.outputs:
                    t, mr, ms, diff = decode_out(o, getattr(self, "dtok", None))
                    self.m_raw += mr; self.m_safe += ms; self.n_diff += int(diff); self.n_out += 1
                    grp.append((t, o.finish_reason == "stop"))
                res.append(grp)
            print(f"[디토큰화 계측] 샘플 누적 {self.n_out} · vLLM o.text 마커 {self.m_raw} · "
                  f"쓰는 텍스트 마커 {self.m_safe} · 두 텍스트가 다른 샘플 {self.n_diff}", flush=True)
            return res, sum(len(o.token_ids) for out in outs for o in out.outputs)
        import torch
        res, ntok = [], 0
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.hf.device)
            with torch.no_grad():
                out = self.hf.generate(**enc, do_sample=temp > 0, temperature=temp if temp > 0 else None,
                                       top_p=0.95 if temp > 0 else None, num_return_sequences=k,
                                       max_new_tokens=self.max_new, pad_token_id=self.tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for j in range(len(chunk)):
                grp = []
                for r in new[j * k:(j + 1) * k]:
                    nz = int((r != self.tok.pad_token_id).sum()); ntok += nz
                    grp.append((self.tok.decode(r, skip_special_tokens=True), bool((r == self.tok.eos_token_id).any())))
                res.append(grp)
        return res, ntok

    def close(self):
        import gc
        import torch
        for a in ("llm", "hf"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect(); torch.cuda.empty_cache()


def chat(tok, text, alt=0):
    sysmsg = SOLVE_PROMPT_ALT2 if alt == 2 else (SOLVE_PROMPT_ALT if alt else SOLVE_PROMPT)   # noqa: F821
    return tok.apply_chat_template([{"role": "user", "content": f"{sysmsg}\n\n{text}"}], tokenize=False, add_generation_prompt=True)


BPE_MARKS = {"Ġ": " ", "Ċ": "\n", "ĉ": "\t"}   # Ġ Ċ ĉ — 바이트 BPE 마커


def debpe(t):
    """vLLM 디토큰화 손상 복구용. **2026-10-11 사고**: vLLM 이 DeepSeek-R1-0528-Qwen3-8B 를
    옛 LlamaTokenizer 경로로 디코드해 공백을 `Ġ`, 줄바꿈을 `Ċ` 로 남겼다
    (Skywork · Olmo 는 0건, DeepSeek 는 41/41 전행). 그 결과 boxed 안의 선행 공백이
    **`Ġ90`** 으로 추출돼 same_answer 가 False → **정답이 오답으로 집계**됐고,
    DeepSeek 라벨의 base_acc 가 약 0.28 낮게 기록됐다(갭 0.280 ≈ 오판율 0.293).
    라벨은 텍스트를 저장하지 않아 사후 복원이 불가능했다 — 그래서 **집계 전에** 막는다."""
    for k, v in BPE_MARKS.items():
        t = t.replace(k, v)
    return t


def safe_detok(model_id):
    """**디토큰화를 vLLM 에 맡기지 않는다** (2026-10-11 #TOK-AUDIT 의 근본 수정).

    vLLM 은 DeepSeek-R1-0528-Qwen3-8B 의 `tokenizer_config.json` 을 보고 옛 Llama 계열 경로로
    디토큰화해 공백을 `Ġ`, 줄바꿈을 `Ċ` 로 남겼다 → `oxed{ 90}` 이 `Ġ90` 으로 추출돼
    **정답이 오답으로 집계**됐다(DeepSeek 41/41 전행 · 오판율 0.293 · base_acc 약 0.28 과소).
    `debpe` 는 **집계 단계의 사후 수습**이고, 이 함수는 **텍스트가 생기는 단계**를 고친다.

    근거(감사에서 측정): 손상 텍스트를 옛 토크나이저로 재인코딩하면 vLLM 의 토큰 id 가
    복원되고(길이비 0.999~1.000) 그 **같은 id** 를 `PreTrainedTokenizerFast` 로 디코드하면
    마커가 0건이 된다 → **vocab/id 는 정상이고 decode 단계만 달랐다.** 그래서 vLLM 이 주는
    `o.token_ids` 를 올바른 토크나이저로 다시 디코드하면 손상이 원천에서 사라진다.
    못 열리면 None 을 돌려주고 호출부가 `o.text` 로 떨어진다(그 경우 debpe 가 2차 방어)."""
    try:
        from transformers import PreTrainedTokenizerFast
        t = PreTrainedTokenizerFast.from_pretrained(model_id)
        print(f"[디토큰화] 안전 경로 확보 {type(t).__name__} — o.token_ids 를 다시 디코드한다", flush=True)
        return t
    except Exception as e:
        print(f"[디토큰화] !! PreTrainedTokenizerFast 실패({type(e).__name__}) → o.text 를 쓴다 "
              f"(debpe 가 집계 단계에서 막는다)", flush=True)
        return None


def marks_in(t):
    return sum(t.count(m) for m in BPE_MARKS)


def decode_out(o, dtok):
    """vLLM 출력 하나의 텍스트. 안전 토크나이저가 있으면 **토큰 id 에서 다시 디코드**한다.
    반환 (텍스트, 원본 o.text 의 마커 수, 새 텍스트의 마커 수, 두 텍스트가 다른가)."""
    raw = o.text
    if dtok is None:
        return raw, marks_in(raw), marks_in(raw), False
    try:
        fixed = dtok.decode(list(o.token_ids), skip_special_tokens=True)
    except Exception:
        return raw, marks_in(raw), marks_in(raw), False
    return fixed, marks_in(raw), marks_in(fixed), fixed != raw


def same_answer(pred, gold):
    """정답 비교. **추출기가 str 을 주든 int 를 주든 같게 판정한다.**
    2026-10-06 사고: gold 는 int, inline 된 solve_common.extract_answer 는 str 을 돌려줘
    `"42" == 42` 가 항상 False → 파일럿 20문제가 전부 0/80 이 됐다. 비교를 구현에 의존시키면 안 된다."""
    if pred is None:
        return False
    pred = debpe(str(pred))
    try:
        return int(str(pred).replace(",", "").strip()) == int(str(gold).replace(",", "").strip())
    except (TypeError, ValueError):
        return str(pred).strip() == str(gold).strip()


def acc_of(groups, gold, k=None):
    """정확도 = (완주 & 정답) / k. **미완주는 오답** (개정 #37: 완주만 분모로 쓰면 선택 효과로 1.0 에 몰린다).
    반환: acc, n_fin(완주 수), n_hit(완주&정답 수)"""
    k = k or len(groups)
    n_fin = sum(1 for _, f in groups if f)
    if any(("Ġ" in t or "Ċ" in t) for t, _ in groups):
        print("[경고] 생성 텍스트에 BPE 마커가 있다 — 디토큰화 손상. debpe 로 걷어내고 채점한다.",
              flush=True)
    n_hit = sum(1 for t, f in groups if f and same_answer(extract_answer(debpe(t)), gold))   # noqa: F821
    return n_hit / max(1, k), n_fin, n_hit


def _dump_texts(path, rows):
    """샘플 텍스트를 쓰고 **마커를 세어 같이 적는다** — 손상이 또 생기면 파일만 보고 알 수 있게."""
    nm = sum(marks_in(sm["text"]) for r in rows for sm in r["samples"])
    Path(path).write_text(json.dumps({"n_rows": len(rows), "n_marks": nm, "rows": rows}), encoding="utf-8")
    mb = Path(path).stat().st_size / 1e6
    print(f"[텍스트 저장] {Path(path).name} · {len(rows)}행 · "
          f"샘플 {sum(len(r['samples']) for r in rows)} · BPE 마커 {nm} · {mb:.1f} MB", flush=True)


def run(model_id, out_dir, n_pool, k, max_new, source, max_seqs, do_traj=True, screen_glob=None, cap=0,
        pool_offset=0, save_text=True):
    # save_text: **원본·변형 샘플 텍스트를 전부 따로 저장한다** (지휘 #DS-RELABEL).
    #   #TOK-AUDIT 에서 DeepSeek 라벨이 디토큰화 손상 위에서 집계됐는데 **텍스트가 없어
    #   사후 복원이 불가능**했다. 집계(n_hit·base_acc)만 남기면 집계의 버그는 보이지 않는다.
    #   라벨 JSON 을 키우지 않으려고 *_texts.json 으로 분리해 쓴다(다운스트림 코드 불변).
    alias = model_id.split("/")[-1].replace(".", "_")
    if pool_offset:                     # 슬라이스마다 다른 파일명 (안 하면 서로 덮어쓴다)
        alias = f"{alias}_off{pool_offset}"
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # **풀 슬라이스**: 같은 seed 풀에서 앞 pool_offset 개를 건너뛴다. 커널 하나가 Kaggle 9시간 세션에
    # 들도록 쪼개는 용도다(cap 16384 에서 모델당 16h 가 걸려 한 커널로는 못 끝낸다).
    probs = load_pool(n_pool + pool_offset, source)[pool_offset:]
    if pool_offset:
        first = probs[0]["pidx"] if probs else "-"
        print(f"[슬라이스] 앞 {pool_offset}개 건너뜀 → 이번 커널 {len(probs)}문제 (pidx {first} 부터)", flush=True)
    reuse = None
    if screen_glob:
        import glob as _glob
        hits = sorted(_glob.glob(screen_glob, recursive=True))
        if hits:
            reuse = {int(r["pidx"]): r for r in json.loads(Path(hits[-1]).read_text(encoding="utf-8"))}
            print(f"[선별 재사용] {hits[-1]} · {len(reuse)}문제 (개정 #37 규칙으로 base_acc 재계산)", flush=True)
        else:
            print(f"[선별 재사용] 파일 없음: {screen_glob} → 새로 선별", flush=True)
    g = Gen(model_id, max_new, max_seqs)
    t0 = time.time()

    if reuse is not None:
        store = []
        for p in probs:
            r = reuse.get(p["pidx"])
            if r is None:
                continue
            # **n_hit 가 있으면 그걸 쓴다.** 예전엔 base_acc×n_fin 으로 되돌렸는데, 그건 base_acc 가
            # 완주분모(옛 규칙)일 때만 맞다. 로컬 선별 파일은 개정 #37 규칙(n_hit/k)으로 저장되므로
            # 되돌리면 값이 깨진다. 옛 파일도 n_hit 를 저장했으니 둘 다 안전하다.
            nfin = int(r.get("n_fin", 0))
            nhit = int(r["n_hit"]) if r.get("n_hit") is not None else int(round((r.get("base_acc") or 0) * nfin))
            acc = nhit / max(1, k)
            store.append({**p, "base_acc": acc, "n_fin": nfin, "n_hit": nhit, "accepted": bool(BAND[0] <= acc <= BAND[1]),
                          "var_acc": {}, "var_detail": {}, "max_drop": None, "label": None})
        kept = [s for s in store if s["accepted"]]
        import collections as _c
        print(f"[선별 재사용] base_acc 분포 {dict(sorted(_c.Counter(round(s['base_acc'], 2) for s in store).items()))} · "
              f"대역{BAND} 채택 **{len(kept)}**", flush=True)
        if cap and len(kept) > cap:                     # 예산 초과 시 base_acc 1.0 군을 무작위 축소
            rng = np.random.default_rng(0)
            top = [s for s in kept if s["base_acc"] >= 0.999]
            rest = [s for s in kept if s["base_acc"] < 0.999]
            keep_top = list(rng.permutation(len(top))[:max(0, cap - len(rest))])
            kept = rest + [top[i] for i in sorted(keep_top)]
            print(f"[예산] 채택을 {len(kept)}개로 축소 (1.0 군 {len(top)} -> {len(kept) - len(rest)})", flush=True)
        (Path(out_dir) / f"{model_id.split('/')[-1].replace('.', '_')}_v3_screen.json").write_text(json.dumps(store), encoding="utf-8")
        return _variants_and_labels(g, model_id, out_dir, kept, k, do_traj)

    # 1. 선별
    groups, ntok = g.samples([chat(g.tok, p["problem"]) for p in probs], k, 0.6)
    print(f"[선별] {len(probs)}문제 × k={k} · {time.time() - t0:.0f}s · {ntok / max(1, time.time() - t0):.1f} tok/s", flush=True)
    store = []
    for p, grp in zip(probs, groups):
        a, nfin, nhit = acc_of(grp, p["gold"], k)
        store.append({**p, "base_acc": a, "n_fin": nfin, "n_hit": nhit, "accepted": bool(BAND[0] <= a <= BAND[1]),
                      "var_acc": {}, "var_detail": {}, "max_drop": None, "label": None})
    acc_list = [s["base_acc"] for s in store if s["base_acc"] is not None]
    kept = [s for s in store if s["accepted"]]
    print(f"  정확도 분포: n={len(acc_list)} · 0 {sum(a == 0 for a in acc_list)} · 1 {sum(a == 1 for a in acc_list)} · "
          f"대역{BAND} 채택 **{len(kept)}**", flush=True)
    (out / f"{alias}_v2_screen.json").write_text(json.dumps([{kk: vv for kk, vv in s.items()} for s in store]), encoding="utf-8")
    if save_text:
        _dump_texts(out / f"{alias}_v2_screen_texts.json",
                    [{"pidx": p["pidx"], "kind": "orig", "gold": p["gold"],
                      "samples": [{"text": t, "fin": bool(f)} for t, f in grp]}
                     for p, grp in zip(probs, groups)])

    # 2. 변형 × k
    jobs, prompts = [], []
    for s in kept:
        for kind, text in official_variants(s["problem"], seed=s["pidx"]):        # noqa: F821
            jobs.append((s, kind)); prompts.append(chat(g.tok, text))
    if prompts:
        t1 = time.time()
        vgroups, vtok = g.samples(prompts, k, 0.6)
        print(f"[변형] {len(prompts)}건 × k={k} · {time.time() - t1:.0f}s · {vtok / max(1, time.time() - t1):.1f} tok/s", flush=True)
        if save_text:
            _dump_texts(out / f"{alias}_v2_var_texts.json",
                        [{"pidx": sj["pidx"], "kind": kind, "gold": sj["gold"],
                          "samples": [{"text": t, "fin": bool(f)} for t, f in grp]}
                         for (sj, kind), grp in zip(jobs, vgroups)])
        for (s, kind), grp in zip(jobs, vgroups):
            a, nfin, nhit = acc_of(grp, s["gold"], k)
            s["var_acc"][kind] = a
            s["var_detail"][kind] = {"n_fin": nfin, "n_hit": nhit, "n_unfin": k - nfin, "n_wrong": nfin - nhit}   # 지휘 #37-2 원인 분해
        for s in kept:
            va = [v for v in s["var_acc"].values() if v is not None]
            if not va or s["base_acc"] is None:
                continue
            d = s["base_acc"] - min(va)
            s["max_drop"] = d
            s["label"] = True if d <= DROP_ROBUST else (False if d >= DROP_SPUR else None)
            # **대조 라벨**: 완주한 샘플만 분모로 쓴 정확도(미완주 제외). 파일럿2 에서 spurious 의
            # 91% 가 미완주로 설명돼, 주 라벨이 길이 아티팩트인지 확인할 대조군이 필요하다(지휘 #72).
            bf = (s["n_hit"] / s["n_fin"]) if s.get("n_fin") else None
            vf = [(dd["n_hit"] / dd["n_fin"]) for dd in s["var_detail"].values() if dd.get("n_fin")]
            if bf is not None and vf:
                df = bf - min(vf)
                s["base_acc_fin"], s["max_drop_fin"] = bf, df
                s["label_fin"] = True if df <= DROP_ROBUST else (False if df >= DROP_SPUR else None)
            else:
                s["base_acc_fin"], s["max_drop_fin"], s["label_fin"] = bf, None, None
    # **과반 규칙 라벨을 같이 저장한다 (지휘 #RULE-A, 민감도 분석용).**
    # 주라벨 max_drop 규칙은 유형 중 **최솟값**을 보므로 k=4 에서 1샘플 실패가 곧 spurious 가 된다
    # (Skywork 88 실측: spurious 26건 중 14건이 그 경우). 논문 정의는 '섭동의 **과반**' 이므로
    # 그 규칙도 같이 계산해 둔다 — var_acc 가 유형별로 있어 공짜다. 주라벨은 바꾸지 않는다.
    for s in kept:
        va = {kk: vv for kk, vv in (s.get("var_acc") or {}).items() if vv is not None}
        if va and s.get("base_acc") is not None:
            nb = sum(1 for vv in va.values() if (s["base_acc"] - vv) >= DROP_SPUR)
            s["n_broken_types"], s["n_types"] = nb, len(va)
            s["label_major"] = bool(nb <= len(va) / 2)
        else:
            s["n_broken_types"], s["n_types"], s["label_major"] = None, None, None

    # **라벨마다 생성 예산을 기록한다 (지휘 #LG-32k-3).** cap 을 올리면 미완주율이 바뀌고 미완주는
    # 오답으로 세므로 **라벨 정의가 cap 에 달려 있다**. cap 16384 라벨과 32768 라벨을 섞어 학습하면
    # 서로 다른 기준으로 만든 라벨을 한 분포로 취급하게 된다 — 그걸 사후에 가릴 수 있게 남긴다.
    for s in kept:
        s["gen_cap"] = int(max_new)
        s["gen_k"] = int(k)
    lab = [s for s in kept if s["label"] is not None]
    print(f"[라벨] 채택 {len(kept)} · 라벨 {len(lab)} (robust {sum(bool(s['label']) for s in lab)} / "
          f"spurious {sum(s['label'] is False for s in lab)}) · 중간대역 제외 {sum(1 for s in kept if s['max_drop'] is not None and s['label'] is None)}"
          f" · gen_cap {max_new}", flush=True)
    labf = [s for s in kept if s.get("label_fin") is not None]
    agree = sum(1 for s in kept if s["label"] is not None and s.get("label_fin") is not None and s["label"] == s["label_fin"])
    both = sum(1 for s in kept if s["label"] is not None and s.get("label_fin") is not None)
    nun = sum(d.get("n_unfin", 0) for s in kept if s["label"] is False for d in s["var_detail"].values())
    nwr = sum(d.get("n_wrong", 0) for s in kept if s["label"] is False for d in s["var_detail"].values())
    print(f"[대조라벨] label_fin {len(labf)}건 (robust {sum(1 for s in labf if s['label_fin'])} / "
          f"spurious {sum(1 for s in labf if s['label_fin'] is False)}) · 주라벨과 일치 {agree}/{both}", flush=True)
    print(f"[원인] spurious 쪽 변형 실패: 미완주 {nun} · 오답 {nwr} "
          f"(미완주 비중 {nun / max(1, nun + nwr):.0%}) — 높으면 길이 아티팩트 의심", flush=True)

    # 3. 궤적 (greedy 1회)
    if do_traj and kept:
        gp, _ = g.samples([chat(g.tok, s["problem"]) for s in kept], 1, 0.0)
        for s, grp in zip(kept, gp):
            s["greedy_text"] = grp[0][0]; s["greedy_fin"] = grp[0][1]
    g.close()
    (out / f"{alias}_v2_labels.json").write_text(json.dumps(kept), encoding="utf-8")
    if do_traj and kept:
        traj(model_id, kept, out / f"{alias}_v2.npz")
    print(f"[done] {alias} · {time.time() - t0:.0f}s", flush=True)


def _variants_and_labels(g, model_id, out_dir, kept, k, do_traj):
    """선별을 재사용할 때의 변형·라벨·궤적 경로 (개정 #37)."""
    alias = model_id.split("/")[-1].replace(".", "_")
    out = Path(out_dir)
    jobs, prompts = [], []
    for s in kept:
        for kind, text in official_variants(s["problem"], seed=s["pidx"]):        # noqa: F821
            jobs.append((s, kind)); prompts.append(chat(g.tok, text))
    print(f"[변형] {len(prompts)}건 × k={k}", flush=True)
    t1 = time.time()
    vgroups, vtok = g.samples(prompts, k, 0.6)
    print(f"[변형] {time.time() - t1:.0f}s · {vtok / max(1, time.time() - t1):.1f} tok/s", flush=True)
    for (s, kind), grp in zip(jobs, vgroups):
        a, nfin, nhit = acc_of(grp, s["gold"], k)
        s["var_acc"][kind] = a
        s["var_detail"][kind] = {"n_fin": nfin, "n_hit": nhit, "n_unfin": k - nfin, "n_wrong": nfin - nhit}
    for s in kept:
        s["gen_cap"] = int(getattr(g, "max_new", 0))      # 지휘 #LG-32k-3: 라벨은 cap 에 달려 있다
        s["gen_k"] = int(k)
        va = [v for v in s["var_acc"].values() if v is not None]
        if not va:
            continue
        worst = min(s["var_acc"], key=lambda kk: s["var_acc"][kk])
        s["max_drop"] = s["base_acc"] - s["var_acc"][worst]
        s["worst_kind"] = worst
        s["worst_cause"] = s["var_detail"][worst]
        s["label"] = True if s["max_drop"] <= DROP_ROBUST else (False if s["max_drop"] >= DROP_SPUR else None)
        # **대조 라벨도 여기서 쓴다 (2026-10-11).** 이 경로(선별 재사용)는 `label_fin` 을 안 써서
        # Olmo off40 · DeepSeek off40 라벨에 그 필드가 아예 없었다 → #77 통과 기준 3(대조 라벨이
        # 0.5 미만으로 뒤집히지 않는지)을 판정할 수 없었다. 전체경로와 **같은 식**이다
        # (`analysis/backfill_derived_labels.py --dry-run` 이 기존 264행을 불일치 0으로 재현해 확인).
        bf = (s["n_hit"] / s["n_fin"]) if (s.get("n_fin") and s.get("n_hit") is not None) else None
        vf = [(dd["n_hit"] / dd["n_fin"]) for dd in s["var_detail"].values() if dd.get("n_fin")]
        if bf is not None and vf:
            df = bf - min(vf)
            s["base_acc_fin"], s["max_drop_fin"] = bf, df
            s["label_fin"] = True if df <= DROP_ROBUST else (False if df >= DROP_SPUR else None)
        else:
            s["base_acc_fin"], s["max_drop_fin"], s["label_fin"] = bf, None, None
    lab = [s for s in kept if s["label"] is not None]
    spur = [s for s in lab if s["label"] is False]
    print(f"[라벨] 채택 {len(kept)} · 라벨 {len(lab)} (robust {sum(1 for s in lab if s['label'])} / spurious {len(spur)}) · "
          f"중간 제외 {sum(1 for s in kept if s['max_drop'] is not None and s['label'] is None)}", flush=True)
    if spur:
        print(f"[spurious 원인] 미완주로 떨어진 건 {sum(1 for s in spur if s['worst_cause']['n_unfin'] > s['worst_cause']['n_wrong'])} / "
              f"오답으로 떨어진 건 {sum(1 for s in spur if s['worst_cause']['n_wrong'] >= s['worst_cause']['n_unfin'])}", flush=True)
    if do_traj and kept:
        gp, _ = g.samples([chat(g.tok, s["problem"]) for s in kept], 1, 0.0)
        for s, grp in zip(kept, gp):
            s["greedy_text"] = grp[0][0]; s["greedy_fin"] = grp[0][1]
    g.close()
    (out / f"{alias}_v3_labels.json").write_text(json.dumps(kept), encoding="utf-8")
    if do_traj and kept:
        traj(model_id, kept, out / f"{alias}_v3.npz")
    print("[done] 변형·라벨 경로 완료", flush=True)


def traj(model_id, kept, out_npz):
    """greedy 생성문을 teacher forcing 으로 한 번 통과 → L//2 32토큰 청크 평균."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True).eval()
    layers = max((m for m in model.modules() if isinstance(m, torch.nn.ModuleList)), key=len)
    mid = (len(layers) + 1) // 2
    grab = {}
    h = layers[mid - 1].register_forward_hook(lambda m, a, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    t0 = time.time()
    for i, s in enumerate(kept, 1):
        try:
            pid = tok(chat(tok, s["problem"]), return_tensors="pt")["input_ids"][0].tolist()
            gid = tok(s.get("greedy_text", ""), add_special_tokens=False)["input_ids"]
            ids = torch.tensor([pid + gid], device=model.device)
            with torch.no_grad():
                model(input_ids=ids, logits_to_keep=1)
            H = grab["h"][0, len(pid):].float()
            C = [H[j:j + CHUNK].mean(0) for j in range(0, len(H), CHUNK)]
            s["chunks"] = torch.stack(C).cpu().numpy().astype(np.float16) if C else np.zeros((0, 1), np.float16)
            s["prompt_tokens"] = len(pid); s["gen_tokens"] = len(gid)
        except Exception as e:
            print(f"  [궤적 실패] pidx={s['pidx']}: {type(e).__name__}", flush=True)
            s["chunks"] = np.zeros((0, 1), np.float16)
        if i % 20 == 0:
            print(f"  궤적 {i}/{len(kept)} · {time.time() - t0:.0f}s", flush=True)
    h.remove()
    for s in kept:
        s["label_official"] = s["label"]; s["label_unofficial"] = None
    np.savez_compressed(out_npz, store=np.array(kept, dtype=object), model=model_id, mid=mid)
    print(f"[궤적] 저장 {out_npz} · {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=globals().get("RUN_MODELS", ["Skywork/Skywork-OR1-Math-7B"]))
    ap.add_argument("--out", default="/kaggle/working/genlab")
    ap.add_argument("--n", type=int, default=globals().get("RUN_N", 90))
    ap.add_argument("--k", type=int, default=globals().get("RUN_K", 4))
    ap.add_argument("--max-new", type=int, default=globals().get("RUN_MAX_NEW", 6144))
    ap.add_argument("--source", default=globals().get("RUN_SOURCE", "aime"))
    ap.add_argument("--max-seqs", type=int, default=globals().get("RUN_MAX_SEQS", 48))
    ap.add_argument("--no-traj", action="store_true")
    ap.add_argument("--screen-glob", default=globals().get("RUN_SCREEN", None), help="이전 커널의 *_screen.json 글롭")
    ap.add_argument("--cap", type=int, default=globals().get("RUN_CAP", 0), help="채택 상한 (0=무제한)")
    ap.add_argument("--pool-offset", type=int, default=globals().get("RUN_OFFSET", 0),
                    help="풀 앞쪽 건너뛸 문제 수(슬라이스)")
    a = ap.parse_args()
    for m in a.models:
        run(m, a.out, a.n, a.k, a.max_new, a.source, a.max_seqs, not a.no_traj, a.screen_glob, a.cap,
            a.pool_offset)
