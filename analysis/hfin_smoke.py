"""#FINAL hfin 제출물 스모크 — 계측만 한다, 주장하지 않는다.

검사 목록
  1. `h15c_decide` 가 **출하된 h15C 제출물과 같은 판정**을 내는가 (폴백 동일성)
  2. `score_fin` 단위검사 (k=1·k=2·빈 입력)
  3. **라우팅**: 대상 아닌 model_id 는 **모델을 열지 않고** 폴백 · 순수 bool
  4. **예산 가드**: 추정이 예산의 80% 를 넘으면 **생성 전에** 폴백
  5. **공식 33 재현**: 제출물의 `score_fin`·`h15c_decide` 로 하이브리드 정확도를 다시 계산해
     `hfin_hybrid_report.txt`(0.6818)와 맞댄다 — 평가기와 제출물이 같은 양을 쓰는지의 게이트
  6. 반환 계약: 길이 일치 · `type(x) is bool` · 빈 리스트 · 문제 1개
  7. (`--e2e <model_id>`) 로컬 GPU end-to-end. **cap 을 낮춰서 생성 경로만** 본다 —
     로컬 4bit 는 9~10 tok/s 라 cap 12288 이 시퀀스당 20분이다. 낮춘 cap 은 판정용 숫자가 아니다.

사용: .venv/Scripts/python.exe analysis/hfin_smoke.py [--e2e Skywork/Skywork-OR1-Math-7B]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solutions" / "hfin"))
sys.path.insert(0, str(ROOT / "solutions" / "h15C"))

RES = ROOT / "experiments/results"
OUT = RES / "hfin_smoke.txt"
L = []
FAILED = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def check(name, ok, detail=""):
    out("  %-52s %s %s" % (name, "OK" if ok else "**FAIL**", detail))
    if not ok:
        FAILED.append(name)


def load_h20m(alias, fn):
    p = RES / ("h20m/aimo-h20m-%s/h14/%s_h14.json" % (alias, fn))
    rows = json.loads(p.read_text(encoding="utf-8"))
    recs = []
    for r in rows:
        o = r["by_kind"].get("orig")
        if o:
            recs.append(dict(y=int(r["y"]), problem=r["problem"],
                             tokens=[int(t) for t in o["tokens"]]))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="")
    ap.add_argument("--e2e-cap", type=int, default=256)
    ap.add_argument("--e2e-n", type=int, default=2)
    a = ap.parse_args()

    import hfin_lib

    out("#FINAL hfin 스모크")
    out("  라이브러리 상수: CAP %d · k %d · THR_FIN %.2f · SOFT %.0f"
        % (hfin_lib.CAP, hfin_lib.KSAMP, hfin_lib.THR_FIN, hfin_lib.SOFT))
    out("  H-FIN 라우팅 모델: %s" % sorted(hfin_lib.HFIN_MODELS))
    out()

    # ---- 1. 폴백이 출하된 h15C 와 같은 판정인가
    out("[1] h15C 폴백 동일성")
    import h15_lib
    probs = [
        "What is 2+2?",
        "Let $x$ be a real number such that $x^2 = 4$. Find $x$.",
        " ".join(["word"] * 81),
        " ".join(["w"] * 27),
        " ".join(["w"] * 59),
        " ".join(["w"] * 60),
        "one",
    ]
    mine = hfin_lib.h15c_decide(probs)
    ship = h15_lib.decide(probs, "C")
    agree = sum(1 for x, y in zip(mine, ship) if bool(x) == bool(y))
    check("h15c_decide == 출하 h15C decide(mode C)", agree == len(probs),
          "%d/%d 일치 · 임계 %.12f" % (agree, len(probs), hfin_lib.H15C_THR))
    check("폴백 반환이 순수 bool", all(type(x) is bool for x in mine))
    out()

    # ---- 2. score_fin 단위검사
    out("[2] score_fin 단위검사")
    check("k=1 완주 -> 1.0", hfin_lib.score_fin([True]) == 1.0)
    check("k=1 미완주 -> 0.0", hfin_lib.score_fin([False]) == 0.0)
    check("k=2 반반 -> 0.5 (임계와 같아 robust)", hfin_lib.score_fin([True, False]) == 0.5)
    check("빈 입력 -> None (그 행만 폴백)", hfin_lib.score_fin([]) is None)
    out()

    # ---- 3. 라우팅: 대상 아닌 model_id 는 모델을 열지 않는다
    out("[3] 라우팅 (모델 로드 없음)")
    import time as _t
    sys.path.insert(0, str(ROOT / "solutions" / "hfin"))
    import solution
    cases = [(m, "default") for m in ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                                      "Qwen/Qwen3.5-4B", "some/unknown-model", "")]
    #  gpt-oss 는 Main 전용이고 로컬에 없다 -> **모든 effort 에서** 모델을 열지 않아야 한다
    cases += [("openai/gpt-oss-120b", e) for e in ("default", "low", "medium", "high")]
    for mid, eff in cases:
        t = _t.time()
        r = solution.are_robust(mid, eff, probs)
        dt = _t.time() - t
        ok = (len(r) == len(probs) and all(type(x) is bool for x in r)
              and r == hfin_lib.h15c_decide(probs) and dt < 5.0)
        check("%-38s eff=%-7s -> h15C · %.2fs" % (mid or "(빈 문자열)", eff, dt), ok,
              "True %d/%d" % (sum(r), len(r)))
    out()

    # ---- 4. 예산 가드
    out("[4] 예산 가드 (생성 전에 폴백하는가)")

    class _StubTok:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"]

        def __call__(self, p, **kw):
            if isinstance(p, str):
                return {"input_ids": [0] * 600}
            return {"input_ids": [[0] * 600 for _ in p]}

    class _StubCfg:
        num_hidden_layers, num_key_value_heads, num_attention_heads, hidden_size = 28, 4, 28, 3584
        head_dim = 128

    class _StubModel:
        config = _StubCfg()
        generation_config = None

    saved_get, saved_gen = hfin_lib.get_model, hfin_lib._gen_safe
    gen_called = {"n": 0}

    def _fake_gen(*args, **kw):
        gen_called["n"] += 1
        return None

    hfin_lib.get_model = lambda mid: (_StubTok(), _StubModel())
    hfin_lib._gen_safe = _fake_gen
    saved_soft = hfin_lib.SOFT
    try:
        hfin_lib.SOFT = 1.0         # 예산 1초 -> 어떤 cap 도 못 산다
        r = hfin_lib.decide(probs, "allenai/Olmo-3-7B-Think", "default")
        check("예산 1s -> 생성 호출 0회 · h15C 와 동일", gen_called["n"] == 0
              and r == hfin_lib.h15c_decide(probs))
        slots = hfin_lib.token_slots(_StubModel(), hfin_lib.CAP, 600.0)
        for nq in (3, 10, 35, 40):
            est = hfin_lib.seconds_for(nq, hfin_lib.CAP, slots)
            out("      문제 %2d개 · 배치칸 %d · 추정 %7.0fs  (호출예산 825s 의 %4.0f%%)"
                % (nq, slots, est, 100 * est / 825.0))
    finally:
        hfin_lib.SOFT = saved_soft
        hfin_lib.get_model, hfin_lib._gen_safe = saved_get, saved_gen
    out()

    # ---- 5. 공식 33 하이브리드 재현 (제출물 함수로)
    out("[5] 공식 33 하이브리드 재현 — 제출물 함수로 다시 계산")
    srcs = [("Skywork", "skywork-or1-math-7b", "Skywork-OR1-Math-7B", True),
            ("Olmo", "olmo-3-7b-think", "Olmo-3-7B-Think", True),
            ("DeepSeek", "deepseek-r1-0528-qwen3-8b", "DeepSeek-R1-0528-Qwen3-8B", False)]
    base = load_h20m(srcs[0][1], srcs[0][2])
    y = np.array([r["y"] for r in base], int) == 1
    texts = [r["problem"] for r in base]
    accs, rows = [], []
    for nm, alias, fn, use_hfin in srcs:
        recs = load_h20m(alias, fn)
        if use_hfin:
            pred = np.array([bool(hfin_lib.score_fin(
                [t < 12288 for t in r["tokens"]]) >= hfin_lib.THR_FIN) for r in recs], bool)
            rule = "H-FIN"
        else:
            pred = np.array(hfin_lib.h15c_decide(texts), bool)
            rule = "h15C"
        acc = float((pred == y).mean())
        rows.append((nm, rule, acc))
        if use_hfin:
            accs.append(acc)
    hl = float(np.mean(np.array(hfin_lib.h15c_decide(texts), bool) == y))
    hyb = float(np.mean(accs + [hl, hl]))       # DeepSeek·Qwen3.5 두 칸이 같은 h15C 예측
    for nm, rule, acc in rows:
        out("      %-9s %-6s 정확도 %.4f" % (nm, rule, acc))
    out("      Qwen3.5   h15C   정확도 %.4f  (텍스트 규칙이라 DeepSeek 칸과 같다)" % hl)
    rep = RES / "hfin_hybrid_report.txt"
    want = None
    if rep.is_file():
        for ln in rep.read_text(encoding="utf-8").splitlines():
            if "하이브리드 모델평균 정확도" in ln:
                want = float(ln.split("=")[-1].strip().strip("*"))
    #  보고서는 소수 4자리로 찍으므로 그 자리에서 비교한다(1e-9 은 인쇄 정밀도보다 좁다)
    check("하이브리드 모델평균 %.4f == 평가기 %s" % (hyb, want),
          want is not None and abs(hyb - want) < 5e-5)
    out()

    # ---- 5b. 워커 VRAM 기준 시간표 (로컬 GPU 로 재면 칸이 작아 숫자가 틀린다)
    out("[5b] 워커(96 GB) 기준 시간표 — [4] 는 로컬 8 GB 라 배치칸이 작게 나온다")

    class _Cfg2:
        def __init__(self, nl, nkv, nh, hs, hd):
            self.num_hidden_layers, self.num_key_value_heads = nl, nkv
            self.num_attention_heads, self.hidden_size, self.head_dim = nh, hs, hd

    class _M2:
        def __init__(self, cfg):
            self.config = cfg

    #  config 는 CLAUDE.md 환경 제약 표의 값. KV/토큰이 Skywork 56 KB · Olmo 512 KB 로 나와야 맞다.
    cfgs = [("Skywork", _Cfg2(28, 4, 28, 3584, 128)),
            ("Olmo", _Cfg2(32, 32, 32, 4096, 128))]
    FREE_KB = 90.0 * 1024 * 1024        # 96 GB 중 가중치를 뺀 여유를 보수적으로 90 GB
    for nm, cfg in cfgs:
        kvkb = hfin_lib.kv_kb_per_token(_M2(cfg))
        slots_w = max(1, int(FREE_KB * hfin_lib.KV_SAFE / ((600.0 + hfin_lib.CAP) * kvkb)))
        slots_w = max(1, min(slots_w, int(hfin_lib.MAX_PREFILL_TOK / 600.0)))
        line = "      %-9s KV %6.1f KB/토큰 · 배치칸 %2d" % (nm, kvkb, slots_w)
        for nq in (3, 5, 10, 21, 35, 40):
            line += " | %d문제 %.0fs" % (nq, hfin_lib.seconds_for(nq, hfin_lib.CAP, slots_w))
        out(line)
    out("      호출예산은 3300s 를 호출 수로 나눈 값이다. 생성하는 모델이 2개뿐이므로")
    out("      그 둘이 약 1650s 씩 쓸 수 있다 → 리더보드형(3문제)은 들어가고 35문제는 못 들어간다.")
    out()
    out("[5c] **Main 트랙 per-case 예산** (N=21 인스턴스 · 5모델 · 호출 5회)")
    out("      Main 은 21 인스턴스를 5모델로 나눈다 → 모델당 약 4~5문제."
        " gpt-oss·DeepSeek·Qwen3.5 는 h15C 라 생성 0s,")
    out("      따라서 Skywork·Olmo 가 3300s 를 거의 다 쓸 수 있다(각 약 1650s).")
    out("      %-9s %-7s %s" % ("모델", "cap", "문제 수별 추정 (per-case)"))
    for nm, cfg in cfgs:
        kvkb = hfin_lib.kv_kb_per_token(_M2(cfg))
        for cand in hfin_lib.CAP_LADDER:
            sw = max(1, int(FREE_KB * hfin_lib.KV_SAFE / ((600.0 + cand) * kvkb)))
            sw = max(1, min(sw, int(hfin_lib.MAX_PREFILL_TOK / 600.0)))
            line = "      %-9s %-7d" % (nm, cand)
            for nq in (4, 5, 21):
                e = hfin_lib.seconds_for(nq, cand, sw)
                line += " | %2d문제 %6.0fs (%5.1fs/문제)" % (nq, e, e / nq)
            out(line)
    out("      판정: Main 에서 모델당 4~5문제면 cap 12288 이 호출당 약 434~582s 로 "
        "1650s 예산의 26~35%% → **첫 칸으로 돈다**.")
    out()

    # ---- 6. 반환 계약
    out("[6] 반환 계약")
    check("빈 리스트 -> []", solution.are_robust("allenai/Olmo-3-7B-Think", "default", []) == [])
    one = solution.are_robust("some/unknown", "default", ["x " * 5])
    check("문제 1개 -> 길이 1 · 순수 bool", len(one) == 1 and type(one[0]) is bool)
    legacy = solution.are_robust("some/unknown", probs)       # 옛 2-인자 경로
    check("옛 2-인자 호출 == 3-인자 호출", legacy == hfin_lib.h15c_decide(probs))
    out()

    # ---- 7. 로컬 GPU e2e (선택)
    if a.e2e:
        out("[7] 로컬 GPU e2e — **cap 을 %d 로 낮춘 생성 경로 확인**(판정 숫자가 아니다)" % a.e2e_cap)
        os.environ["HFIN_QUANT"] = os.environ.get("HFIN_QUANT", "4bit")
        hfin_lib.CAP = a.e2e_cap
        ex = ["What is $255^2 - 245^2$?",
              "How many positive divisors does 2024 have?"][:a.e2e_n]
        t = _t.time()
        r = solution.are_robust(a.e2e, "default", ex)
        out("      반환 %s · %.0fs · 순수 bool %s · 길이 %d"
            % (r, _t.time() - t, all(type(x) is bool for x in r), len(r)))
        check("e2e 반환 계약", len(r) == len(ex) and all(type(x) is bool for x in r))
        out("      주의: cap %d 에서는 올림피아드 풀이가 끝나지 않으므로 '자연종료' 가 전부 False 에"
            % a.e2e_cap)
        out("      가깝다. 이건 설계대로이고(낮은 cap = 침묵) 판정용 숫자로 쓰지 않는다.")
        out()

    out("결과: 검사 %d개 중 실패 %d개%s"
        % (len([x for x in L if " OK " in x or "**FAIL**" in x]), len(FAILED),
           (" -> " + ", ".join(FAILED)) if FAILED else ""))
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
