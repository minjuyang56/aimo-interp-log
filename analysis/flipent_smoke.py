"""#FLIPENT-ZIP 스모크 — 제출물이 **측정과 같은 양을 재는지** 수치로 검사한다. 비용 $0.

검사 항목 (하나라도 실패하면 제출물이 아니다)
  1. `ent_front` 가 `analysis/entlens_eval.ent_feats` 의 첫 성분과 **같은 값**인가
     (난수 배열 + **실제 P2 산출물 행 전수**. 베껴 적은 식이 어긋나는 사고를 막는 유일한 방법이다)
  2. 저장된 모델별 임계가 공식 33(P2) 중앙값이고 **CLAUDE.md 기록과 같은가**
  3. h15C 폴백이 **출하된 h15C 제출물과 같은 판정**인가 (같은 점수식·같은 임계)
  4. 임계 없는 `model_id`(gpt-oss-120b · 미지 · 빈 문자열)에서 **모델을 열지 않고** 폴백하는가
  5. 짧은·비유한 엔트로피 흔적은 그 행만 폴백으로 가는가
  6. 예산이 cap 1024 를 못 사면 생성을 시작하지 않고 폴백하는가 (cap 을 깎지 않는다)
  7. 반환이 전부 순수 `bool` 이고 길이가 입력과 같은가 · 빈 입력은 `[]`
  8. (`--e2e`) **4모델 x 2문제 로컬 e2e** — 실제 생성 경로로 ent_front 를 받아 판정까지 간다

사용:
  .venv/Scripts/python.exe analysis/flipent_smoke.py              # 1~7 (GPU 불필요)
  .venv/Scripts/python.exe analysis/flipent_smoke.py --e2e        # + 4모델 x 2문제 로컬 4bit
  .venv/Scripts/python.exe analysis/flipent_smoke.py --e2e --cap 1024
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "solutions/flipent"))
sys.path.insert(0, str(ROOT / "solutions/h15C"))

import flipent_lib as FL  # noqa: E402
from entlens_eval import ent_feats  # noqa: E402
from flipent_eval import load_p2  # noqa: E402

RES = ROOT / "experiments/results"
OUT = RES / "flipent_smoke.txt"
PROBLEMS = [
    "What is the value of $255^2 - 245^2$?",
    "Find the number of positive divisors of 2024.",
]
SMALL = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
RECORDED = {"deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 0.1906, "allenai/Olmo-3-7B-Think": 0.2414,
            "Skywork/Skywork-OR1-Math-7B": 0.1364}
L, FAILED = [], []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def check(name, ok, detail=""):
    out("   [%s] %-44s %s" % ("OK" if ok else "FAIL", name, detail))
    if not ok:
        FAILED.append(name)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true", help="4모델 x 2문제 로컬 생성까지 돈다")
    ap.add_argument("--cap", type=int, default=1024, help="e2e 생성 상한 (사전등록 값 1024)")
    ap.add_argument("--only", default="", help="e2e 를 이 model_id 하나만 (로컬 8 GB 단편화 회피)")
    ap.add_argument("--tag", default="", help="보고서 파일 접미사 (--only 와 같이 쓴다)")
    a = ap.parse_args()

    out("#FLIPENT-ZIP 스모크 — 제출물이 측정과 같은 양을 재는지 검사 · $0")
    out()

    # ---------------------------------------------------------------- 1. ent_front 동일성
    out("## 1. ent_front == entlens_eval.ent_feats[0] (제출물이 측정과 같은 식인가)")
    rng = np.random.default_rng(0)
    worst = 0.0
    for T in [8, 9, 10, 33, 100, 255, 256, 257, 1024]:
        v = rng.random(T) * 2
        mine, ref = FL.ent_front(v.tolist()), ent_feats(v)[0]
        worst = max(worst, abs(mine - ref))
    check("난수 배열 9종(T=8..1024) 최대절대차", worst < 1e-12, "%.3e" % worst)
    #  실제 P2 산출물 전수 — 측정에 쓴 바로 그 값들
    nrow, wr = 0, 0.0
    for al in ["Qwen3_5-4B", "Skywork-OR1-Math-7B", "Olmo-3-7B-Think",
               "DeepSeek-R1-0528-Qwen3-8B"]:
        rows = load_p2(RES / "entlens", al) or []
        for r in rows:
            ref = ent_feats(r["ent"])
            mine = FL.ent_front(r["ent"])
            if ref is None:
                continue
            wr = max(wr, abs(mine - ref[0]))
            nrow += 1
    check("실제 P2 행 %d개 최대절대차" % nrow, nrow >= 100 and wr < 1e-12, "%.3e" % wr)
    #  T<8 · 비유한 → None (그 행만 폴백)
    check("T<8 은 None", FL.ent_front([0.1] * 7) is None and FL.ent_front([]) is None)
    check("비유한(nan/inf) 흔적은 None",
          FL.ent_front([float("nan")] * 20) is None and FL.ent_front([float("inf")] * 20) is None)
    out()

    # ------------------------------------------------- 1b. 프롬프트·토큰 id 동일성 (측정 커널 대조)
    out("## 1b. 프롬프트·토큰 id == P2 커널 (임계가 그 측정에 묶여 있다)")
    sys.path.insert(0, str(ROOT / "experiments/kaggle"))
    import entlens as EK  # noqa: E402  — 임계를 만든 그 커널
    check("SOLVE_PROMPT 문자열 동일", FL.SOLVE_PROMPT == EK.SOLVE_PROMPT)
    from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402
    for mid in SMALL:
        try:
            try:
                tk = PreTrainedTokenizerFast.from_pretrained(mid, local_files_only=True)
            except Exception:
                tk = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        except Exception as e:
            out("        %-34s 토크나이저 없음 (%s) — 건너뜀" % (mid, type(e).__name__))
            continue
        same_str = all(FL.chat_prompt(tk, q) == EK.chat(tk, q) for q in PROBLEMS)
        ids_f = [tk(FL.chat_prompt(tk, q), add_special_tokens=False)["input_ids"] for q in PROBLEMS]
        ids_k = [tk(EK.chat(tk, q), add_special_tokens=False)["input_ids"] for q in PROBLEMS]
        ids_t = [tk(FL.chat_prompt(tk, q), add_special_tokens=True)["input_ids"] for q in PROBLEMS]
        dbl = [len(b) - len(a) for a, b in zip(ids_f, ids_t)]
        check("%-34s 프롬프트·id 동일" % mid.split("/")[-1], same_str and ids_f == ids_k,
              "토큰 %s · add_special_tokens=True 면 +%s 토큰" % ([len(x) for x in ids_f], dbl))
    out("        (마지막 칸이 0 이 아닌 모델에서는 기본 토크나이즈가 **BOS 를 두 번** 넣는다 —")
    out("         제출물은 커널과 같게 `add_special_tokens=False` 를 쓴다)")
    out()

    # ---------------------------------------------------------------- 2. 임계
    out("## 2. 저장된 임계 == 공식 33(P2) 중앙값 == CLAUDE.md 기록")
    art = FL.load_art()
    per = art.get("per_model", {})
    check("모델 4종 임계 존재", sorted(per) == sorted(SMALL), " ".join(sorted(per.keys())))
    for mid, rec in RECORDED.items():
        t = FL.thr_for(art, mid)
        check("기록값 재현 %s" % mid.split("/")[-1], t is not None and abs(round(t, 4) - rec) < 5e-5,
              "%.6f vs 기록 %.4f" % (t if t else float("nan"), rec))
    qt = FL.thr_for(art, "Qwen/Qwen3.5-4B")
    out("        Qwen3.5-4B 임계 %.6f (CLAUDE.md 기록 없음 — 이번에 처음 계산, 게이트 없음)" % qt)
    #  중앙값을 다시 계산해 저장값과 맞댄다 (가중치 파일이 손으로 수정되지 않았나)
    bad = []
    for al, mid in [("Qwen3_5-4B", "Qwen/Qwen3.5-4B"),
                    ("Skywork-OR1-Math-7B", "Skywork/Skywork-OR1-Math-7B"),
                    ("Olmo-3-7B-Think", "allenai/Olmo-3-7B-Think"),
                    ("DeepSeek-R1-0528-Qwen3-8B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")]:
        rows = load_p2(RES / "entlens", al) or []
        v = sorted(ent_feats(r["ent"])[0] for r in rows if ent_feats(r["ent"]) is not None)
        if abs(float(np.median(v)) - FL.thr_for(art, mid)) > 1e-12:
            bad.append(al)
    check("저장값 == 다시 계산한 중앙값 (4모델)", not bad, " ".join(bad) or "불일치 0")
    check("cap·부호 메타 기록", art.get("cap") == 1024 and art.get("experimental") is True,
          "cap %s · experimental %s" % (art.get("cap"), art.get("experimental")))
    out()

    # ---------------------------------------------------------------- 3. h15C 동일성
    out("## 3. 폴백 == 출하된 h15C 제출물의 판정")
    import h15_lib  # noqa: E402  — 출하된 h15C 그대로
    texts = PROBLEMS + ["x" * 3, " ".join(["word"] * 81), " ".join(["w"] * 27),
                        " ".join(["q"] * 59), " ".join(["z"] * 60), "one"]
    mine = FL.h15c_decide(texts)
    ref = h15_lib.decide(texts, "C")
    check("판정 %d/%d 일치" % (sum(int(x == y) for x, y in zip(mine, ref)), len(texts)), mine == ref,
          "단어수 " + " ".join(str(len(t.split())) for t in texts))
    check("폴백 임계 == h15C 출하값",
          abs(art.get("h15c_thr", 0) - h15_lib.load_art()["thr"]["C"]) < 1e-12)
    out()

    # ---------------------------------------------------------------- 4. 임계 없는 model_id
    out("## 4. 임계 없는 model_id — 모델을 열지 않고 폴백 (규칙 4: KeyError 금지)")
    for mid in ["openai/gpt-oss-120b", "qwen3-8b:low", "totally-unknown-model", ""]:
        t0 = time.time()
        got = FL.decide(PROBLEMS, mid)
        el = time.time() - t0
        ok = (got == FL.h15c_decide(PROBLEMS) and all(type(x) is bool for x in got) and el < 2.0)
        check("%-24s → h15C 폴백" % (mid or "(빈 문자열)"), ok, "%.2fs · %s" % (el, got))
    check("빈 입력은 []", FL.decide([], "Skywork/Skywork-OR1-Math-7B") == [])
    out()

    # ---------------------------------------------------------------- 5. 예산 가드
    out("## 5. 예산이 cap 1024 를 못 사면 생성을 시작하지 않는다 (cap 을 깎지 않는다)")
    #  **스텁 모델**을 끼워 로드 없이 가드만 잰다. `decide` 는 `token_slots` 가 그 모델의 KV 를
    #  읽어야 해서 예산 검사보다 로드가 먼저다(워커에서는 warm cache 2초) — 로컬 8 GB 에서
    #  7~8B 를 fp16 으로 올리면 RAM 으로 흘러 분 단위가 되므로 스텁으로 가른다.
    class _Cfg:
        num_hidden_layers, num_key_value_heads, num_attention_heads, hidden_size = 28, 4, 28, 3584
        head_dim = 128

    class _Stub:
        config = _Cfg()

    old, keep = FL.SOFT, dict(FL._m)
    try:
        FL._m.update(model=_Stub(), tok=None, id="Skywork/Skywork-OR1-Math-7B")
        FL.SOFT = time.time() - FL.T0 + 0.001      # 남은 예산 ~0
        got = FL.decide(PROBLEMS, "Skywork/Skywork-OR1-Math-7B")
        check("예산 0 → 생성 전에 폴백", got == FL.h15c_decide(PROBLEMS)
              and all(type(x) is bool for x in got), str(got))
        FL.SOFT = old                               # 예산을 돌려주면 같은 스텁에서 계획이 선다
        slots = FL.token_slots(_Stub(), FL.CAP)
        est = FL.seconds_for(40, FL.CAP, slots)
        check("예산 충분 → 계획 성립(칸 %d · 추정 %.0fs)" % (slots, est), est <= FL.SOFT * 0.8)
    finally:
        FL.SOFT = old
        FL._m.update(keep)
    check("CAP 사다리가 없다(점수 정의가 cap 에 묶여 있다)", FL.CAP == 1024, "CAP=%d" % FL.CAP)
    out()

    # ---------------------------------------------------------------- 6. 시간모델 자기일관
    out("## 6. 시간 견적 (flipent_lib 내부 모형 · 워커 칸 가정)")
    for n in [5, 14, 35, 40]:
        s = FL.seconds_for(n, FL.CAP, slots=max(1, min(n, 87)))     # Olmo 칸 87(병목)
        out("        문제 %2d · Olmo 칸 87 가정 → %5.0fs (3300s 의 %.1f%%)" % (n, s, 100 * s / 3300))
    out()

    # ---------------------------------------------------------------- 7. e2e
    if a.e2e:
        out("## 7. 로컬 e2e — 4모델 x 2문제 (로컬 8 GB 라 **4bit**, 워커는 fp16)")
        os.environ["FLIPENT_QUANT"] = "4bit"
        FL.CAP = int(a.cap)
        out("        cap %d · 문제 2개 · 4bit nf4 (임계는 fp16 측정에서 나왔으므로 **판정 자체는" % FL.CAP)
        out("        교정 비교가 아니다** — 경로·형식·반환 계약을 재는 것이다)")
        gen_ok = 0
        for mid in ([a.only] if a.only else SMALL):
            t0 = time.time()
            try:
                got = FL.decide(PROBLEMS, mid)
                el = time.time() - t0
                loaded = (FL._m["id"] == mid and FL._m["model"] is not None)
                gen_ok += int(loaded)
                #  **경로를 반드시 구분해 적는다**: 반환 계약만 보면 폴백도 통과로 보인다.
                #  로컬 8 GB 에서 모델을 연달아 올리면 `device_map="auto"` 가 CPU/디스크로 흘려
                #  ValueError 를 내고 그 모델은 h15C 로 떨어진다(#ORG-LABEL 에서 본 그 증상).
                check("%-34s %s" % (mid, got),
                      len(got) == 2 and all(type(x) is bool for x in got),
                      "%.0fs · 경로 %s" % (el, "생성" if loaded else "**폴백(로드 실패)**"))
            except Exception as e:
                check("%-34s 예외" % mid, False, "%s %s" % (type(e).__name__, str(e)[:100]))
        out("        생성 경로를 실제로 돈 모델 %d/%d" % (gen_ok, 1 if a.only else len(SMALL)))
        out()

    out("## 판정: %s" % ("전부 통과" if not FAILED else "실패 %d개 — %s" % (len(FAILED), FAILED)))
    #  `--tag` 가 있으면 보고서를 따로 쓴다 — 개별 모델 실행이 전체 보고서를 덮지 않게 한다
    #  (이 결함으로 개별 e2e 실행이 전체 보고서를 한 번 덮었다. .log 에 남아 복원했고 여기서 고쳤다.)
    dst = OUT.with_name("flipent_smoke%s.txt" % (("_" + a.tag) if a.tag else ""))
    dst.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(dst))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
