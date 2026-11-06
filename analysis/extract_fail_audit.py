"""`extract_answer` 추출 실패 감사 — 정답인데 못 뽑는 샘플이 얼마나 되나. (규칙 ② 가 잡은 결함)

## 발견 경로 (2026-10-12, #TEMP-SENS 선별 직후)
상설 규칙 ②(라벨 생성 전 원문 5개 육안)대로 Olmo T=1.0 선별 원문 5개를 봤더니 **pidx 44** 가
풀이 끝이 `\\[\\n\\boxed{8}\\n\\]` 이고 gold 가 8 인데 **추출답이 None** 이었다 — 완주·정답인데
오답으로 집계된다. 나머지 4개는 `\\(\\boxed{67}\\)` 또는 `ANSWER: 336` 형태라 정상 추출됐다.
즉 **서식 선택이 샘플마다 달라지고 일부 서식에서 추출이 실패한다** — Ġ 사고와 같은 계열
(정답이 오답으로 위장)이고, 그때와 달리 이번엔 **원문을 저장했기 때문에** 측정할 수 있다.

## 이 감사가 답하는 것
1. 추출 실패율: `extract_answer` 가 None 을 주는 샘플 비율
2. **그중 '사실은 정답' 인 비율**: 본문의 마지막 `\\boxed{...}` 가 gold 와 같은 샘플
   → 이것이 base_acc 를 부당하게 깎는 양이다
3. 서식별 분포: 어떤 서식에서 실패하는가

## 왜 수치가 중요한가 (#TEMP-SENS 주지표와의 관계)
주지표 ②는 **base_acc 를 T=0.6 ↔ T=1.0 로 짝지어 비교**한다. 두 조건이 **같은 추출기**를 쓰므로
비교 자체는 공정하지만, **서식 선택 분포가 T 에 따라 달라지면** 추출 실패율도 달라지고 그것이
'디코딩 효과' 로 위장할 수 있다. 그 교란 크기를 재는 것이 이 스크립트의 목적이다.
한계: Olmo off40 **T=0.6 실행은 원문을 저장하지 않았다**(`save_text` 도입 전) → 그쪽 실패율은
직접 못 잰다. 대신 같은 모델·같은 프롬프트의 **greedy(T=0) 풀이**(genprobe 산출)로 참고값을 낸다.

사용:  .venv/Scripts/python.exe analysis/extract_fail_audit.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def load_extract():
    ns = {"__name__": "sc"}
    exec(compile((ROOT / "experiments/kaggle/solve_common.py").read_text(encoding="utf-8"),
                 "solve_common.py", "exec"), ns)
    return ns["extract_answer"]


def same(a, b):
    if a is None:
        return False
    try:
        return int(str(a).replace(",", "").strip()) == int(str(b).replace(",", "").strip())
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def last_boxed(t):
    m = BOXED.findall(t)
    return m[-1].strip() if m else None


def fmt_of(t):
    """꼬리 400자에서 답을 어떤 서식으로 냈는지 거칠게 분류한다."""
    tail = t[-400:]
    if re.search(r"ANSWER:\s*[-+]?\d", tail):
        return "ANSWER: 줄"
    if re.search(r"\\\(\s*\\boxed", tail):
        return r"\( \boxed \)"
    if re.search(r"\\\[\s*\\boxed", tail):
        return r"\[ \boxed \]"
    if "\\boxed" in tail:
        return "boxed (기타)"
    return "boxed 없음"


def audit(name, rows, ex, out):
    """rows: [{pidx, gold, samples:[{text, fin}]}]"""
    tot = fail = fail_but_right = fin_cnt = 0
    byfmt = {}
    fail_ex = []
    for r in rows:
        for s in r["samples"]:
            t, g = s["text"], r["gold"]
            tot += 1
            fin_cnt += int(bool(s.get("fin")))
            f = fmt_of(t)
            got = ex(t)
            bad = got is None or not same(got, g)
            d = byfmt.setdefault(f, {"n": 0, "none": 0, "none_right": 0})
            d["n"] += 1
            if got is None:
                fail += 1
                d["none"] += 1
                if same(last_boxed(t), g):
                    fail_but_right += 1
                    d["none_right"] += 1
                    if len(fail_ex) < 3:
                        fail_ex.append((r["pidx"], g, t[-150:]))
            _ = bad
    out(f"## {name}")
    out(f"   샘플 {tot} · 완주 {fin_cnt} ({fin_cnt / max(1, tot):.0%})")
    out(f"   추출 None **{fail}** ({fail / max(1, tot):.1%}) · "
        f"그중 **마지막 boxed 가 정답인 것 {fail_but_right}** "
        f"({fail_but_right / max(1, tot):.1%} of 전체)")
    out(f"   {'서식':16s} {'샘플':>5s} {'None':>5s} {'None이지만정답':>13s}")
    for f, d in sorted(byfmt.items(), key=lambda kv: -kv[1]["n"]):
        out(f"   {f:16s} {d['n']:>5d} {d['none']:>5d} {d['none_right']:>13d}")
    for pidx, g, tail in fail_ex:
        out(f"   예: pidx {pidx} · gold {g} · 꼬리 {tail!r}")
    out("")
    return {"n": tot, "none": fail, "none_right": fail_but_right}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/results/extract_fail_audit.txt"))
    a = ap.parse_args()
    L = []

    def out(s=""):
        print(s, flush=True)
        L.append(s)

    ex = load_extract()
    out("extract_answer 추출 실패 감사 — '정답인데 못 뽑는' 샘플 (규칙 ② 가 잡은 결함)")
    out("")
    res = {}

    # (1) #TEMP-SENS T=1.0 선별 원문
    p = ROOT / "experiments/results/runpod/labelgen32k/i6hxd8z437eas4/out/Olmo-3-7B-Think_off40_v2_screen_texts.json"
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf-8"))
        res["olmo_T1.0_screen"] = audit("Olmo · T=1.0 top_k40 · 선별 (#TEMP-SENS)", d["rows"], ex, out)

    # (2) 같은 모델 greedy(T=0) — 참고값 (genprobe 산출)
    g = ROOT / "experiments/results/runpod/genprobe/oi8q4v8ia3ojq0/out/Olmo-3-7B-Think_greedy.json"
    if g.is_file():
        rows = [{"pidx": r["pidx"], "gold": r["gold"],
                 "samples": [{"text": r["greedy_text"], "fin": r.get("finished")}]}
                for r in json.loads(g.read_text(encoding="utf-8")) if r.get("greedy_text")]
        res["olmo_T0_greedy"] = audit("Olmo · greedy T=0 · 참고 (genprobe)", rows, ex, out)

    # (3) 대조군: DeepSeek 깨끗한 재생성 선별 (T=0.6) — 다른 모델이지만 같은 추출기
    for lbl, rel in (("DeepSeek · T=0.6 · 선별 (#DS-RELABEL off40)",
                      "experiments/results/runpod/labelgen32k/mbpc29h3v7um64/out/"
                      "DeepSeek-R1-0528-Qwen3-8B_off40_v2_screen_texts.json"),):
        q = ROOT / rel
        if q.is_file():
            d = json.loads(q.read_text(encoding="utf-8"))
            res[lbl] = audit(lbl, d["rows"], ex, out)

    out("## 읽는 법")
    out("   '추출 None 이지만 마지막 boxed 가 정답' 이 곧 **부당하게 깎인 base_acc** 다.")
    out("   Ġ 사고(오판율 0.293)와 같은 계열이지만 이번엔 원문을 저장했으므로 측정된다.")
    out("")
    out("   #TEMP-SENS 주지표 ②(base_acc 짝 비교)에 대한 함의: 두 조건이 같은 추출기를 쓰므로")
    out("   비교는 공정하지만, **서식 분포가 T 에 따라 달라지면** 실패율 차이가 '디코딩 효과' 로")
    out("   위장한다. Olmo off40 T=0.6 실행은 원문을 저장하지 않아(save_text 도입 전) 그쪽 실패율을")
    out("   직접 못 재는 것이 이 감사의 한계다 — greedy(T=0) 값이 유일한 참고점이다.")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
