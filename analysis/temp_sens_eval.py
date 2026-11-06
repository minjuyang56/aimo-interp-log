"""#TEMP-SENS — 디코딩 민감도(T 0.6→1.0 · top_k 40) 쌍별 평가. (지휘 #TEMP-SENS, 2026-10-12)

사전등록(CLAUDE.md `## #TEMP-SENS`, 커밋 a6e5a62)대로만 계산한다. 설정을 바꿔 다시 돌리면
그 사실을 기록에 남기고 '탐색' 으로 표기한다.

## 묻는 것
라벨 생성이 줄곧 `T=0.6 · top_p 0.95 · top_k 없음` 이었다. 그 설정이 샘플 다양성을 억눌러
'흔들림' 이 라벨에 덜 잡힌 것인가. → **같은 20문제**에서 디코딩만 바꿔 라벨을 다시 만들고
짝지어 비교한다.

## 주지표와 판정 (결과 보기 전 고정)
* ① spurious 비율: 새 라벨의 `label is False` 비율 vs T=0.6 의 **35%**(7/20).
* ② base_acc: 문제별 쌍의 차이(새 − 옛) 평균·부호 분포.
* 전환표 2×2 + McNemar **정확 이항검정** (불일치 b, c 에 대해 p = 2·P(X >= max(b,c)),
  X~Bin(b+c, 0.5)). `b+c < 5` 면 'n 부족' 으로 적고 p 를 주장하지 않는다.
* **판정**: `spurious >= 70%`(14/20) **그리고** `b > c`(robust→spurious 우세) 면
  "디코딩이 흔들림을 억눌렀다" **채택**, 아니면 **기각**.
* 두 집계를 다 낸다: **(ㄱ) 20문제 전체**(주판정 — BAND 해제로 전부 변형을 돌았다) ·
  (ㄴ) 새 `base_acc >= 0.5` 통과분만(옛 BAND 와 같은 조건).

## 옛 기준값을 npz 에서 읽는 이유
Olmo off40 k=8 의 **라벨 JSON 이 저장소에 없다**(`sprwmaodvz095s` 에 screen 만 있고 변형 단계
산출물이 회수되지 않았다). 라벨은 특징 npz 에만 남아 있다 — `own_k8.npz` 의 `y`·`y_fin`·
`y_major`·`pidx`, `gen512.npz` 의 `base_acc`. 그래서 옛 쪽은 `var_detail`(유형별 미완주/오답)과
`worst_kind` 를 쓸 수 없고, **결정 변형 유형·미완주 비율은 새 쪽만 적는다.**

사용:
  .venv/Scripts/python.exe analysis/temp_sens_eval.py --new <새 labels.json> [--out <보고서>]
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "experiments" / "results" / "probe"
SEL = [41, 42, 43, 44, 45, 47, 49, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64]
OLD_SPUR_RATE = 7 / 20          # 사전등록에 박은 T=0.6 기준값
OLD_BASE_MEAN = 0.9062


def binom_exact_two_sided(b, c):
    """McNemar 정확 이항검정. 불일치 n=b+c 에서 p=0.5 대비 양측."""
    n = b + c
    if n == 0:
        return None
    m = max(b, c)
    tail = sum(comb(n, i) for i in range(m, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def lab_str(v):
    return {True: "robust", False: "spurious", None: "중간대역"}.get(v, "라벨없음")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="새 라벨 JSON (*_off40_v2_labels.json)")
    ap.add_argument("--texts", default=None, help="새 screen_texts.json (평균 길이·마커 확인용)")
    ap.add_argument("--out", default=str(ROOT / "experiments/results/temp_sens_report.txt"))
    a = ap.parse_args()
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    # ---- 옛 (T=0.6) 기준값: npz 에서
    k8 = np.load(P / "Olmo-3-7B-Think_own_k8.npz", allow_pickle=True)
    g5 = np.load(P / "Olmo-3-7B-Think_gen512.npz", allow_pickle=True)
    old = {}
    for q, y, yf, ym in zip(k8["pidx"].astype(int), k8["y"].astype(int),
                            k8["y_fin"].astype(int), k8["y_major"].astype(int)):
        old[int(q)] = {"label": bool(y), "label_fin": (None if yf < 0 else bool(yf)),
                       "label_major": (None if ym < 0 else bool(ym))}
    for q, ba in zip(g5["pidx"].astype(int), g5["base_acc"].astype(float)):
        old[int(q)]["base_acc"] = float(ba)

    # ---- 새 (T=1.0 · top_k 40)
    rows = json.loads(Path(a.new).read_text(encoding="utf-8"))
    new = {int(r["pidx"]): r for r in rows}

    say("#TEMP-SENS — 디코딩 민감도 (T 0.6 → 1.0 · top_k 없음 → 40) · Olmo · 같은 20문제")
    say("사전등록: CLAUDE.md `## #TEMP-SENS` (커밋 a6e5a62, 생성 전 기록) · 1회 실행")
    say("")
    say(f"새 라벨 파일 {Path(a.new).name} · 행 {len(rows)} · "
        f"선정 20개 중 새 결과가 있는 것 {sum(1 for q in SEL if q in new)}")
    miss = [q for q in SEL if q not in new]
    if miss:
        say(f"!! 새 결과가 없는 pidx {miss} — 그 칸은 '라벨없음' 으로 센다")
    gk = sorted({r.get("gen_k") for r in rows}); gc = sorted({r.get("gen_cap") for r in rows})
    say(f"설정 확인: gen_k {gk} · gen_cap {gc} (옛 라벨과 같아야 한다: k=8 · 32768)")
    say("")

    # ---- 문제별 표
    say("## 문제별 (옛 → 새)")
    say(f"{'pidx':>5} {'옛label':>8} {'새label':>9} {'옛base':>7} {'새base':>7} {'Δbase':>7} "
        f"{'새max_drop':>10} {'새 미완주(원본)':>14} {'결정유형':>10}")
    pairs = []
    for q in SEL:
        o, n = old[q], new.get(q)
        if n is None:
            say(f"{q:>5} {lab_str(o['label']):>8} {'(없음)':>9} {o['base_acc']:>7.3f}")
            continue
        nb = n.get("base_acc")
        md = n.get("max_drop")
        nun = (n.get("n_fin") is not None) and (int(n.get("gen_k", 8)) - int(n["n_fin"]))
        va = {kk: vv for kk, vv in (n.get("var_acc") or {}).items() if vv is not None}
        worst = min(va, key=va.get) if va else "-"
        say(f"{q:>5} {lab_str(o['label']):>8} {lab_str(n.get('label')):>9} {o['base_acc']:>7.3f} "
            f"{(nb if nb is not None else float('nan')):>7.3f} "
            f"{((nb - o['base_acc']) if nb is not None else float('nan')):>7.3f} "
            f"{(md if md is not None else float('nan')):>10.3f} {str(nun):>14} {worst:>10}")
        pairs.append((q, o, n))
    say("")

    def block(title, qs):
        say(f"## {title} (n={len(qs)})")
        on = [old[q] for q in qs]
        nn = [new[q] for q in qs]
        # ① spurious 비율
        olds = sum(1 for o in on if o["label"] is False)
        news = sum(1 for n in nn if n.get("label") is False)
        nmid = sum(1 for n in nn if "label" in n and n.get("label") is None)
        say(f"① spurious  옛 {olds}/{len(qs)} = {olds / len(qs):.0%}  →  "
            f"새 {news}/{len(qs)} = {news / len(qs):.0%}   "
            f"(새 중간대역 제외 {nmid}건)")
        # ② base_acc
        d = [n["base_acc"] - o["base_acc"] for o, n in zip(on, nn) if n.get("base_acc") is not None]
        if d:
            d = np.array(d)
            say(f"② base_acc  옛 평균 {np.mean([o['base_acc'] for o in on]):.4f}  →  "
                f"새 평균 {np.mean([n['base_acc'] for n in nn if n.get('base_acc') is not None]):.4f}   "
                f"쌍별 Δ 평균 {d.mean():+.4f} · 중앙 {np.median(d):+.4f} · "
                f"내림 {int((d < 0).sum())} / 같음 {int((d == 0).sum())} / 오름 {int((d > 0).sum())}")
        # 전환표
        say("전환표 (행=옛, 열=새)")
        cats = [True, False, None, "없음"]
        hdr = "".join(f"{lab_str(c) if c != '없음' else '라벨없음':>10}" for c in cats)
        say(f"{'':>10}{hdr}")
        tab = {}
        for oc in (True, False):
            line = f"{lab_str(oc):>10}"
            for nc in cats:
                if nc == "없음":
                    v = sum(1 for o, n in zip(on, nn) if o["label"] is oc and "label" not in n)
                else:
                    v = sum(1 for o, n in zip(on, nn)
                            if o["label"] is oc and "label" in n and n.get("label") is nc)
                tab[(oc, nc)] = v
                line += f"{v:>10}"
            say(line)
        b = tab[(True, False)]      # robust → spurious
        c = tab[(False, True)]      # spurious → robust
        p = binom_exact_two_sided(b, c)
        say(f"McNemar  robust→spurious b={b} · spurious→robust c={c} · 불일치 {b + c}")
        if b + c < 5:
            say(f"  불일치 {b + c} < 5 → **n 부족, p 를 주장하지 않는다**"
                + (f" (참고 p={p:.3f})" if p is not None else ""))
        else:
            say(f"  정확 이항검정 p = {p:.4f}")
        # 판정
        rate = news / len(qs)
        g1 = rate >= 2 * OLD_SPUR_RATE
        g2 = b > c
        say(f"판정  기준1 spurious {rate:.0%} >= {2 * OLD_SPUR_RATE:.0%}(2배) → "
            f"{'통과' if g1 else '미달'}   기준2 b>c → {'통과' if g2 else '미달'}")
        say(f"  → **{'채택: 디코딩이 흔들림을 억눌렀다' if (g1 and g2) else '기각: 디코딩 설정으로 설명되지 않는다'}**")
        # 민감도 라벨
        for key in ("label_fin", "label_major"):
            os_ = sum(1 for o in on if o.get(key) is False)
            ns_ = sum(1 for n in nn if n.get(key) is False)
            ov = sum(1 for o in on if o.get(key) is not None)
            nv = sum(1 for n in nn if n.get(key) is not None)
            say(f"민감도 {key}: 옛 spurious {os_}/{ov} → 새 {ns_}/{nv}"
                + ("  (양성 0 → 적용 불가)" if nv and ns_ == 0 and os_ == 0 else ""))
        say("")

    have = [q for q in SEL if q in new]
    block("(ㄱ) 20문제 전체 — 주판정", have)
    band = [q for q in have if (new[q].get("base_acc") or 0) >= 0.5]
    if len(band) != len(have):
        block("(ㄴ) 새 base_acc >= 0.5 통과분만 (옛 BAND 와 같은 조건)", band)
    else:
        say(f"## (ㄴ) 새 base_acc >= 0.5 통과분 = 전체 {len(have)}개 → (ㄱ)과 같다\n")

    # ---- 추출 교정 (사전등록 주지표의 대체가 **아니다** — 병기만 한다)
    say("## 추출 교정 base_acc (민감도 · 판정에 쓰지 않는다)")
    say("   규칙 ② 육안 점검이 잡은 결함: `extract_answer` 가 `\\[ \\boxed{..} \\]` 서식을 못 뽑아")
    say("   **완주·정답 샘플이 오답으로 집계된다**(Olmo T=1.0 선별 160샘플 중 8 = 5.0%).")
    say("   원문을 저장했으므로(규칙 ①) 교정값을 계산할 수 있다 — 마지막 `\\boxed{}` 가 gold 와")
    say("   같으면 정답으로 센 base_acc. **옛 T=0.6 쪽은 원문이 없어 교정이 불가능하므로**")
    say("   이 열은 '교정하면 어느 방향으로 움직이나' 만 보여주고 짝 비교에 쓰지 않는다.")
    tx = Path(a.new).parent / Path(a.new).name.replace("_labels.json", "_screen_texts.json")
    if tx.is_file():
        import re as _re
        bx = _re.compile(r"\\boxed\{([^{}]*)\}")

        def eq(x, g):
            try:
                return int(str(x).replace(",", "").strip()) == int(str(g).replace(",", "").strip())
            except (TypeError, ValueError):
                return str(x).strip() == str(g).strip()

        # **합집합 규칙이어야 교정이다.** 처음엔 '마지막 boxed 로 재채점' 만 했는데 그건 교정이
        # 아니라 **다른 추출기**다 — `ANSWER:` 줄만 있고 boxed 가 중간 계산값인 샘플을 망쳐
        # 올라간 문제 6 / 내려간 문제 9 로 양방향이 됐다. 규칙 ② 가 잡은 결함은
        # '`extract_answer` 가 None 을 주는데 사실 정답' 인 경우이므로, 교정은
        # **기존 추출기가 맞히면 그대로 + None 일 때만 마지막 boxed 로 구제** 다.
        import sys as _sys
        _ns = {"__name__": "sc"}
        exec(compile((ROOT / "experiments/kaggle/solve_common.py").read_text(encoding="utf-8"),
                     "solve_common.py", "exec"), _ns)
        _ex = _ns["extract_answer"]
        d = json.loads(tx.read_text(encoding="utf-8"))
        fix, rescued = {}, 0
        for r in d["rows"]:
            k = len(r["samples"])
            hit = 0
            for s in r["samples"]:
                if not s.get("fin"):
                    continue
                got = _ex(s["text"])
                if got is not None and eq(got, r["gold"]):
                    hit += 1
                elif got is None:
                    m = bx.findall(s["text"])
                    if m and eq(m[-1], r["gold"]):
                        hit += 1
                        rescued += 1
            fix[int(r["pidx"])] = hit / max(1, k)
        say(f"   합집합 교정으로 **구제된 샘플 {rescued}개** (추출 None 이었지만 마지막 boxed 가 정답)")
        qs = [q for q in SEL if q in new and q in fix]
        obs = np.array([new[q]["base_acc"] for q in qs], dtype=float)
        cor = np.array([fix[q] for q in qs], dtype=float)
        say(f"   n {len(qs)} · 관측 base_acc 평균 **{obs.mean():.4f}** → 교정 **{cor.mean():.4f}** "
            f"(차 {cor.mean() - obs.mean():+.4f}) · 올라간 문제 {int((cor > obs).sum())} / "
            f"같음 {int((cor == obs).sum())} / 내려간 문제 {int((cor < obs).sum())}")
        say(f"   참고: 옛 T=0.6 관측 평균 {OLD_BASE_MEAN:.4f} (교정 불가)")
    else:
        say(f"   !! 원문 파일이 없다: {tx.name} — 교정 생략")
    say("")

    # ---- 부가 (판정에 쓰지 않는다)
    say("## 부가 (판정에 쓰지 않는다)")
    k = int(new[have[0]].get("gen_k", 8)) if have else 8
    nfin = [int(new[q]["n_fin"]) for q in have if new[q].get("n_fin") is not None]
    if nfin:
        say(f"원본 미완주율(새): {1 - sum(nfin) / (k * len(nfin)):.1%} "
            f"(완주 {sum(nfin)} / {k * len(nfin)} 샘플)")
    vu = vw = vn = 0
    for q in have:
        for dd in (new[q].get("var_detail") or {}).values():
            vu += dd.get("n_unfin", 0); vw += dd.get("n_wrong", 0); vn += dd.get("n_fin", 0) + dd.get("n_unfin", 0)
    if vn:
        say(f"변형 샘플(새) {vn}개 · 미완주 {vu} ({vu / vn:.1%}) · 완주오답 {vw} ({vw / vn:.1%})")
    kinds = {}
    for q in have:
        n = new[q]
        if n.get("label") is False:
            va = {kk: vv for kk, vv in (n.get("var_acc") or {}).items() if vv is not None}
            if va:
                w = min(va, key=va.get)
                kinds[w] = kinds.get(w, 0) + 1
    say(f"spurious 결정 변형 유형(새): {kinds or '없음'} "
        f"— 옛 쪽은 라벨 JSON 이 없어 비교 불가(docstring 참조)")
    if a.texts and Path(a.texts).is_file():
        t = json.loads(Path(a.texts).read_text(encoding="utf-8"))
        ch = [len(sm["text"]) for r in t["rows"] for sm in r["samples"]]
        say(f"새 원본 샘플 {len(ch)}개 · 평균 {np.mean(ch):,.0f}자 · 중앙 {np.median(ch):,.0f}자 · "
            f"파일 기록 BPE 마커 {t.get('n_marks')}")
    say("")
    say("한계(사전등록에 미리 적음): vLLM 은 seed 고정에도 배치·스케줄링에 따라 같은 출력을")
    say("보장하지 않는다 → 라벨 변화에서 '디코딩 설정' 과 '재샘플 변동' 을 가를 수 없다.")
    say("그래서 기준1 을 2배로 높게 잡았다. n=20 이고 옛 spurious 7건이라 전환 한두 건이")
    say("비율을 5~10%p 움직인다. 이 결과는 #77·#GEN-PROBE 의 판정을 바꾸지 않는다.")

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
