"""라벨 재현성 — **같은 모델·같은 k·같은 cap·같은 디코딩의 두 독립 실행**에서 같은 문제가
같은 라벨을 받는가. 비용 $0 (이미 받은 라벨 JSON 만 읽는다).

#TEMP-SENS 가 "가르려면 같은 설정으로 한 번 더 돌려 변동폭을 재야 하고(+$6~9) 그게 이 숫자들의
해석을 정하는 유일한 실험이다" 라고 적어 둔 그 실험이 **이미 공짜로 돌아 있었다** — 다만 아무도
그걸 몰랐다. 이유는 `genlab_v2.load_pool(n)` 의 성질이다:

    idx = permutation(len(rows))[:n] ;  out = [rows[i] for i in sorted(idx)] ;  pidx = 0..n-1

즉 **n 이 달라지면 같은 pidx 가 다른 문제**이고(그건 #TEMP-SENS 에 적혀 있다), 그 반대쪽 얼굴로
**큰 풀의 뒤쪽 슬라이스가 작은 풀의 뒤쪽과 문제를 공유한다**. off40 = `load_pool(80)[40:]` ·
off80 = `load_pool(120)[80:]` 이므로 pidx 는 40–79 / 80–119 로 안 겹치는데 **문제 텍스트는 겹친다.**
→ 그 겹친 문제들이 **같은 설정의 재실행**이다.

보고하는 것
  * 묶음 쌍별 겹친 문제 수 · **라벨 일치율** · `max_drop` 완전동일 수
  * **천장 효과 분리**: 양쪽 다 `base_acc == 1.0` 이고 `max_drop == 0` 인 쌍(자동 일치)과 나머지
  * 부수: 이 겹침이 과거 기록의 '독립 묶음' 서술을 정정한다는 점(#DS-REP · #GEN-PROBE)

사용: .venv/Scripts/python.exe analysis/label_repeat.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments/results"
OUT = RES / "label_repeat.txt"
WS = re.compile(r"\s+")
LINES = []

#  깨끗한 묶음만. Ġ 손상 pod(67elv2je7wzefw · 1f7jul0dgk7o9s · luvw8yu3mkxsza · x04ghnc52b6dll)과
#  T=1.0 pod(i6hxd8z437eas4)는 **설정이 다르므로** 재현성 쌍에 넣지 않는다(#R5-TRANSFER 와 같은 규칙).
BUNDLES = [
    ("DeepSeek k=8 off40", 8, "runpod/labelgen32k/mbpc29h3v7um64/out/DeepSeek-R1-0528-Qwen3-8B_off40_v2_labels.json"),
    ("DeepSeek k=8 off80", 8, "runpod/labelgen32k/9svqsw8dzrtjsv/out/DeepSeek-R1-0528-Qwen3-8B_off80_v2_labels.json"),
    ("Skywork k=8 off40", 8, "runpod/labelgen32k/pj7vh1qllei51i/out/Skywork-OR1-Math-7B_off40_v2_labels.json"),
]
#  Skywork k=4 는 두 슬라이스(off0 · off50)가 **한 파일로 병합**돼 있다 — 그 안의 중복이 쌍이다.
MERGED = ("Skywork k=4 (off0 ↔ off50, 병합 파일 내부 중복)", 4,
          "genlab/skywork_labels_merged.json")


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def norm(t):
    return WS.sub(" ", str(t)).strip()


def wilson_upper(k, n, z=1.96):
    """이항 비율의 윌슨 95% 상단. k=1·n=17 처럼 작은 수에서 정규근사보다 정직하다."""
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c + h) / d


def load(rel):
    p = RES / rel
    if not p.is_file():
        return None
    o = json.loads(p.read_text(encoding="utf-8"))
    rows = o if isinstance(o, list) else o.get("rows", [])
    return [r for r in rows if isinstance(r.get("label"), bool)]


def trivial(a, b):
    """양쪽 다 '원본 전부 맞히고 변형이 하나도 안 무너짐' → 일치가 자동이다(천장 효과)."""
    return (a.get("base_acc") == 1.0 and b.get("base_acc") == 1.0
            and a.get("max_drop") == 0 and b.get("max_drop") == 0)


def report_pairs(name, k, pairs):
    if not pairs:
        out("   %s — 겹친 문제 0" % name)
        return None
    agree = sum(1 for a, b in pairs if a["label"] == b["label"])
    same_md = sum(1 for a, b in pairs if a.get("max_drop") == b.get("max_drop"))
    triv = [(a, b) for a, b in pairs if trivial(a, b)]
    hard = [(a, b) for a, b in pairs if not trivial(a, b)]
    ha = sum(1 for a, b in hard if a["label"] == b["label"])
    out("## %s · k=%d" % (name, k))
    out("   겹친 문제(라벨 양쪽) **%d** · **라벨 일치 %d/%d = %.3f** · max_drop 완전동일 %d"
        % (len(pairs), agree, len(pairs), agree / len(pairs), same_md))
    out("   천장 쌍(양쪽 base_acc 1.0 · max_drop 0 → 자동 일치) %d · **나머지 %d 중 일치 %d**%s"
        % (len(triv), len(hard), ha,
           " = %.3f" % (ha / len(hard)) if hard else ""))
    flips = [(a, b) for a, b in pairs if a["label"] != b["label"]]
    for a, b in flips:
        out("   뒤집힘: pidx %s→%s · label %s→%s · max_drop %s→%s · base_acc %s→%s"
            % (a.get("pidx"), b.get("pidx"), a["label"], b["label"],
               a.get("max_drop"), b.get("max_drop"), a.get("base_acc"), b.get("base_acc")))
    return dict(name=name, k=k, n=len(pairs), agree=agree, hard=len(hard), hard_agree=ha)


def main():
    out("라벨 재현성 — 같은 모델·같은 k·같은 cap·같은 디코딩의 **두 독립 실행** 일치율 · 비용 $0")
    out("   #TEMP-SENS 가 +$6~9 로 적어 둔 실험이 load_pool 의 슬라이스 겹침으로 이미 돌아 있었다")
    out()

    got = {}
    for name, k, rel in BUNDLES:
        rows = load(rel)
        got[name] = (k, rows)
        out("   읽음 %-22s %s" % (name, ("%d행" % len(rows)) if rows else "**없음**"))
    mrows = load(MERGED[2])
    out("   읽음 %-22s %s" % ("Skywork k=4 merged", ("%d행" % len(mrows)) if mrows else "**없음**"))
    out()

    stats = []
    #  (1) 서로 다른 두 파일 사이의 겹침 — 같은 모델·같은 k 끼리만 맞댄다
    names = [n for n in got if got[n][1]]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            ka, ra = got[na]
            kb, rb = got[nb]
            if ka != kb or na.split()[0] != nb.split()[0]:
                continue
            da = {norm(r["problem"]): r for r in ra}
            db = {norm(r["problem"]): r for r in rb}
            pairs = [(da[t], db[t]) for t in sorted(set(da) & set(db))]
            s = report_pairs("%s ↔ %s" % (na, nb), ka, pairs)
            if s:
                stats.append(s)
            out()

    #  (2) 병합 파일 내부 중복 — 두 슬라이스가 한 파일로 합쳐진 경우
    if mrows:
        g = defaultdict(list)
        for r in mrows:
            g[norm(r["problem"])].append(r)
        pairs = [(v[0], v[1]) for v in g.values() if len(v) > 1]
        out("   (병합 파일: 행 %d · 고유 문제 %d · 중복 문제 %d — 과거 'n=88' 서술의 유효 표본은 %d 에 가깝다)"
            % (len(mrows), len(g), len(pairs), len(g)))
        s = report_pairs(MERGED[0], MERGED[1], pairs)
        if s:
            stats.append(s)
        out()

    out("## 요약 — k 가 재현성을 바꾸는가")
    out("   %-46s%4s%6s%9s%7s%11s" % ("쌍", "k", "쌍수", "일치율", "비천장", "비천장일치"))
    for s in sorted(stats, key=lambda d: -d["k"]):
        out("   %-46s%4d%6d%9.3f%7d%11s"
            % (s["name"][:46], s["k"], s["n"], s["agree"] / s["n"], s["hard"],
               ("%.3f" % (s["hard_agree"] / s["hard"])) if s["hard"] else "—"))
    out()
    out("## 읽는 법 — 무엇이 확정되고 무엇이 안 되는가")
    out("   * **원 일치율은 천장 쌍이 지배한다**: k=8 쪽 17쌍 중 14쌍이 양쪽 base_acc 1.0 ·")
    out("     max_drop 0 으로 **자동 일치**다(그 모델 라벨이 29/3 으로 쏠려 있다). 그래서")
    out("     0.941 vs 0.737 을 'k 효과' 로 읽으면 안 된다 — 클래스 쏠림이 섞여 있다.")
    out("   * **비천장 쌍만 보면 3개 vs 9개다** → 방향은 일관하지만(0.667 vs 0.444) **n 이 3이라")
    out("     크기도 유의성도 주장할 수 없다.** 이 표의 값은 '재현성이 완벽하지 않다' 까지다.")
    out("   * **기제는 선명하다**: k=4 의 뒤집힘 5건이 전부 max_drop 0.0↔0.25(또는 0.5) 다 —")
    out("     Amendment 2 가 '변형 12샘플 중 하나만 틀려도 spurious' 로 지적한 **경계 1샘플 판정**")
    out("     바로 그것이다. 즉 뒤집힘의 원인이 무작위가 아니라 **임계 근처의 해상도**다.")
    out("   * **정정**: #DS-REP 사전등록의 'off40 과 off80 교집합 0' 은 **pidx 기준**이었고 문제")
    out("     텍스트로는 겹친다. #GEN-PROBE 가 off80 을 '독립 묶음' 민감도로 쓴 서술도 같이 내린다.")
    out("     두 곳 모두 결론은 바뀌지 않지만 '독립' 이라는 말은 못 쓴다.")
    out("   * **함의(약한 증거)**: k=8 에서 관측된 뒤집힘은 17쌍 중 **1건**이다. 윌슨 95% 상단이")
    out("     %.0f%% 이므로 #TEMP-SENS 의 40%%(8/20) 변동을 **순수 재샘플만으로 설명하기는 어렵다**"
        % (100 * wilson_upper(1, 17)))
    out("     → 그 변동의 상당 부분은 개입(디코딩 변경) 쪽이라는 쪽으로 기운다. 다만 **모델이")
    out("     다르고**(DeepSeek 29/3 vs Olmo 21/10) 클래스 쏠림도 달라 **같은 축의 비교가 아니다** —")
    out("     확정하려면 #TEMP-SENS 와 같은 묶음에서 T=0.6 을 한 번 더 돌려야 한다(+$6~9, 미실행).")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))


if __name__ == "__main__":
    main()
