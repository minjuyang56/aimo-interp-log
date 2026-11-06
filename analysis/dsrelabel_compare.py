"""#DS-RELABEL — 옛(손상) DeepSeek 라벨과 새(깨끗) 라벨을 **같은 잣대로 맞댄다**. GPU 0.

왜 이 표가 먼저 나와야 하나: 재평가 숫자가 달라졌을 때 그 원인이 (a) 디토큰화 손상 제거인지
(b) k=8 재샘플의 변동인지 나눠 읽어야 한다. 둘을 완전히 가를 수는 없지만(사전등록 한계에 적었다),
**base_acc 가 올랐는가**는 (a) 의 직접 증거다 — 손상은 정답을 오답으로 세어 base_acc 를
약 0.28 깎았고(갭 0.280 ≈ 오판율 0.293), 그래서 채택(>=0.5)·라벨이 같이 틀어졌다.

적는 것: 채택 수 · 라벨 수·클래스 · base_acc 평균 · 라벨이 겹치는 pidx 에서의 **라벨 일치율** ·
경계 1샘플 판정 · spurious 원인(오답 vs 미완주) · label_fin 일치 · 결정 변형 유형.

사용: .venv/Scripts/python.exe analysis/dsrelabel_compare.py
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LG = ROOT / "experiments" / "results" / "runpod" / "labelgen32k"
OUT = ROOT / "experiments" / "results" / "dsrelabel_compare.txt"

# 옛 경로는 #TOK-AUDIT 에 적힌 그 두 파드다. 새 경로는 글롭으로 찾는다(파드 id 를 미리 모른다).
OLD = {"off40": "luvw8yu3mkxsza/out/DeepSeek-R1-0528-Qwen3-8B_v3_labels.json",
       "off80": "1f7jul0dgk7o9s/out/DeepSeek-R1-0528-Qwen3-8B_off80_v2_labels.json"}
NEW_GLOB = {"off40": "*/out/DeepSeek-R1-0528-Qwen3-8B_off40_v2_labels.json",
            "off80": "*/out/DeepSeek-R1-0528-Qwen3-8B_off80_v2_labels.json"}
# **옛 파드를 이름으로 배제한다.** 새 off80 산출물의 파일명이 옛 off80 두 개와 같아서
# 글롭이 옛 파일을 집어 '옛 vs 옛' 을 비교했다(첫 실행에서 100% 일치가 나와서 들통났다).
# mtime 으로 가르면 scp 회수 시각에 끌려가므로 파드 id 로 가른다 — 감사 가능한 쪽이다.
OLD_PODS = {"luvw8yu3mkxsza", "67elv2je7wzefw", "1f7jul0dgk7o9s", "x04ghnc52b6dll"}
DROP_SPUR = 0.25
lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def stats(rows, k=8):
    lab = [r for r in rows if r.get("label") is not None]
    sp = [r for r in lab if r["label"] is False]
    # 경계 판정 = 그 유형에서 **실패가 1샘플뿐**인데 spurious 가 된 경우
    edge = 0
    for r in sp:
        det = r.get("var_detail") or {}
        worst = min(((v, kk) for kk, v in (r.get("var_acc") or {}).items() if v is not None),
                    default=(None, None))
        if worst[1] is None:
            continue
        d = det.get(worst[1]) or {}
        if (d.get("n_unfin", 0) + d.get("n_wrong", 0)) == 1:
            edge += 1
    kinds = Counter()
    for r in sp:
        worst = min(((v, kk) for kk, v in (r.get("var_acc") or {}).items() if v is not None),
                    default=(None, None))
        if worst[1]:
            kinds[worst[1]] += 1
    nun = sum(d.get("n_unfin", 0) for r in sp for d in (r.get("var_detail") or {}).values())
    nwr = sum(d.get("n_wrong", 0) for r in sp for d in (r.get("var_detail") or {}).values())
    both = [r for r in lab if r.get("label_fin") is not None]
    agree = sum(1 for r in both if r["label"] == r["label_fin"])
    maj = [r for r in lab if r.get("label_major") is not None]
    return dict(
        accepted=len(rows), n=len(lab),
        rob=sum(1 for r in lab if r["label"]), spur=len(sp),
        base=sum(r["base_acc"] for r in rows) / max(1, len(rows)),
        edge=edge, kinds=dict(kinds), n_unfin=nun, n_wrong=nwr,
        fin_agree=f"{agree}/{len(both)}",
        maj_pos=sum(1 for r in maj if r["label_major"] is False), maj_n=len(maj),
        pidx={r["pidx"]: bool(r["label"]) for r in lab})


def main():
    out("#DS-RELABEL — 옛(손상) vs 새(깨끗) DeepSeek k=8 라벨 (GPU 0)")
    out("손상은 정답을 오답으로 세어 base_acc 를 약 0.28 깎았다 → 채택(>=0.5)·라벨이 같이 틀어졌다.")
    out("그래서 **base_acc 평균이 올랐는가**가 손상 제거의 직접 증거다.")
    out()
    for slc in ("off40", "off80"):
        old_p = LG / OLD[slc]
        new_hits = sorted((q for q in LG.glob(NEW_GLOB[slc])
                           if q.parent.parent.name not in OLD_PODS),
                          key=lambda q: q.stat().st_mtime)
        out(f"## {slc}")
        if not old_p.is_file():
            out(f"   옛 파일 없음: {old_p}")
            continue
        o = stats(load(old_p))
        if not new_hits:
            out(f"   새 라벨 아직 없음 (글롭 {NEW_GLOB[slc]}) — 옛 값만 적는다")
            out(f"   옛: 채택 {o['accepted']} · 라벨 {o['n']} (robust {o['rob']} / spurious {o['spur']}) · "
                f"base_acc 평균 {o['base']:.3f}")
            out()
            continue
        n = stats(load(new_hits[-1]))
        out(f"   파일: 옛 {old_p.parent.parent.name} · 새 {new_hits[-1].parent.parent.name}")
        out(f"   {'항목':<22}{'옛(손상)':>14}{'새(깨끗)':>14}")
        rowfmt = "   {:<22}{:>14}{:>14}"
        out(rowfmt.format("채택 문제", o["accepted"], n["accepted"]))
        out(rowfmt.format("라벨 수", o["n"], n["n"]))
        out(rowfmt.format("robust / spurious", f"{o['rob']} / {o['spur']}", f"{n['rob']} / {n['spur']}"))
        out(rowfmt.format("spurious 비율", f"{o['spur'] / max(1, o['n']):.0%}", f"{n['spur'] / max(1, n['n']):.0%}"))
        out(rowfmt.format("base_acc 평균", f"{o['base']:.3f}", f"{n['base']:.3f}"))
        out(rowfmt.format("경계 1샘플 판정", o["edge"], n["edge"]))
        out(rowfmt.format("spurious 원인 오답", o["n_wrong"], n["n_wrong"]))
        out(rowfmt.format("spurious 원인 미완주", o["n_unfin"], n["n_unfin"]))
        out(rowfmt.format("label_fin 일치", o["fin_agree"], n["fin_agree"]))
        out(rowfmt.format("과반 양성(spurious)", f"{o['maj_pos']}/{o['maj_n']}", f"{n['maj_pos']}/{n['maj_n']}"))
        out(rowfmt.format("결정 유형", str(o["kinds"]), str(n["kinds"])))
        共 = sorted(set(o["pidx"]) & set(n["pidx"]))
        same = sum(1 for p in 共 if o["pidx"][p] == n["pidx"][p])
        out(f"   라벨이 양쪽에 다 있는 pidx {len(共)}개 · **일치 {same}/{len(共)}**"
            f"{'' if not 共 else f' = {same / len(共):.0%}'}")
        out(f"   새로 라벨된 pidx {sorted(set(n['pidx']) - set(o['pidx']))}")
        out(f"   사라진 pidx {sorted(set(o['pidx']) - set(n['pidx']))}")
        out()
    out("## 읽는 법")
    out("   base_acc 가 올랐고 라벨이 바뀌었다면 두 원인(손상 제거 · k=8 재샘플 변동)이 섞여 있다.")
    out("   같은 설정 재실행 없이는 그 둘을 가를 수 없다(사전등록 한계에 미리 적었다).")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
