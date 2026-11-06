"""파일럿 판정 — 통과/실패를 **종료코드로** 돌려준다(감시 스크립트가 그걸 보고 본실행을 push). GPU 0.

사전등록 통과 기준 (지휘 #70 계획서 §6, 결과 보기 전 고정)
  라벨(robust/spurious 확정분) 중 **spurious 비율 ≥ 15%**.
  근거: v1 본실행은 공식 라벨 85건 중 spurious 2건(2%)이라 null 구간이 [0.11, 0.89] 로 벌어져
  어떤 지표도 판정할 수 없었다. 한쪽으로 쏠린 라벨은 수를 늘려도 쓸모가 없다.

같이 보고하는 것(판정에는 쓰지 않음): 채택률 · base_acc 분포 · max_drop 분포 ·
**원인 분해**(오답 n_wrong vs 미완주 n_unfin) — 개정 #37-2 의 '가짜 spurious' 감시용.

종료코드: 0 통과 · 2 기준 미달 · 3 파일 없음
사용:  .venv/Scripts/python.exe analysis/labelgen_pilot_eval.py [--glob ...] [--min-spurious 0.15]
"""

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "experiments/results/genlab/aimo-lg-pilot-*/**/*_v2_*.json"))
    # 실제 파일명은 <별칭>_v2_labels.json 과 _v2_screen.json 이다. 라벨 파일이 비어 있으면(채택 0)
    # screen 파일로 떨어져 base_acc 분포까지는 보고한다 — 그래야 "왜 0 인지" 가 로그에 남는다.
    ap.add_argument("--min-spurious", type=float, default=0.15)
    a = ap.parse_args()
    hits = sorted(glob.glob(a.glob, recursive=True))
    if not hits:
        alt = sorted(glob.glob(str(ROOT / "experiments/results/genlab/aimo-lg-pilot-*/**/*.json"), recursive=True))
        print(f"파일럿 결과 없음: {a.glob}\n  후보: {[Path(x).name for x in alt][:8]}")
        return 3
    rows = []
    for f in hits:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (isinstance(d, list) and d and isinstance(d[0], dict) and "base_acc" in d[0]):
            continue
        has_label = any(isinstance(r.get("label"), bool) for r in d)
        # **라벨이 든 파일을 우선한다.** _v2_labels.json 과 _v2_screen.json 이 같이 오는데
        # screen 쪽은 label 이 전부 None 이라(변형 단계 전에 저장됨) 덮어쓰면 "라벨 0건" 으로 오판한다.
        if has_label or not rows:
            rows = d
            src = f
        if has_label:
            break
    if not rows:
        print(f"스키마가 맞는 파일 없음 (후보 {len(hits)}개)")
        return 3

    acc = collections.Counter(round(r["base_acc"], 2) for r in rows if r.get("base_acc") is not None)
    kept = [r for r in rows if r.get("accepted")]
    # label 은 **bool** 로 저장된다: True=robust · False=spurious · None=중간대역 제외.
    # 처음엔 문자열("robust"/"spurious")로 가정해 라벨 16건을 0건으로 오판했다.
    labeled = [r for r in kept if isinstance(r.get("label"), bool)]
    spur = [r for r in labeled if r["label"] is False]
    rob = [r for r in labeled if r["label"] is True]
    mid = [r for r in kept if r.get("max_drop") is not None and not isinstance(r.get("label"), bool)]
    print(f"파일럿 결과: {src}")
    print(f"  문제 {len(rows)} · base_acc 분포 {dict(sorted(acc.items()))}")
    print(f"  채택 {len(kept)} ({len(kept) / max(1, len(rows)):.0%}) · 라벨 확정 {len(labeled)} "
          f"(robust {len(rob)} / spurious {len(spur)}) · 중간대역 제외 {len(mid)}")
    drops = [r["max_drop"] for r in kept if r.get("max_drop") is not None]
    if drops:
        print(f"  max_drop: 최소 {min(drops):.2f} · 중앙 {sorted(drops)[len(drops) // 2]:.2f} · 최대 {max(drops):.2f}")
    # 원인 분해 — 가짜 spurious(변형이 길이만 늘려 상한에 걸린 경우) 감시
    nw = nu = 0
    for r in spur:
        for kind, det in (r.get("var_detail") or {}).items():
            if isinstance(det, dict):
                nw += int(det.get("n_wrong", 0))
                nu += int(det.get("n_unfin", 0))
    if nw or nu:
        print(f"  spurious 원인 분해: 오답 {nw} · 미완주 {nu} (미완주 비중 {nu / max(1, nw + nu):.0%})")
    if not labeled:
        print("판정: **실패** — 라벨이 0건이다")
        return 2
    frac = len(spur) / len(labeled)
    ok = frac >= a.min_spurious
    print(f"판정: **{'통과' if ok else '실패'}** — spurious 비율 {frac:.0%} "
          f"(기준 {a.min_spurious:.0%}) · 라벨 {len(labeled)}건")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
