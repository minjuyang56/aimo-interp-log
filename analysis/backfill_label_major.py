"""기존 라벨 파일에 과반 규칙 라벨(`label_major`)을 소급 기록한다. GPU 0. (지휘 #RULE-A)

왜 공짜인가: `var_acc` 가 **섭동 유형별로** 저장돼 있어서 규칙만 다시 적용하면 된다.
GPU 가 필요한 축은 k(샘플 해상도)뿐이다.

주라벨은 **건드리지 않는다**(#77 사전등록 유지). `label_major` 는 민감도 분석용 열이다.
  규칙: 섭동 유형 중 `(base_acc − var_acc) ≥ 0.25` 인 유형이 **과반**이면 spurious.
  같이 저장: `n_broken_types` · `n_types` (사후에 규칙을 재구성할 수 있게).

사용:  .venv/Scripts/python.exe analysis/backfill_label_major.py [--write]
       --write 없이 돌리면 무엇이 바뀔지만 찍는다.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DROP_SPUR = 0.25
BASES = [ROOT / "experiments" / "results" / "genlab",
         ROOT / "experiments" / "results" / "runpod"]


def major_of(r):
    va = {k: v for k, v in (r.get("var_acc") or {}).items() if v is not None}
    base = r.get("base_acc")
    if not va or base is None:
        return None, None, None
    nb = sum(1 for v in va.values() if (base - v) >= DROP_SPUR)
    return bool(nb <= len(va) / 2), nb, len(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    files = []
    for b in BASES:
        if b.is_dir():
            files += list(b.rglob("*labels*.json"))
    files = sorted(set(files))
    if not files:
        print("라벨 파일 없음")
        return

    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  건너뜀 {f.name}: {type(e).__name__}")
            continue
        if not isinstance(d, list):
            continue
        n_set = flips = 0
        for r in d:
            if not isinstance(r, dict):
                continue
            m, nb, nt = major_of(r)
            r["label_major"], r["n_broken_types"], r["n_types"] = m, nb, nt
            n_set += 1
            if isinstance(r.get("label"), bool) and m is not None and bool(r["label"]) != m:
                flips += 1
        lab = [r for r in d if isinstance(r.get("label"), bool)]
        maj = [r for r in d if isinstance(r.get("label_major"), bool)]
        print(f"{f.relative_to(ROOT)}")
        print(f"   레코드 {n_set} · 주라벨 {len(lab)} "
              f"(spurious {sum(1 for r in lab if not r['label'])}) · "
              f"과반라벨 {len(maj)} (spurious {sum(1 for r in maj if not r['label_major'])}) · "
              f"**어긋남 {flips}**")
        if a.write:
            f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print()
    print("기록했다." if a.write else "--write 를 주면 파일에 기록한다 (지금은 미리보기).")


if __name__ == "__main__":
    main()
