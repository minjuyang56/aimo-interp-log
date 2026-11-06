"""cases.jsonl의 permutation_type이 라벨을 그대로 누출하는지 확인한다.

이 필드는 "어떤 방식으로 문제를 흔들었는가"를 담는 리스트다. 문제의 성질이 아니라
라벨을 만든 절차의 흔적이다. 절차가 라벨과 상관돼 있으면 그대로 누출이 된다.

여기서 확인하는 것:
  1. 리스트 길이만으로 라벨이 갈리는가
  2. ingestion.py가 이 필드를 솔루션에 실제로 넘기는가
"""

import json
import re
from collections import Counter

from _paths import CASES, GETTING_STARTED, LABELS, require


def main():
    ingestion = GETTING_STARTED / "components" / "ingestion_program" / "ingestion.py"
    require(CASES, LABELS, ingestion)

    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["id"]] = row["is_robust"]

    print(f"cases {len(cases)}건 / labels {len(labels)}건")

    # 2026-09-14 업스트림 schema_version 2: problem이 dict가 아니라 str이고
    # permutation_type 필드 자체가 사라졌다. 누출 경로가 데이터에서 제거된 것.
    if cases and isinstance(cases[0].get("problem"), str):
        print()
        print("schema_version 2 — cases.jsonl에 permutation_type이 더는 없다.")
        print("  업스트림 커밋 ad7d3af 'Remove obsolete permutation_type field'.")
        print("  이 스크립트가 보여주던 28/28 누출은 옛 스키마(v1)의 기록이다.")
        print("  옛 스키마로 다시 보려면 HF parquet를 직접 읽는 model_confound.py를 쓸 것.")
        return

    # 1. 길이만으로 갈리는가 --------------------------------------------------
    print()
    print("[1] permutation_type 리스트 길이 x 라벨")
    table = Counter((len(c["problem"]["permutation_type"]), labels[c["id"]]) for c in cases)
    lengths = sorted({length for length, _ in table})
    print(f"    {'길이':>6} {'robust=True':>12} {'robust=False':>13}")
    separable = True
    for length in lengths:
        n_true = table[(length, True)]
        n_false = table[(length, False)]
        if n_true and n_false:
            separable = False
        print(f"    {length:>6} {n_true:>12} {n_false:>13}")

    if separable:
        correct = sum(
            1 for c in cases
            if (len(c["problem"]["permutation_type"]) > 1) == labels[c["id"]]
        )
        print()
        print(f"    길이 > 1 을 그대로 예측으로 쓰면: {correct}/{len(cases)}"
              f" = {correct / len(cases):.3f}")
        print("    문제 텍스트를 한 글자도 안 보고 나온 점수다. 이건 신호가 아니라 누출이다.")

    # 2. 개별 값 빈도 ---------------------------------------------------------
    print()
    print("[2] 개별 값 빈도 (리스트를 펼쳐서 셈)")
    flat = Counter(t for c in cases for t in c["problem"]["permutation_type"])
    for name, count in flat.most_common():
        print(f"    {name:24s} {count}")

    # 3. ingestion.py가 넘기는 필드 -------------------------------------------
    print()
    print("[3] ingestion.py가 솔루션에 실제로 넘기는 것")
    source = ingestion.read_text(encoding="utf-8")
    for match in re.finditer(r"^\s*(problems\s*=.*|case\[\"problem\"\].*|results\s*=\s*solution\.are_robust.*)$", source, re.M):
        print("   ", match.group(1).strip())
    hit = re.search(r'case\["problem"\]\["(\w+)"\]', source)
    if hit:
        print()
        print(f"    -> problem 딕셔너리에서 꺼내는 키는 {hit.group(1)!r} 하나뿐이다.")
        print("       permutation_type은 서버에서 솔루션까지 도달하지 않는다.")
        print("       로컬 cases.jsonl에는 들어 있으니 학습 특징으로 새어들기 쉽다.")


if __name__ == "__main__":
    main()
