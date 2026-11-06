"""변형된 문제 텍스트가 데이터 어디에, 어떤 편향으로 들어 있는지 확인한다.

cases.jsonl에는 original_problem만 있다. 변형본 텍스트는 CSV의
permutations_causing_decay 컬럼(JSON 배열 문자열)에만 있다.

컬럼 이름이 곧 편향이다: "decay를 일으킨 변형"만 담는다.
robust 케이스에서 변형본이 얼마나 남아 있는지 세어 보면 학습 가능 여부가 갈린다.
"""

import csv
import json
from collections import Counter

from _paths import CASES, FINAL_CSV, require


def as_bool(text):
    return text.strip().lower() == "true"


def main():
    require(CASES, FINAL_CSV)

    # cases.jsonl: 키 전수 -----------------------------------------------------
    keys = set()

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        else:
            keys.add(prefix)

    for line in CASES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            walk(json.loads(line))
    print("[1] cases.jsonl 전체 키:", sorted(keys))
    print("    -> 변형 텍스트 없음. 문제 원문 하나뿐."
          + (" (schema v2: problem이 문자열)" if "problem" in keys else ""))

    # CSV: permutations_causing_decay ----------------------------------------
    with FINAL_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    parsed = [(row, json.loads(row["permutations_causing_decay"])) for row in rows]
    total = sum(len(items) for _, items in parsed)
    first = next(items[0] for _, items in parsed if items)

    print()
    print("[2] CSV permutations_causing_decay")
    print(f"    변형 텍스트 총 개수: {total}")
    print(f"    항목 키           : {list(first.keys())}")
    print(f"    예시 question     : {first['question'][:100]}")

    by_type = Counter(item["permutation_type"] for _, items in parsed for item in items)
    print()
    print("    유형별 개수:")
    for name, count in by_type.most_common():
        print(f"      {name:10s} {count:4d}  ({count / total:5.1%})")

    # 편향 ----------------------------------------------------------------------
    n_true = sum(1 for row, _ in parsed if as_bool(row["model_is_robust"]))
    n_true_with_text = sum(
        1 for row, items in parsed if as_bool(row["model_is_robust"]) and items
    )
    n_false = len(rows) - n_true
    n_false_with_text = sum(
        1 for row, items in parsed if not as_bool(row["model_is_robust"]) and items
    )

    print()
    print("[3] 라벨별 변형 텍스트 보유 행")
    print(f"    robust=True  : {n_true_with_text:3d}/{n_true}")
    print(f"    robust=False : {n_false_with_text:3d}/{n_false}")
    if n_true_with_text == 0:
        print()
        print("    -> robust 쪽은 예외 없이 빈 배열이다.")
        print("       '이 변형은 견뎠다'는 예시가 데이터에 한 건도 없다.")
        print("       변형 텍스트로 분류기를 학습시키려면 음성 클래스가 없다.")


if __name__ == "__main__":
    main()
