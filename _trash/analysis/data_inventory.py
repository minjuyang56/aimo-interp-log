"""두 저장소의 데이터 파일 5개를 전수 열거하고 스키마·분포를 찍는다.

한 번에 전부 보는 용도. 개별 발견은 label_rule.py / leakage_permutation_type.py /
perturbed_text_bias.py 로 쪼개 두었다.
"""

import csv
import json
from collections import Counter

from _paths import AGG_CSV, CASES, FINAL_CSV, LABELS, METADATA, require


def infer_type(value):
    if value == "":
        return "empty"
    try:
        int(value)
        return "int"
    except ValueError:
        pass
    try:
        float(value)
        return "float"
    except ValueError:
        pass
    if value.lower() in ("true", "false"):
        return "bool-str"
    return "str"


def show_jsonl(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    schema = "v2 (problem=str)" if isinstance(rows[0].get("problem"), str) else "v1 (problem=dict)"
    print(f"## {path.name}  ({path.stat().st_size:,} B, {len(rows)}행, schema {schema})")

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            print(f"   {prefix:28s} list[{len(obj)}]  {str(obj[:2])[:60]}")
        else:
            text = str(obj).replace("\n", "\\n")
            print(f"   {prefix:28s} {type(obj).__name__:8s} {text[:60]}")

    walk(rows[0])
    print()


def show_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    print(f"## {path.name}  ({path.stat().st_size:,} B, {len(rows)}행, {len(rows[0])}컬럼)")
    for key, value in rows[0].items():
        values = [r[key] for r in rows]
        uniq = set(values)
        text = value.replace("\n", "\\n")
        if len(text) > 50:
            text = text[:50] + f"...({len(value)}자)"
        print(f"   {key:28s} {infer_type(value):8s} 고유{len(uniq):4d}  {text}")
        if len(uniq) <= 8:
            print(f"   {'':28s} {'':8s} {'':8s} {dict(Counter(values).most_common())}")
    print()


def main():
    require(CASES, LABELS, METADATA, FINAL_CSV, AGG_CSV)
    print(f"## {METADATA.name}")
    print("   " + METADATA.read_text(encoding="utf-8").replace("\n", "\n   "))
    show_jsonl(CASES)
    show_jsonl(LABELS)
    show_csv(FINAL_CSV)
    show_csv(AGG_CSV)


if __name__ == "__main__":
    main()
