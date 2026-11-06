"""15개 컬럼의 값 분포를 찍어본다 — 각 필드가 실제로 무엇을 담는지 확인용."""

import io
import json
import urllib.request
from collections import Counter

import pandas as pd

URL = (
    "https://huggingface.co/datasets/aimo-interp/val-sample/resolve/"
    "1ae454ec1fad9727084eda8f9f3c9ae2239b21de/data/validation-00000-of-00001.parquet"
)


def main() -> None:
    frame = pd.read_parquet(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))

    print("=== 식별자 ===")
    print("dataset_id:", dict(Counter(frame.dataset_id)))
    print("problem_id 고유:", frame.problem_id.nunique())
    print("model_id 고유:", frame.model_id.nunique())
    print()

    print("=== permutation_source 값 ===")
    sources = Counter()
    for value in frame.permutation_source:
        parsed = json.loads(value) if value.startswith("[") else [value]
        sources.update(parsed)
    for name, count in sources.most_common():
        print(f"  {name:34s} {count}")
    print()

    print("=== 실전에서 받는 것 vs 못 받는 것 ===")
    given = {"model_id", "original_problem"}
    for column in frame.columns:
        mark = "받음" if column in given else "  ·"
        print(f"  {mark}  {column}")


if __name__ == "__main__":
    main()
