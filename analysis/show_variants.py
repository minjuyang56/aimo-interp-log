import io
import json
import urllib.request

import pandas as pd

URL = (
    "https://huggingface.co/datasets/aimo-interp/val-sample/resolve/"
    "1ae454ec1fad9727084eda8f9f3c9ae2239b21de/data/validation-00000-of-00001.parquet"
)

frame = pd.read_parquet(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
row = frame.iloc[12]

print("MODEL:", row.model_id)
print("base_accuracy", row.base_accuracy, "-> permuted_accuracy", row.permuted_accuracy)
print()
print("[원본 문제]")
print(row.original_problem)
print()

variants = json.loads(row.permutations_causing_decay)
seen = set()
for item in variants:
    kind = item["permutation_type"]
    if kind in seen:
        continue
    seen.add(kind)
    print("[{}]  변형본 정확도 {}  (n={})".format(
        kind, item["variant_accuracy"], item["n_predictions"]
    ))
    print(item["question"][:340])
    print()
